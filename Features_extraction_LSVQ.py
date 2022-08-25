import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import pandas as pd
import h5py
import numpy as np
import random
from argparse import ArgumentParser
from skimage.feature import local_binary_pattern
class VideoDataset(Dataset):

    def __init__(self, videos_dir, video_list, mos, video_format='RGB', width=None, height=None):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_list = video_list
        self.mos = mos
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        LBP = torch.Tensor()
        LBP = LBP.to(device)
        video_list = self.video_list[idx] +'.mp4'              #other dataset
        print(video_list)
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_list), self.height, self.width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_list))
            for i in range(video_data.shape[0]):
                img = cv2.cvtColor(video_data[i], cv2.COLOR_BGR2GRAY)
                middle = local_binary_pattern(img, 10, 4)
                middle = torch.from_numpy(middle)
                middle = middle.to(device)
                middle = torch.unsqueeze(middle, 2)
                middle = torch.cat((middle, middle, middle), dim=2)
                middle = middle.unsqueeze(0)
                LBP = torch.cat((LBP, middle), 0)              #detect texture information
        LBP = LBP.cpu().numpy()
        LBP = np.asarray(LBP)
        video_mos = self.mos[idx]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transformed_LBP = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame_video = video_data[frame_idx]
            frame_LBP = LBP[frame_idx]
            frame_video = Image.fromarray(frame_video)
            frame_LBP = Image.fromarray(np.uint8(frame_LBP))
            frame_video = transform(frame_video)
            frame_LBP = transform(frame_LBP)
            transformed_video[frame_idx] = frame_video
            transformed_LBP[frame_idx] = frame_LBP
        sample = {'video': transformed_video,
                  'mos': video_mos,
                  'LBP': transformed_LBP}
        return sample

class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        for ii, model in enumerate(self.features):
            x1 = model(x1)
            x2 = model(x2)
            if ii == 7:
                content_aware_mean_features = nn.functional.adaptive_avg_pool2d(x1, 1)   #extract content-aware features
                texture_mean_features = nn.functional.adaptive_avg_pool2d(x2, 1)         #extract texture features
                content_aware_std_features = global_std_pool2d(x1)
                texture_std_features = global_std_pool2d(x2)
                return content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, video_LBP, frame_batch_size=16, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    output3 = torch.Tensor().to(device)
    output4 = torch.Tensor().to(device)
    extractor.eval()

    with torch.no_grad():
        while frame_end < video_length:
            batch_video = video_data[frame_start:frame_end].to(device)
            batch_LBP = video_LBP[frame_start:frame_end].to(device)
            content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features = extractor(batch_video, batch_LBP)
            output1 = torch.cat((output1, content_aware_mean_features), 0).to(device)    #content_aware featuers
            output2 = torch.cat((output2, content_aware_std_features), 0).to(device)
            output3 = torch.cat((output3, texture_mean_features), 0).to(device)                 #texture features
            output4 = torch.cat((output4, texture_std_features), 0).to(device)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        last_LBP = video_LBP[frame_start:frame_end].to(device)
        content_aware_mean_features, content_aware_std_features, texture_mean_features, texture_std_features = extractor(last_batch, last_LBP)
        output1 = torch.cat((output1, content_aware_mean_features), 0).to(device)
        output2 = torch.cat((output2, content_aware_std_features), 0).to(device)
        output3 = torch.cat((output3, texture_mean_features), 0).to(device)
        output4 = torch.cat((output4, texture_std_features), 0).to(device)
        output = torch.cat((output1, output2, output3, output4), 1).to(device)  #concate texture and content-aware features
        output = output.squeeze()
        print(output)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features and Texture Features')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='LSVQ', type=str, help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=16, help='frame batch size for feature extraction (default: 32)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'LSVQ':
        videos_dir = 'LSVQ/'  # videos dir
        features_dir = 'LBP_P10_R4_std_CNN_features_LSVQ/'  # features dir
        datainfo1 = '/data/aoxiang/LSVQ/labels_train_test.csv'
        datainfo2 = '/data/aoxiang/LSVQ/labels_test_1080p.csv'



    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info1 = pd.read_csv(datainfo1, encoding='gbk', nrows=17223)
    Info2 = pd.read_csv(datainfo2, encoding='gbk', nrows=2794)
    Info3 = pd.read_csv(datainfo1, encoding='gbk', skiprows=17223)
    Info3.columns = ['name', 'height', 'width', 'mos', 'frame_number']
    Info4 = pd.read_csv(datainfo2, encoding='gbk', skiprows=2794)
    Info4.columns = ['name', 'p1', 'p2', 'p3', 'height', 'width', 'mos_p1', 'mos_p2', 'mos_p3', 'mos', 'frame_number',
                     'fn_last_frame', 'left_p1', 'right_p1', 'top_p1', 'bottom_p1', 'start_p1', 'end_p1', 'left_p2',
                     'right_p2', 'top_p2', 'bottom_p2', 'start_p2', 'end_p2', 'left_p3', 'right_p3', 'top_p3',
                     'bottom_p3', 'start_p3', 'end_p3', 'top_vid', 'left_vid', 'bottom_vid', 'right_vid', 'start_vid',
                     'end_vid', 'is_valid']

    merge = [Info1, Info2, Info3, Info4]
    Info = pd.concat(merge, ignore_index=True)
    mos = Info['mos']
    width = Info['width']
    height = Info['height']
    video_list = Info['name']
    video_format = 'RGB'

    dataset = VideoDataset(videos_dir, video_list, mos, video_format, width, height)
    skip = [17157, 17158, 17159, 17162, 17163, 17165, 17167, 17168, 17169, 17170, 17177, 17179, 17181, 17182, 17184,  #Not in the LSVQ dataset
            17185, 17186, 17187, 17188, 17189, 17190, 17192, 17193, 17194, 17195, 17196, 17197, 17198, 17201, 17202,
            17204, 17205, 17208, 17209, 17211, 17212, 17214, 17215, 17216, 17217, 17218, 17221, 17222, 20008, 20009,
            20010, 20011, 20012, 20014, 20015, 38036, 38039, 38040, 38043, 38045, 38093, 38114, 38128, 38171, 38183,
            38184, 38218, 38242, 38245, 38289, 38290]


    for i in range(len(dataset)):
        if i not in skip:
            current_data = dataset[i]
            current_LBP = current_data['LBP']
            current_video = current_data['video']
            current_mos = current_data['mos']
            print('Video {}: length {}'.format(i, current_video.shape[0]))
            features = get_features(current_video, current_LBP, args.frame_batch_size, device)
            print("features", features.shape)
            np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
            np.save(features_dir + str(i) + '_score', current_mos)
