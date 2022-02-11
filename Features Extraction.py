import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
from skimage.feature import local_binary_pattern

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        if self.video_list[idx][11] == '_':
            video_list = self.video_list[idx][0:11] + '.mp4'
        elif self.video_list[idx][10] == '_':
            video_list = self.video_list[idx][0:10] + '.mp4'
        else:
            video_list = self.video_list[idx][0:9] + '.mp4'
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
                LBP = torch.cat((LBP, middle), 0)
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
        for p in self.features.parameters():  # 获取网络的参数
            p.requires_grad = False

    def forward(self, x1, x2):
        for ii, model in enumerate(self.features):
            x1 = model(x1)
            x2 = model(x2)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x1, 1)
                x2_mean = nn.functional.adaptive_avg_pool2d(x2, 1) #LBP_mean_features
                features_std = global_std_pool2d(x1)
                x2_std = global_std_pool2d(x2) #LBP_std_features
                return features_mean, features_std, x2_mean, x2_std


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
            features_mean, features_std, LBP_mean_features, LBP_std_features = extractor(batch_video, batch_LBP)
            output1 = torch.cat((output1, features_mean), 0).to(device)
            output2 = torch.cat((output2, features_std), 0).to(device)
            output3 = torch.cat((output3, LBP_mean_features), 0).to(device)
            output4 = torch.cat((output4, LBP_std_features), 0).to(device)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        last_LBP = video_LBP[frame_start:frame_end].to(device)
        features_mean, features_std, LBP_mean_features, LBP_std_features = extractor(last_batch, last_LBP)
        output1 = torch.cat((output1, features_mean), 0).to(device)
        output2 = torch.cat((output2, features_std), 0).to(device)
        output3 = torch.cat((output3, LBP_mean_features), 0).to(device)
        output4 = torch.cat((output4, LBP_std_features), 0).to(device)
        output = torch.cat((output1, output2, output3, output4), 1).to(device)
        output = output.squeeze()
        print(output)
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features and Texture Features')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 32)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = '/home/zax/KoNViD-1k_video/'  # videos dir
        features_dir = 'LBP_P10_R4_std_CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = '/home/zax/CVD2014/'
        features_dir = 'LBP_P10_R4_std_CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Quaclomm':
        videos_dir = '/home/zax/LIVE-Quaclomm/'
        features_dir = 'LBP_P10_R4_std_CNN_features_LIVE-Quaclomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)

    for i in range(len(dataset)):
        current_data = dataset[i]
        current_LBP = current_data['LBP']
        current_video = current_data['video']
        current_mos = current_data['mos']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, current_LBP, args.frame_batch_size, device)
        print("features", features.shape)
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_mos)
