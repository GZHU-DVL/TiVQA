import torch.nn.functional as F
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
from argparse import ArgumentParser
from skimage.feature import local_binary_pattern
from Features_Extraction import get_features
import time
class ANN(nn.Module):
    def __init__(self, input_size=8192, reduced_size=248, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)
    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=44,beta=0.52):  #temporal pooling
    q = torch.unsqueeze(torch.t(q), 0)   #frame level quality
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
    x = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)  #memory quality element
    y = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    y = y / n   #current quality element
    return beta * y + (1 - beta) * x


class TiVQA(nn.Module):
    def __init__(self, input_size=8192, reduced_size=248, hidden_size=64): 
        super(TiVQA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)
    def forward(self, input, input_length):
        input = self.ann(input)
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)
        for i in range(input_length.shape[0]):
            qi = q[i, :np.int(input_length[i].numpy())]
            qi = TP(qi,tau=args.tau,beta=args.beta)  #temporal pooling
            score[i] = torch.mean(qi)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

if __name__ == "__main__":
    parser = ArgumentParser(description='"TiVQA: Texture information boosts video quality assessment')
    parser.add_argument("--tau", type=int, default=44)
    parser.add_argument("--beta", type=int, default=0.52)
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument("--reduced_size", type=int, default=248)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument('--model', default='TiVQA', type=str, help='model name (default: TiVQA)')
    parser.add_argument('--video_format', default='RGB', type=str, help='video format: GRB or YUV420 (default: RGB)')
    parser.add_argument('--video_path', default='data/test.mp4', type=str, help='video path (default: data/test.mp4)')
    parser.add_argument('--video_width', type=int, default=None, help='video width')
    parser.add_argument('--video_height', type=int, default=None, help='video height')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    args = parser.parse_args()
    datainfo = './data/KoNViD-1kinfo.mat'
    trained_model_file = './models/TiVQA-KoNViD-1k-44--0.52--16--248--64--LBP_10_4'
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    start = time.time()
    LBP = torch.Tensor()
    LBP = LBP.to(device)
    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    scale = Info['scores'][0, :].max()  # label normalization factor
    model = TiVQA(reduced_size=args.reduced_size,hidden_size=args.hidden_size).to(device)  # Initial TiVQA
    assert args.video_format == 'YUV420' or args.video_format == 'RGB'
    if args.video_format == 'YUV420':
        video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(args.video_path)
        for i in range(video_data.shape[0]):
            img = cv2.cvtColor(video_data[i], cv2.COLOR_BGR2GRAY)
            middle = local_binary_pattern(img, 10, 4)
            middle = torch.from_numpy(middle)
            middle = middle.to(device)
            middle = torch.unsqueeze(middle, 2)
            middle = torch.cat((middle, middle, middle), dim=2)
            middle = middle.unsqueeze(0)
            LBP = torch.cat((LBP, middle), 0)  # detect texture information
    LBP = LBP.cpu().numpy()
    LBP = np.asarray(LBP)
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
    model.load_state_dict(torch.load(trained_model_file, map_location='cpu'))  # load TiVQA
    model.eval()
    length = []
    with torch.no_grad():
        y_pred = np.zeros(1)
        features = get_features(transformed_video,transformed_LBP,frame_batch_size=1)  # get the content-aware and texture features
        length.append(video_length)                  #video length
        length = torch.tensor(length)
        features = torch.unsqueeze(features, 0)
        length = torch.unsqueeze(length, 0)
        outputs = model(features,length.float())
        y_pred = scale * outputs.item()
        print("Predicted perceptual quality: {}".format(y_pred))
    end = time.time()
    print('Time: {} s'.format(end - start))
