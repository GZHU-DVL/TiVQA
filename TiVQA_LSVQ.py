from argparse import ArgumentParser
import os
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
import pandas as pd

def split(full_list, shuffle=False, ratio=0.8):
    n_total = len(full_list)
    train_len = int(n_total * ratio)
    if shuffle:
        random.shuffle(full_list)
    train = full_list[:train_len + 1]
    test = full_list[train_len + 1:]
    return train, test


class VQADataset(Dataset):
    def __init__(self, features_dir='/', index=None, max_len=240, feat_dim=8192, scale=1, way=''):
        super(VQADataset, self).__init__()
        self.mos = np.zeros((len(index), 1))
        self.way = way
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        features = np.zeros((max_len, self.feat_dim))
        if self.way == 'train':
            feature = np.load(features_dir + str(int(train_index[idx])) + '_resnet-50_res5c.npy')
            mos = np.load(features_dir + str(int(train_index[idx])) + '_score.npy')
            features[:feature.shape[0], :] = feature
        else:
            feature = np.load(features_dir + str(int(test_index[idx])) + '_resnet-50_res5c.npy')
            mos = np.load(features_dir + str(int(test_index[idx])) + '_score.npy')
            features[:feature.shape[0], :] = feature
        length = feature.shape[0]
        label = mos / scale
        sample = features, length, label
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=8192, reduced_size=208, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)  #
    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=20,beta=0.5):
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class TiVQA(nn.Module):
    def __init__(self, input_size=8192, reduced_size=208, hidden_size=56):
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
        for i in range(input_length.shape[0]):  #
            qi = q[i, :np.int(input_length[i].numpy())]
            qi = TP(qi)
            score[i] = torch.mean(qi)
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == "__main__":
    parser = ArgumentParser(description='"TiVQA: Texture Information Boosts Video Quality Assessment')
    parser.add_argument('--tau', type=int, default=20)
    parser.add_argument('--beta', type=int, default=0.5)
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16 ,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--database', default='LSVQ', type=str,
                        help='database name (default: LSVQ)')
    parser.add_argument('--model', default='TiVQA', type=str,
                        help='model name (default: TiVQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'LSVQ':
        features_dir = '/data/aoxiang/TiVQA/LBP_P10_R4_CNN_features_LSVQ/'
        datainfo1 = './data/labels_train_test.csv'
        datainfo2 = './data/labels_test_1080p.csv'
        trained_model_file = './models/TiVQA-KoNViD-1k-EXP44--0.52--16--248_64--LBP_10_4'

    print('EXP ID: {}'.format(args.seed))
    print(args.database)
    print(args.model)
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print("device",device)

    Info1 = pd.read_csv(datainfo1, encoding='gbk', nrows=17223)
    Info2 = pd.read_csv(datainfo2, encoding='gbk', nrows=2794)
    Info3 = pd.read_csv('/home/zax/labels_train_test.csv', encoding='gbk', skiprows=17223)
    Info3.columns = ['name', 'height', 'width', 'mos', 'frame_number']
    Info4 = pd.read_csv('/home/zax/labels_test_1080p.csv', encoding='gbk', skiprows=2794)
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
    max_len = max(Info['frame_number'])
    scale = max(Info['mos'])
    skip = [17157, 17158, 17159, 17162, 17163, 17165, 17167, 17168, 17169, 17170, 17177, 17179, 17181, 17182, 17184,
            17185, 17186, 17187, 17188, 17189, 17190, 17192, 17193, 17194, 17195, 17196, 17197, 17198, 17201, 17202,
            17204, 17205, 17208, 17209, 17211, 17212, 17214, 17215, 17216, 17217, 17218, 17221, 17222, 20008, 20009,
            20010, 20011, 20012, 20014, 20015, 38036, 38039, 38040, 38043, 38045, 38093, 38114, 38128, 38171, 38183,
            38184, 38218, 38242, 38245, 38289, 38290]
    index = []
    for i in range(39072):
        if i not in skip:
            index.append(i)

    train_index,test_index = split(index, shuffle=True, ratio=0.8)
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale,way = 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale,way = 'test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    model = TiVQA().to(device)  #
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    model.load_state_dict(torch.load(trained_model_file, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(len(test_index))
        y_test = np.zeros(len(test_index))
        L = 0
        for i, (features, length, label) in enumerate(test_loader):
            y_test[i] = scale * label.item()  #
            features = features.to(device).float()
            label = label.to(device).float()
            outputs = model(features, length.float())
            y_pred[i] = scale * outputs.item()
            loss = criterion(outputs, label)
            L = L + loss.item()
    test_loss = L / (i + 1)
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
          .format(test_loss, SROCC, KROCC, PLCC, RMSE))
