from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
class VQADataset(Dataset):
    def __init__(self, features_dir='/', index=None, max_len=830, feat_dim=8192, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir  + str(index[i]) +'_resnet-50_res5c.npy')  #load the content-aware and texture features
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) +'_score.npy')
        self.scale = scale
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class ANN(nn.Module):  #Fully connected
    def __init__(self, input_size=8192, reduced_size=248, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)
        for i in range(self.n_ANNlayers-1):
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
        self.ann = ANN(input_size, reduced_size, 1)  #FC
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)  #GRU
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        input = self.ann(input)
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)
        for i in range(input_length.shape[0]):  #
            qi = q[i, :np.int(input_length[i].numpy())]
            qi = TP(qi,tau=args.tau,beta=args.beta)  #Temporal pooling
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
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train (default: 2000)')
    parser.add_argument('--database', default='KoNViD-1k', type=str, help='database name (default: KoNViD-1k)')
    parser.add_argument('--model', default='TiVQA', type=str, help='model name (default: TiVQA)')
    parser.add_argument('--exp_id', default=0, type=int, help='exp id for train-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument("--notest_during_training", action='store_true', help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true', help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs", help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    args = parser.parse_args()
    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        features_dir = 'LBP_P10_R4_std_CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'
    if args.database == 'CVD2014':
        features_dir = 'LBP_P10_R4_std_CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-VQC':
        features_dir = 'LBP_P10_R4_std_CNN_features_LIVE-VQC/'
        datainfo = 'data/LIVE-VQC.mat'
    if args.database == 'LSVQ':
        features_dir = 'LBP_P10_R4_std_CNN_features_LSVQ/'
        datainfo1 = '/data/aoxiang/LSVQ/labels_train_test.csv'
        datainfo2 = '/data/aoxiang/LSVQ/labels_test_1080p.csv'

    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    Info = h5py.File(datainfo, 'r')
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]
    ref_ids = Info['ref_ids'][0, :]
    max_len = int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1 - args.test_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, test_index = [], []
    for i in range(len(ref_ids)):
        if ref_ids[i] in trainindex:
            train_index.append(i)
        else:
            test_index.append(i)
    scale = Info['scores'][0, :].max()
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    print(len(train_index))
    print(len(test_index))
    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    model = TiVQA(reduced_size=args.reduced_size,hidden_size=args.hidden_size).to(device)  #

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-{}--{}--{}--{}--{}--{}'.format(args.model, args.database, args.tau,args.beta, args.batch_size,args.reduced_size,args.hidden_size,'LBP_10_4')
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-{}--{}--{}--{}--{}--{}'.format(args.model, args.database, args.tau,args.beta, args.batch_size,args.reduced_size,args.hidden_size,'LBP_10_4')

    if not args.disable_visualization:
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}'.format(args.log_dir, args.exp_id, args.database, args.model,
                                       args.lr, args.batch_size, args.epochs,
                                       datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    criterion = nn.L1Loss()  #Loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_test_criterion = -1
    for epoch in range(args.epochs):
        print(epoch)
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)
        model.eval()

        if args.test_ratio > 0 and not args.notest_during_training:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.item()
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, length.float())
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]


        print("best_test_criterion",best_test_criterion)
        if SROCC > best_test_criterion:
            print("EXP ID={}: Update best model using best_test_criterion in epoch {}".format(args.seed, epoch))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                print("tau", args.tau, " beta", args.beta, " bs", args.batch_size)
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_test_criterion = SROCC  # update best test SROCC

    if args.test_ratio > 0:
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
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
