import argparse
import models
import time
from tqdm import tqdm
import sys, os
import warnings
from pathlib import Path
import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from common import utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import scipy
import load_and_log
import torch.nn as nn

def parse_args(parser):
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='Models/.pth')
    parser.add_argument('--desc', default='Transformer_AAI')
    return parser



def iter_(dataloader):
    net.eval()
    ema_, pred_, loss = [], [], []
    for ema, mfcc, tlens, sp in tqdm(dataloader):
        ema, mfcc, sp = ema.to(args.device), mfcc.to(args.device), sp.to(args.device)
        pred = net(mfcc.float(), dur_tgt=None, speaker=sp)[0].float().permute(0, 2, 1)
        ema_.extend(ema.float().detach().cpu().permute(0, 2, 1).tolist())
        pred_.extend(pred.detach().cpu().tolist())

    ema_ = np.array(ema_, dtype=np.float)
    pred_ = np.array(pred_, dtype=np.float)
    m = []
    rMSE = []
    for j in range(len(pred_)):
        c  = []
        rmse = []
        for k in range(12):
            c.append(scipy.stats.pearsonr(ema_[j][k][:test_lens[j]], pred_[j][k][:test_lens[j]])[0])
            rmse.append(np.sqrt(np.mean(np.square(np.asarray(pred_[j][k][:test_lens[j]])-np.asarray(ema_[j][k][:test_lens[j]])))))
        m.append(c)
        rMSE.append(rmse)
    cc = np.mean(np.array(m), axis=0)
    rmse = np.mean(np.array(rMSE), axis=0)
    cc = sum(cc)/len(cc)
    rmse = sum(rmse)/len(rmse)
    return round(cc, 3), round(rmse, 3)

subjects = []


loss_fn = nn.MSELoss()
vals = {}
for sub_name in subjects:
    X_testseq, Youttest, test_lens = load_and_log.load_data(subjects, test_only=True)
    max_len = 60
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youttest), torch.from_numpy(X_testseq),torch.from_numpy(np.array(test_lens)), torch.from_numpy(np.array([subjects.index(sub_name) for i in range(len(Youtval))])))
    valloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    desc = args.desc
    parser = models.parse_model_args(desc, parser)
    args, unk_args = parser.parse_known_args()
    model_config = models.get_model_config(desc, args)
    net = models.get_model(desc, model_config, args.device).float()
    net.load_state_dict(torch.load(args.checkpoint))
    cc, rmse = iter_(valloader)
    vals[sub_name] = [cc, rmse]
print(np.array(list(vals.values()))[:, 0])
print(np.array(list(vals.values()))[:, 1])
