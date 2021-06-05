import argparse
import models
import time
from tqdm import tqdm
import sys
import warnings
from pathlib import Path
import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import load_and_log
import utils
import matplotlib.pyplot as plt

def parse_args(parser):
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='Models/.pth')
    parser.add_argument('--desc', default='FastPitch')
    return parser


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))
    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model.to(device)



def iter_(dataloader):
    net.eval()
    coefficients=[]
    rMSE=[]
    for ema, phon, dur, ema_lengths, sp, dur_lens in tqdm(dataloader):
        ema, phon, dur = ema.detach().cpu().numpy(), phon.to(args.device), dur.to(args.device)
        ema_out, dec_len, dur_pred, phon_lens = net(phon.float(), None)
        phon = phon.detach().cpu().tolist()
        dur_pred = dur_pred.detach().cpu().tolist()
        dur = dur.detach().cpu().tolist()
        ema_out = ema_out.permute(0, 2, 1).detach().cpu().numpy()
        for i in range(ema_out.shape[0]):
            X = ema[i][:ema_lengths[i]]
            Y = ema_out[i][:dec_len[i]]
            dis, pth = fastdtw(X,Y , dist=euclidean)
            coefficients_, rMSE_ = [], []
            for artic in range(0,12):
                out, gt = [], []
                for i in range(0,len(pth)):
                    out.append(Y[pth[i][1]][artic])
                    gt.append(X[pth[i][0]][artic])
                coef=pearsonr(out,gt)[0]
                coefficients_.append(coef)
                rMSE_.append(np.sqrt(np.mean(np.square(np.asarray(out)-np.asarray(gt)))))
            coefficients.append(coefficients_)
            rMSE.append(rMSE_)
    cc = np.mean(np.array(coefficients), axis=0)
    rmse = np.mean(np.array(rMSE), axis=0)
    cc = sum(cc)/len(cc)
    rmse = sum(rmse)/len(rmse)
    return cc, rmse, coefficients, rMSE

subjects = []
for sub_name in subjects:
        Youttest, phon_testseq, phon_test_dur, test_lengths = load_and_log.load_data(subjects, test_only=True)
        max_len = 60
        lengths = np.array(test_lengths)
        phon_testseq=pad_sequences(phon_testseq, padding='post',maxlen=max_len,dtype='float', value=40.0)
        phon_test_dur=pad_sequences(phon_test_dur, padding='post',maxlen=max_len,dtype='float')
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youttest), torch.from_numpy(phon_testseq), torch.from_numpy(phon_test_dur),torch.from_numpy(lengths), torch.from_numpy(np.array([subjects.index(sub_name) for i in range(len(phon_testseq))])), torch.from_numpy(dur_lens))
        valloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        parser = argparse.ArgumentParser(description='PTA test',
                                         allow_abbrev=False)
        parser = parse_args(parser)
        args, unk_args = parser.parse_known_args()
        torch.backends.cudnn.benchmark = True
        args.fastpitch = None
        desc = args.desc
        net = load_and_setup_model(
                desc, parser, args.fastpitch, args.amp, args.device,
                unk_args=unk_args, forward_is_infer=True, ema=args.ema,
                jitable=args.torchscript)
        st = time.time()
        cc, rmse, = iter_(valloader, speaker_embed=speaker_embed)
        cc = round(cc, 4)
        rmse = round(rmse, 4)
        metrics[sub_name] = [cc, rmse]

cc = np.array(list(metrics.values()))[:, 0]
rmse = np.array(list(metrics.values()))[:, 1]
print(cc)
print(rmse)
