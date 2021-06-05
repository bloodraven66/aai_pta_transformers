import warnings
import librosa
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)#remove numpy warnings
import numpy as np
import scipy.io
import csv
import os, time
import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import to_categorical
import random
import load_and_log
import model
import models
import argparse
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import early_stopper
from keras.utils import to_categorical
import logger
from shutil import copyfile

def parse_args(parser):

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=500)
    training.add_argument('--device', default='cuda:0')
    training.add_argument('--cudnn-benchmark', default=True)
    training.add_argument('--desc', type='str', default='exp') #describe any details of exp
    training.add_argument('--log_path', type='str', default='logs/exp1.csv')
    training.add_argument('--exp_results', type='str', default='logs/exp1_all_values.csv')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--optimizer', type=str, default='adam')
    optimization.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    optimization.add_argument('--weight-decay', default=1e-6, type=float)
    optimization.add_argument('-batch_size', '--h-size', type=int, default=4,

    setup = parser.add_argument_group('experimental setup')
    setup.add_argument('--setup', type=str, required=True) #can be ss, pooled or ft (ss:single subject, ft: fine tuned)
    setup.add_argument('--checkpoint', type=str, required=True)
    setup.add_argument('--pool_checkpoint', type=str, required=False)#trained pooled model for ft
    setup.add_argument('--model_desc', type=str, default='Transformer_AAI')#desc name to select model
    return parser


def iter_(dataloader, train=True):
    if train: net.train();
    else: net.eval();
    step_loss = []
    for ema, mfcc in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        ema, mfcc = ema.to(args.device), mfcc.to(args.device)
        ema_out, dec_mask = net(inputs=mfcc.float(), dur_tgt=None)
        loss = loss_fn(ema_out.float(), ema.float())
        if train:
            loss.backward()
            optimizer.step()
        step_loss.append(loss.item())
    return sum(step_loss)/len(step_loss), step_loss

def get_cc(loader):
    net.eval()
    ema_, pred_ = [], []
    for ema, mfcc in tqdm(loader):
        ema, mfcc = ema.to(args.device), mfcc.to(args.device)
        pred = net(mfcc.float(), dur_tgt=None)[0].float().permute(0, 2, 1)
        ema_.extend(ema.float().detach().cpu().permute(0, 2, 1).tolist())
        pred_.extend(pred.detach().cpu().tolist())
    ema_ = np.array(ema_, dtype=np.float)
    pred_ = np.array(pred_, dtype=np.float)
    m = []
    for j in range(len(pred_)):
        c  = []
        for k in range(12):
            c.append(scipy.stats.pearsonr(ema_[j][k], pred_[j][k])[0])
        m.append(sum(c)/len(c))
    return round(sum(m)/len(m), 3)

def main():
    sub_loss  = {}
    sub_cc = {}
    subjects = []
    parser = argparse.ArgumentParser(description='AAI Training', allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    desc = args.model_desc
    parser = models.parse_model_args(desc, parser)
    args, unk_args = parser.parse_known_args()
    model_config = models.get_model_config(desc, args)
    pool_model = args.pool_checkpoint
    if args.setup not in ['ss', 'ft', 'pooled']:
        raise Exception('experimental setup not found')
    total_runs = 1 if args.setup == 'pooled' else len(subjects)
    for iter_num in range(total_runs):
        subject = 'all' if args.setup == 'pooled' else subjects[iter_num]
        net = models.get_model(desc, model_config, args.device).float()
        if args.setup = 'ft':
            try:
                net.load_state_dict(torch.load(pool_model))
            except:
                raise Exception('Failed to load pooled model')
        kw = dict(lr=args.learning_rate, betas=(0.9, 0.90), eps=1e-9,
                  weight_decay=args.weight_decay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), **kw)
        Youtval, X_valseq, val_lengths, Youttrain, X_trainseq, train_lens = load_and_log.load_data(mode=args.setup, iter_num, subjects)

        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youttrain), torch.from_numpy(X_trainseq))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youtval), torch.from_numpy(X_valseq))
        valloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        min_loss = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor=0.5)
        modelname = args.checkpoint
        early_stopping = early_stopper.EarlyStopping(patience=11, verbose=True, path=modelname, min_run=1, delta=0.001)
        start_time = time.time()
        train_step_loss_arr, val_step_loss_arr = [], []
        train_epoch_loss_arr, val_epoch_loss_arr = [], []
        pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(pytorch_total_params_trainable)
        for i in range(args.epochs):
            a = time.time()
            trainloss, train_step_loss = iter_(dataloader)
            valloss, val_step_loss = iter_(valloader, train=False)
            train_step_loss_arr.extend(train_step_loss)
            val_step_loss_arr.extend(val_step_loss)
            train_epoch_loss_arr.append(trainloss)
            val_epoch_loss_arr.append(valloss)
            if valloss<min_loss:
                min_loss = valloss
                if i>5:
                    get_cc(dataloader)
                    get_cc(valloader)
            scheduler.step(valloss)
            early_stopping(valloss, net, i)
            print('min:', min_loss)
            if early_stopping.early_stop:
                print("Early stopping at epoch ",i)
                print('loss:', min_loss)
                end_time = time.time() - start_time
                break
            end_time = time.time() - start_time
        net.load_state_dict(torch.load(modelname))
        tr = get_cc(dataloader)
        v = get_cc(valloader)
        sub_cc[subject] = [tr, v]
        sub_loss[subject] = min_loss
        with open(args.exp_results, 'wb') as f:
            np.save(f, [train_epoch_loss, val_epoch_loss, train_step_loss, val_step_loss, model_config], allow_pickle=True)
        logger.save(iter_num, args.desc, subject, modelname, min_loss, v, i,args.log_path,end_time, pytorch_total_params_trainable, model_config, args.exp_results)
        print(args.traindesc, sub_cc, sub_loss)
