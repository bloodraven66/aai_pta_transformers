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
from tqdm import tqdm
import early_stopper
from keras.utils import to_categorical
import logger
import utils

def parse_args(parser):

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=100)
    training.add_argument('--select_device', default='cuda:0')
    training.add_argument('--device', default=select_device)
    training.add_argument('--cudnn-benchmark', default=True)
    training.add_argument('--verbose', default=True)
    training.add_argument('--desc', type='str', default='exp') #describe any details of exp
    training.add_argument('--log_path', type='str', default='logs/exp1.csv')
    training.add_argument('--exp_results', type='str', default='logs/exp1_all_values.csv')
    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--optimizer', type=str, default='adam')
    optimization.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    optimization.add_argument('--weight-decay', default=1e-6, type=float)
    optimization.add_argument('-bs', '--h-size', type=int, default=4)

    setup = parser.add_argument_group('experimental setup')
    setup.add_argument('--setup', type=str, required=True) #can be ss, pooled or ft (ss:single subject, ft: fine tuned)
    setup.add_argument('--checkpoint', type=str, required=True)
    setup.add_argument('--pool_checkpoint', type=str, required=False)#trained pooled model for ft
    setup.add_argument('--model_desc', type=str, default='FastPitch')#desc name to select model
    return parser

def main():
    sub_loss  = {}
    sub_cc = {}
    subjects = [] #list of subject_names in dataset
    parser = argparse.ArgumentParser(description='Train PTA', allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    model_desc = args.model_desc
    parser = models.parse_model_args(model_desc, parser)
    args, unk_args = parser.parse_known_args()
    model_config = models.get_model_config(model_desc, args)
    pool_model = args.pool_checkpoint
    if args.setup not in ['ss', 'ft', 'pooled']:
        raise Exception('experimental setup not found')
    total_runs = 1 if args.setup == 'pooled' else len(subjects)
    for iter_num in range(total_runs):
        subject = 'all' if args.setup == 'pooled' else subjects[iter_num]
        net = models.get_model(model_desc, model_config, args.device).float()
        if args.setup = 'ft':
            try:
                net.load_state_dict(torch.load(pool_model))
            except:
                raise Exception('Failed to load pooled model')
        kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
                    weight_decay=args.weight_decay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), **kw)
        Youtval, phon_valseq, phon_val_dur, val_lengths, Youttrain, phon_trainseq, phon_train_dur, train_lens = load_and_log.load_data(mode=args.setup, iter_num, subjects)
        max_len = 60
        phon_trainseq=pad_sequences(phon_trainseq, padding='post',maxlen=max_len,dtype='float', value=40.0)
        phon_valseq=pad_sequences(phon_valseq, padding='post',maxlen=max_len,dtype='float', value=40.0)
        phon_train_dur=pad_sequences(phon_train_dur, padding='post',maxlen=max_len,dtype='float')
        phon_val_dur=pad_sequences(phon_val_dur, padding='post',maxlen=max_len,dtype='float')
        loss_fn = nn.MSELoss()
        loss_fn_dur = nn.MSELoss()

        #If you face memory issues, use custom dataset class to load only 1 batch at a time
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youttrain), torch.from_numpy(phon_trainseq), torch.from_numpy(phon_train_dur), torch.from_numpy(train_lens))
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(Youtval), torch.from_numpy(phon_valseq), torch.from_numpy(phon_val_dur), torch.from_numpy(val_lens))
        valloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
        min_loss = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor=0.6)
        modelname = args.checkpoint
        early_stopping = early_stopper.EarlyStopping(patience=11, verbose=True, path=modelname)
        start_time = time.time()
        pytorch_total_params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('num of params:',pytorch_total_params_trainable)
        train_epoch_loss, val_epoch_loss, train_step_loss, val_step_loss = [], [], [], []
        for i in range(args.epochs):
            tloss, tloss_e, tloss_dur, tr_step_loss = utils.iter_(args, net, trainloader, optimizer, train=True, use_speaker_embed=use_speaker_embed, loss_fn=loss_fn, loss_fn_dur=loss_fn_dur)
            vloss, vloss_e, vloss_dur, v_step_loss = utils.iter_(args, net, valloader, optimizer=None, train=False, use_speaker_embed=use_speaker_embed, loss_fn=loss_fn, loss_fn_dur=loss_fn_dur)
            if args.verbose:
                print(subject, i, round(tloss, 2), round(tloss_e, 2), round(tloss_dur, 2), round(vloss, 2), round(vloss_e, 2), round(vloss_dur, 2))
            train_epoch_loss.append(tloss)
            val_epoch_loss.append(vloss)
            train_step_loss.extend(tr_step_loss)
            val_step_loss.extend(v_step_loss)

            if vloss<min_loss:
                min_loss = vloss
                if i>5:
                    utils.get_cc(args, net, trainloader)
                    utils.get_cc(args, net, valloader)

            scheduler.step(vloss)
            early_stopping(vloss, net, i)

            if early_stopping.early_stop:
                print("Early stopping at epoch ",i)
                print('loss:', min_loss)
                end_time = time.time() - start_time
                break
            end_time = time.time() - start_time
        net.load_state_dict(torch.load(modelname))
        tr = utils.get_cc(args, net, trainloader)
        v = utils.get_cc(args, net, valloader)
        sub_cc['subject'] = [tr, v]
        sub_loss['subject'] = min_loss
        with open(args.exp_results, 'wb') as f:
            np.save(f, [train_epoch_loss, val_epoch_loss, train_step_loss, val_step_loss, model_config], allow_pickle=True)
        logger.save(iter_num, args.desc, subject, modelname, min_loss, v, i, args.log_path ,end_time, pytorch_total_params_trainable, args.exp_results)
        print(args.traindesc, sub_cc, sub_loss)
