import numpy as np
import torch
from tqdm import tqdm
import scipy

def get_cc(args, net, loader):
    net.eval()
    m  = []
    for ema, phon, dur, dec_lens in tqdm(loader):
        ema, phon, dur = ema.to(args.device), phon.to(args.device), dur.to(args.device)
        ema_out, dec_mask, dur_pred, log_dur_pred, dec_lens = net(phon.float(), dur)
        pred = ema_out.float().permute(0, 2, 1)
        for j in range(len(pred)):
            ema_ = ema.float().detach().cpu().permute(0, 2, 1).tolist()
            pred_ = pred.detach().cpu().tolist()
            c = []
            for k in range(12):
                c.append(scipy.stats.pearsonr(ema_[j][k][:dec_lens[j]], pred_[j][k][:dec_lens[j]])[0])
            m.append(sum(c)/len(c))
    return round(sum(m)/len(m), 3)

def iter_(args, net, loader, optimizer, train=True, loss_fn=None, loss_fn_dur=None):
    if train: net.train()
    else: net.eval()
    loss_, ema_loss_, dur_loss_ = 0, 0, 0
    _loss = []
    for ema, phon, dur, speaker, dur_lengths in tqdm(loader):
        if train:
            optimizer.zero_grad()
        ema, phon, dur = ema.to(args.device), phon.to(args.device), dur.to(args.device)
        ema_out, dec_mask, dur_pred, log_dur_pred, dec_lens = net(inputs=phon.float(), dur_tgt=dur.float())
        ids = torch.arange(0, len(dur_pred[0]), device=dur_pred.device, dtype=dur_pred.dtype)
        ema_loss = 0
        for i in range(len(dur_lengths)):
            ema_loss += loss_fn(ema_out[i][:dec_lens[i]].float(), ema[i][:dec_lens[i]].float())
        ema_loss = ema_loss/ema_out.shape[0]
        loss = ema_loss + dur_loss
        if train:
            loss.backward()
            optimizer.step()
        loss_ += loss.item()
        _loss.append(loss.cpu().item())
        ema_loss_ += ema_loss.cpu().item()
        dur_loss_ += dur_loss.cpu().item()
    return loss_/len(loader), ema_loss_/len(loader), dur_loss_/len(loader), _loss
