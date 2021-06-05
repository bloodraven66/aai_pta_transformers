import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from common.utils import mask_from_lens
import numpy as np
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]



class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False, relative=None):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.k = relative
        self.val_rel = relative
        self.training = True
        seq_len = 400
        if relative is not None:
            self.embed_idx = torch.tensor(np.array([max(-self.k, min(self.k, j-i)) for i in range(seq_len) for j in range(seq_len)]), device=device).view(-1, seq_len, seq_len).long().squeeze()+self.k
            self.relative_embed = nn.Embedding(self.k*2+1, 8).float().to(device)
            # self.relative_embed2 = nn.Embedding(self.k*2+1, 8).float().to(device)
            #                                  padding_idx=self.padding_idx).
    def forward(self, inp, attn_mask=None, return_attn=False):
        return self._forward(inp, attn_mask, return_attn=return_attn)

    def _forward(self, inp, attn_mask=None, return_attn=False):
        residual = inp
        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head
        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=-1)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)
        if self.k is not None:
            seq_len = inp.shape[1]
            if not self.training:
                embed_idx = torch.from_numpy(np.array([max(-self.k, min(self.k, j-i)) for i in range(seq_len) for j in range(seq_len)])).view(-1, seq_len, seq_len).long().to(device).squeeze()+self.k
            else:
                embed_idx = self.embed_idx
            rel_pos_vec = self.relative_embed(embed_idx)
            rel_pos_vec = rel_pos_vec.permute(0, 2, 1)
            q = q.permute(1, 0, 2)
            rel_score = torch.matmul(q, rel_pos_vec).permute(1, 0, 2)
            attn_score += rel_score
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask, -float('inf'))
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)
        if return_attn:
            return output, attn_prob
        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, relative
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, relative=relative, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None, return_attn=False):
        output = self.dec_attn(dec_inp, attn_mask=None, return_attn=return_attn)
        if return_attn:
            output, attn = output
        output = self.pos_ff(output)
        if return_attn:
            return output, attn
        else:
            return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False,
                 pos_embed_type=None):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = 40

        self.word_emb = None
        if pos_embed_type == 'concat':
            self.pos_emb = PositionalEmbedding(32)
        elif pos_embed_type == 'additive':
            self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        self.ln = nn.LayerNorm(d_model)
        self.attn = []
        if pos_embed_type == 'relative':
            relative = 10
        else:
            relative = None
        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout, relative
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None,pos_embed_type=None):
        out = dec_inp
        if pos_embed_type == 'additive':
            pos_seq = torch.arange(out.size(1), device=out.device, dtype=out.dtype)
            pos_emb = self.pos_emb(pos_seq)
            pos_emb = torch.cat(out.shape[0]*[pos_emb])
            out = self.ln(out) + pos_emb
        elif pos_embed_type == 'concat':
            pos_seq = torch.arange(out.size(1), device=out.device, dtype=out.dtype)
            pos_emb = self.pos_emb(pos_seq)
            pos_emb = torch.cat(out.shape[0]*[pos_emb])
            out = torch.cat([out, pos_emb], dim=-1)



        for idx, layer in enumerate(self.layers):
            out = layer(out, mask=None)
        return out
