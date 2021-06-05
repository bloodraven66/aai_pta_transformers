import sys
from typing import Optional
from os.path import abspath, dirname

import torch

sys.path.append(abspath(dirname(__file__)+'/'))
from model import FastPitch as _FastPitch
from model_AAI import Transformer_AAI as _Transformer_AAI

def parse_model_args(model_name, parser, add_help=False):
    if model_name == 'FastPitch':
        from arg_parser_fp import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    elif model_name == 'FastPitch_AAI':
        from arg_parser_fp_AAI import parse_fastpitch_args
        return parse_fastpitch_args(parser, add_help)
    else:
        raise NotImplementedError(model_name)


def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)


def get_model(model_name, model_config, device,
              uniform_initialize_bn_weight=False, forward_is_infer=False,
              jitable=False):
    """ Code chooses a model based on name"""
    model = None
    if model_name == 'FastPitch':
        if forward_is_infer:
            if jitable:
                class FastPitch__forward_is_infer(_FastPitchJIT):
                    def forward(self, inputs, input_lengths, pace: float = 1.0,
                                dur_tgt: Optional[torch.Tensor] = None,
                                pitch_tgt: Optional[torch.Tensor] = None,
                                speaker: int = 0):
                        return self.infer(inputs, input_lengths, pace=pace,
                                          dur_tgt=dur_tgt, pitch_tgt=pitch_tgt,
                                          speaker=speaker)
            else:
                class FastPitch__forward_is_infer(_FastPitch):
                    def forward(self, inputs, input_lengths, pace: float = 1.0,
                                dur_tgt: Optional[torch.Tensor] = None,
                                pitch_tgt: Optional[torch.Tensor] = None,
                                pitch_transform=None,
                                speaker: Optional[int] = None):
                        return self.infer(inputs, input_lengths, pace=pace,
                                          dur_tgt=dur_tgt, pitch_tgt=pitch_tgt,
                                          pitch_transform=pitch_transform,
                                          speaker=speaker)

            model = FastPitch__forward_is_infer(**model_config)
        else:
            model = _FastPitch(**model_config)

    elif model_name == 'Transformer_AAI':

        if forward_is_infer:
                class Transformer__forward_is_infer(_Transformer_AAI):
                    def forward(self, inputs, input_lengths, pace: float = 1.0,
                                dur_tgt: Optional[torch.Tensor] = None,
                                pitch_tgt: Optional[torch.Tensor] = None,
                                pitch_transform=None,
                                speaker: Optional[int] = None):
                        return self.infer(inputs, input_lengths, pace=pace,
                                          dur_tgt=dur_tgt, pitch_tgt=pitch_tgt,
                                          pitch_transform=pitch_transform,
                                          speaker=speaker)

            model = Transformer__forward_is_infer(**model_config)
        else:
            model = _Transformer_AAI(**model_config)

    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)
    return model.to(device)


def get_model_config(model_name, args):
    """ Code chooses a model based on name"""

    if model_name == 'FastPitch':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            max_seq_len=args.max_seq_len,
            # symbols
            n_symbols=40,
            padding_idx=0,
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight
        )
        return model_config

    elif model_name == 'Transformer_AAI':
        model_config = dict(
            # io
            n_mel_channels=args.n_mel_channels,
            max_seq_len=args.max_seq_len,
            # symbols
            n_symbols=41,
            padding_idx=40,
            symbols_embedding_dim=args.symbols_embedding_dim,
            # input FFT
            in_fft_n_layers=args.in_fft_n_layers,
            in_fft_n_heads=args.in_fft_n_heads,
            in_fft_d_head=args.in_fft_d_head,
            in_fft_conv1d_kernel_size=args.in_fft_conv1d_kernel_size,
            in_fft_conv1d_filter_size=args.in_fft_conv1d_filter_size,
            in_fft_output_size=args.in_fft_output_size,
            p_in_fft_dropout=args.p_in_fft_dropout,
            p_in_fft_dropatt=args.p_in_fft_dropatt,
            p_in_fft_dropemb=args.p_in_fft_dropemb,
            # output FFT
            out_fft_n_layers=args.out_fft_n_layers,
            out_fft_n_heads=args.out_fft_n_heads,
            out_fft_d_head=args.out_fft_d_head,
            out_fft_conv1d_kernel_size=args.out_fft_conv1d_kernel_size,
            out_fft_conv1d_filter_size=args.out_fft_conv1d_filter_size,
            out_fft_output_size=args.out_fft_output_size,
            p_out_fft_dropout=args.p_out_fft_dropout,
            p_out_fft_dropatt=args.p_out_fft_dropatt,
            p_out_fft_dropemb=args.p_out_fft_dropemb,
            # duration predictor
            dur_predictor_kernel_size=args.dur_predictor_kernel_size,
            dur_predictor_filter_size=args.dur_predictor_filter_size,
            p_dur_predictor_dropout=args.p_dur_predictor_dropout,
            dur_predictor_n_layers=args.dur_predictor_n_layers,
            # pitch predictor
            pitch_predictor_kernel_size=args.pitch_predictor_kernel_size,
            pitch_predictor_filter_size=args.pitch_predictor_filter_size,
            p_pitch_predictor_dropout=args.p_pitch_predictor_dropout,
            pitch_predictor_n_layers=args.pitch_predictor_n_layers,
            # pitch conditioning
            pitch_embedding_kernel_size=args.pitch_embedding_kernel_size,
            # speakers parameters
            n_speakers=args.n_speakers,
            speaker_emb_weight=args.speaker_emb_weight
        )
        return model_config
    else:
        raise NotImplementedError(model_name)
