import torch
import torch.nn as nn
from .encoders import Encoder
from .quantization import LinearQ

LAYER_TYPES = ('qkv', 'proj', 'fc1', 'fc2')


def get_ranks(args):
    """Rank for each layer type. --rank_<type> overrides --rank for that type only."""
    return {k: (getattr(args, f'rank_{k}', None) or args.rank) for k in LAYER_TYPES}


def get_encoder_decoders(selected_layers, input_seq_lens, args):
    hidden_dim = args.hidden_dim
    ranks = get_ranks(args)
    encoders = {}
    decoders = {}
    for layer in selected_layers:
        if 'qkv' in layer and 'qkv' not in encoders:
            encoders['qkv'] = Encoder(hidden_dim, ranks['qkv'], 'qkv', True, input_seq_lens['qkv'])
            decoders['qkv'] = LinearQ(ranks['qkv'], hidden_dim * 3, bias=False, nbits_w=4)
        if 'proj' in layer and 'proj' not in encoders:
            encoders['proj'] = Encoder(hidden_dim, ranks['proj'], 'proj', True, input_seq_lens['proj'])
            decoders['proj'] = LinearQ(ranks['proj'], hidden_dim, bias=False, nbits_w=4)
        if 'fc1' in layer and 'fc1' not in encoders:
            encoders['fc1'] = Encoder(hidden_dim, ranks['fc1'], 'fc1', True, input_seq_lens['fc1'])
            decoders['fc1'] = LinearQ(ranks['fc1'], hidden_dim * 4, bias=False, nbits_w=4)
        if 'fc2' in layer and 'fc2' not in encoders:
            encoders['fc2'] = Encoder(hidden_dim, ranks['fc2'], 'fc2', True, input_seq_lens['fc2'])
            decoders['fc2'] = LinearQ(ranks['fc2'], hidden_dim, bias=False, nbits_w=4)

    for key in encoders.keys():
        encoders[key].to(args.device)
        decoders[key].to(args.device)

    return encoders, decoders


