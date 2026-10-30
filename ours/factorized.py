"""Real factorized (X.Z).D layers for DeepCompress-ViT.

The authors' inference.py rebuilds DENSE weights (decoder(encoder(W))) and runs a normal
model, so no factorized layer is ever executed. Here we keep the encoded weights Z (one per
block) and the shared decoder D, and compute  y = (x @ Z_i) @ D^T + bias  directly.

Facts taken from the authors' code (src/models/*.py, src/utils/model_utils.py):
  * one encoder/decoder per layer type (qkv, proj, fc1, fc2), shared by all 12 blocks
  * encoder input = the 12 blocks' weight.T stacked along dim 0, shape (12*in, out)
  * encoder output Z (4-bit activation-quantized) has shape (12*in, rank); block i uses rows i*in:(i+1)*in
  * decoder = LinearQ(rank, out) with 4-bit weights; decoded weight = Z_i @ D_eff^T
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_PATHS = {  # layer type -> (module path in a timm block, input dim multiplier)
    "qkv": ("attn", "qkv"),
    "proj": ("attn", "proj"),
    "fc1": ("mlp", "fc1"),
    "fc2": ("mlp", "fc2"),
}


class FactorizedLinear(nn.Module):
    """y = (x @ Z) @ D_t + bias.  Z: (in, r). D_t: (r, out)."""

    def __init__(self, Z, D_t, bias):
        super().__init__()
        self.Z = nn.Parameter(Z.detach().clone(), requires_grad=False)
        self.D_t = nn.Parameter(D_t.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        return F.linear(F.linear(x, self.Z.t()), self.D_t.t(), self.bias)


@torch.no_grad()
def effective_decoder_weight(decoder):
    """The (out, rank) weight the 4-bit LinearQ decoder actually uses in eval mode."""
    rank = decoder.weight.shape[1]
    eye = torch.eye(rank, device=decoder.weight.device, dtype=decoder.weight.dtype)
    return decoder(eye).t().contiguous()  # decoder(I) = I @ W_q^T -> (rank, out); transpose -> (out, rank)


@torch.no_grad()
def build_factorized_model(model, encoders, decoders, original_weights, in_dims, total_blocks=12):
    """Replace qkv/proj/fc1/fc2 of every block with FactorizedLinear (in place).

    model: timm ViT whose four layer types still hold the original biases
           (e.g. the 'compressed_model' after process_model_blocks, or any copy of the dense model)
    original_weights: dict layer-type -> stacked (12*in, out) tensor (from process_model_blocks)
    in_dims: dict layer-type -> input dim (qkv/proj/fc1: C; fc2: 4C)
    """
    for key, enc in encoders.items():
        enc.eval()
        decoders[key].eval()
        Z_all = enc(original_weights[key].to(next(enc.parameters()).device))  # (12*in, rank)
        D_eff = effective_decoder_weight(decoders[key])  # (out, rank)
        d_in = in_dims[key]
        sub, name = LAYER_PATHS[key]
        for b in range(total_blocks):
            block = model.blocks[b]
            parent = getattr(block, sub)
            old = getattr(parent, name)
            bias = old.bias
            Z_b = Z_all[b * d_in:(b + 1) * d_in]
            setattr(parent, name, FactorizedLinear(Z_b, D_eff.t(), bias))
    return model


def linear_param_bytes(model):
    """Bytes of all parameters, counting 4-bit encoded weights separately is done in benchmark.py."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
