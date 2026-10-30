"""CPU test: the factorized model must match the 'decoded dense' model (the authors' eval path).
Uses random-initialised encoders/decoders (no checkpoint needed). Run from the repo root:
    python ours/test_equivalence.py
"""
import sys, os, types, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, timm
from src.models.autoencoder import get_encoder_decoders
from src.utils.model_utils import get_layer_names, process_model_blocks, update_model_weights
from ours.factorized import build_factorized_model

torch.manual_seed(0)
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 277
args = types.SimpleNamespace(hidden_dim=384, rank=rank, device="cpu", total_blocks=12, skip_qkv=False)

dense = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=10).eval()
dec_model = copy.deepcopy(dense)           # gets decoded-dense weights (authors' path)
fac_model = copy.deepcopy(dense)           # gets real factorized layers
layers = get_layer_names(args)
orig_w, seq = process_model_blocks(dense, dec_model, args, selected_layers=layers)
encoders, decoders = get_encoder_decoders(layers, seq, args)

# initialise the quantizer scales the way training does (first call in train mode), then freeze
with torch.no_grad():
    for k in encoders:
        encoders[k].train(); decoders[k].train()
        decoders[k](encoders[k](orig_w[k]))
        encoders[k].eval(); decoders[k].eval()

with torch.no_grad():
    decoded = {k: decoders[k](encoders[k](orig_w[k])) for k in decoders}
    update_model_weights(dec_model, decoded, total_blocks=12, selected_layers=layers, hidden_dim=384)

in_dims = {"qkv": 384, "proj": 384, "fc1": 384, "fc2": 1536}
_, _ = process_model_blocks(dense, fac_model, args, selected_layers=layers)  # keeps biases in CustomFC
build_factorized_model(fac_model, encoders, decoders, orig_w, in_dims)
fac_model.eval()

x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    a, b = dec_model(x), fac_model(x)
err = (a - b).abs().max().item()
print(f"rank={rank}  max|decoded-dense - factorized| = {err:.3e}   (output scale ~{a.abs().mean().item():.3f})")
assert err < 1e-3, "factorized model does not match decoded-dense model"
print("PASS")
