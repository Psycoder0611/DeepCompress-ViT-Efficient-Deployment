"""Accuracy on the CIFAR-10 test set (224x224) for three models, on CPU / Apple GPU (mps) / CUDA.

  1. dense      : authors' uncompressed DeiT-S checkpoint            (expected ~97.29%)
  2. decoded    : compressed rank-277 model, dense weights rebuilt   (authors' eval path; expected ~96.73%)
  3. factorized : same compressed model, real (X.Z).D layers         (should equal 'decoded')

Run from the repo root:
  python ours/eval_cifar10.py --device cpu            # ~10-20 min on a laptop for all three
  python ours/eval_cifar10.py --device cpu --limit 500   # quick check on the first 500 test images
  python ours/eval_cifar10.py --device mps            # Apple GPU, faster if it works

Differences from the authors' inference.py (which needs CUDA): runs in fp32, no autocast, and does not
convert biases/LayerNorm to half. Small differences (a few hundredths of a percent) from 96.73 are possible;
report the number you measure.
"""
import argparse, copy, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.autoencoder import get_encoder_decoders
from src.models.vit import quantize_first_and_last_layer
from src.utils.model_utils import get_layer_names, process_model_blocks, update_model_weights
from ours.factorized import build_factorized_model


@torch.inference_mode()
def accuracy(model, loader, device, name):
    model.eval().to(device)
    correct = total = 0
    for x, y in tqdm(loader, desc=name, leave=False):
        out = model(x.to(device))
        correct += (out.argmax(1).cpu() == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def get_loader(args):
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.synthetic:  # smoke test only: random images, accuracy is meaningless
        class Fake(torch.utils.data.Dataset):
            def __len__(self): return args.synthetic
            def __getitem__(self, i): return torch.randn(3, 224, 224), i % 10
        ds = Fake()
    else:
        ds = datasets.CIFAR10(root=args.data_root, train=False, transform=tf, download=True)
        if args.limit:
            ds = Subset(ds, range(args.limit))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense_ckpt", default="saved_models/deit_small_cifar10_best.pth")
    p.add_argument("--state_path", default="saved_models/small_rank_277_cifar10/deit_small_patch16_224.pth")
    p.add_argument("--rank", type=int, default=277)
    p.add_argument("--model_name", default="deit_small_patch16_224")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data_root", default="./data/cifar10")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only the first N test images (0 = all 10,000)")
    p.add_argument("--synthetic", type=int, default=0, help="smoke test with N random images (no CIFAR needed)")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = args.device
    dim = 768 if "base" in args.model_name else 384
    a = types.SimpleNamespace(hidden_dim=dim, rank=args.rank, device="cpu", total_blocks=12, skip_qkv=False)
    loader = get_loader(args)
    print(f"test images: {len(loader.dataset)}   device: {dev}")

    dense = timm.create_model(args.model_name, pretrained=False, num_classes=10)
    dense.load_state_dict(torch.load(args.dense_ckpt, map_location="cpu"), strict=True)
    t0 = time.time()
    acc_dense = accuracy(dense, loader, dev, "dense")
    print(f"dense      : {acc_dense:.2f}%   ({time.time() - t0:.0f}s)")
    dense.cpu()

    comp = timm.create_model(args.model_name, pretrained=False, num_classes=10)
    comp.load_state_dict(dense.state_dict())
    comp = quantize_first_and_last_layer(comp, a)
    layers = get_layer_names(a)
    orig_w, seq = process_model_blocks(dense, comp, a, selected_layers=layers)
    enc, dec = get_encoder_decoders(layers, seq, a)
    state = torch.load(args.state_path, map_location="cpu")
    for k in enc:
        enc[k].load_state_dict(state["encoder_states"][k])
        dec[k].load_state_dict(state["decoder_states"][k])
        enc[k].eval(); dec[k].eval()
    comp.load_state_dict(state["model_state_dict"])
    comp.eval()

    fac = copy.deepcopy(comp)
    in_dims = {"qkv": dim, "proj": dim, "fc1": dim, "fc2": 4 * dim}
    with torch.no_grad():
        decoded = {k: dec[k](enc[k](orig_w[k])) for k in dec}
        update_model_weights(comp, decoded, total_blocks=12, selected_layers=layers, hidden_dim=dim)
        build_factorized_model(fac, enc, dec, orig_w, in_dims)

    t0 = time.time()
    acc_dec = accuracy(comp, loader, dev, "decoded")
    print(f"decoded    : {acc_dec:.2f}%   ({time.time() - t0:.0f}s)")
    comp.cpu()
    t0 = time.time()
    acc_fac = accuracy(fac, loader, dev, "factorized")
    print(f"factorized : {acc_fac:.2f}%   ({time.time() - t0:.0f}s)")
    print(f"\nsummary: dense {acc_dense:.2f} | compressed(decoded) {acc_dec:.2f} | compressed(factorized) {acc_fac:.2f} "
          f"| drop vs dense {acc_dense - acc_fac:.2f} points")


if __name__ == "__main__":
    main()
