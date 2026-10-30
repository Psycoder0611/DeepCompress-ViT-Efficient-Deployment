"""Latency + memory benchmark: dense DeiT-S vs real factorized (X.Z).D DeiT-S.

Run from the repo root. ONE configuration per process (so peak-memory numbers are not mixed):

  # dense baseline (weights do not affect speed, so no checkpoint needed)
  python ours/benchmark.py --mode dense --device cpu

  # factorized layers with random weights: latency vs rank BEFORE any training
  python ours/benchmark.py --mode factorized-random --ranks qkv=277,proj=277,fc1=277,fc2=277 --device cpu

  # factorized layers from the authors' / our trained checkpoint
  python ours/benchmark.py --mode factorized-ckpt --dense_ckpt saved_models/deit_small_cifar10_best.pth \
      --state_path saved_models/small_rank_277_cifar10/deit_small_patch16_224.pth --rank 277 --device cuda:0

Results are appended to ours/results/benchmark.csv.
Reported: parameter bytes held in memory, peak memory (CUDA: torch.cuda.max_memory_allocated;
CPU: peak process RSS, sampled every 2 ms, minus the RSS measured before the model was built;
one run per process. It includes the model weights, activations and PyTorch runtime buffers), latency median/p95
per batch and per image, for each batch size.
"""
import argparse, csv, os, statistics, sys, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, timm
from ours.factorized import FactorizedLinear, LAYER_PATHS

TYPES = ["qkv", "proj", "fc1", "fc2"]


def parse_ranks(s, default):
    r = {k: default for k in TYPES}
    if s:
        for kv in s.split(","):
            k, v = kv.split("=")
            r[k.strip()] = int(v)
    return r


def build_dense(args):
    m = timm.create_model(args.model_name, pretrained=False, num_classes=args.num_classes)
    if args.dense_ckpt:
        m.load_state_dict(torch.load(args.dense_ckpt, map_location="cpu"), strict=True)
    return m


def build_factorized_random(args):
    m = build_dense(args)
    dim = 768 if "base" in args.model_name else 384
    shapes = {"qkv": (dim, 3 * dim), "proj": (dim, dim), "fc1": (dim, 4 * dim), "fc2": (4 * dim, dim)}
    ranks = parse_ranks(args.ranks, args.rank)
    for key in TYPES:
        i, o = shapes[key]
        sub, name = LAYER_PATHS[key]
        for blk in m.blocks:
            parent = getattr(blk, sub)
            old = getattr(parent, name)
            setattr(parent, name, FactorizedLinear(torch.randn(i, ranks[key]) * 0.02,
                                                   torch.randn(ranks[key], o) * 0.02, old.bias))
    return m


def build_factorized_ckpt(args):
    """Mirrors the authors' inference.py, but device-agnostic, then builds real factorized layers."""
    from src.models.autoencoder import get_encoder_decoders
    from src.models.vit import quantize_first_and_last_layer
    from src.utils.model_utils import get_layer_names, process_model_blocks
    from ours.factorized import build_factorized_model
    dim = 768 if "base" in args.model_name else 384
    a = types.SimpleNamespace(hidden_dim=dim, rank=args.rank, device=args.device, total_blocks=12, skip_qkv=False)
    dense = build_dense(args)
    comp = timm.create_model(args.model_name, pretrained=False, num_classes=args.num_classes)
    comp.load_state_dict(dense.state_dict())
    comp = quantize_first_and_last_layer(comp, a)
    layers = get_layer_names(a)
    orig_w, seq = process_model_blocks(dense, comp, a, selected_layers=layers)
    enc, dec = get_encoder_decoders(layers, seq, a)
    state = torch.load(args.state_path, map_location=args.device)
    for k in enc:
        enc[k].load_state_dict(state["encoder_states"][k])
        dec[k].load_state_dict(state["decoder_states"][k])
    comp.load_state_dict(state["model_state_dict"])
    in_dims = {"qkv": dim, "proj": dim, "fc1": dim, "fc2": 4 * dim}
    return build_factorized_model(comp, enc, dec, orig_w, in_dims)


try:
    import psutil
    _PROC = psutil.Process()
except ImportError:  # fall back to Linux /proc; on macOS install psutil: pip install psutil
    _PROC = None


def current_rss_mb():
    """Resident memory of this process in MB (works on macOS/Linux with psutil, Linux without)."""
    if _PROC is not None:
        return _PROC.memory_info().rss / 2**20
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


class RssSampler:
    """Samples this process' resident memory every 2 ms and keeps the maximum (CPU runs, Linux)."""

    def __init__(self):
        self.peak, self._stop = 0.0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, current_rss_mb())
            time.sleep(0.002)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *e):
        self._stop.set()
        self._t.join()


@torch.inference_mode()
def time_model(model, bs, device, warmup, iters, half):
    x = torch.randn(bs, 3, 224, 224, device=device)
    if half:
        x = x.half()
    cuda = device.startswith("cuda")
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "factorized-random", "factorized-ckpt"], required=True)
    p.add_argument("--model_name", default="deit_small_patch16_224")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_sizes", default="1,4,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rank", type=int, default=277)
    p.add_argument("--ranks", default=None, help="per type, e.g. qkv=288,proj=192,fc1=300,fc2=300")
    p.add_argument("--dense_ckpt", default=None)
    p.add_argument("--state_path", default=None)
    p.add_argument("--half", action="store_true", help="run in fp16 (GPU only)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = default)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    base_rss = current_rss_mb()  # library/import overhead, subtracted from CPU peak below

    build = {"dense": build_dense, "factorized-random": build_factorized_random,
             "factorized-ckpt": build_factorized_ckpt}[a.mode]
    model = build(a).to(a.device).eval()
    if a.half:
        model = model.half()
    param_mb = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers())) / 2**20
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    base_after_model = current_rss_mb()

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "benchmark.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "mode", "tag", "device", "half", "ranks", "batch", "latency_ms_median",
                        "latency_ms_p95", "ms_per_image", "param_MB_in_memory", "peak_mem_MB"])
        print(f"mode={a.mode} device={a.device} params in memory={param_mb:.1f} MB")
        for bs in [int(s) for s in a.batch_sizes.split(",")]:
            if a.device.startswith("cuda"):
                med, p95 = time_model(model, bs, a.device, a.warmup, a.iters, a.half)
                peak = torch.cuda.max_memory_allocated() / 2**20
            else:
                with RssSampler() as smp:
                    smp.peak = base_after_model
                    med, p95 = time_model(model, bs, a.device, a.warmup, a.iters, a.half)
                peak = smp.peak - base_rss
            ranks = "-" if a.mode == "dense" else str(parse_ranks(a.ranks, a.rank))
            print(f"  batch {bs:>3}: median {med:8.2f} ms  p95 {p95:8.2f} ms  ({med / bs:6.2f} ms/img)  peak {peak:8.1f} MB")
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.mode, a.tag, a.device, a.half, ranks, bs,
                        f"{med:.3f}", f"{p95:.3f}", f"{med / bs:.3f}", f"{param_mb:.2f}", f"{peak:.1f}"])


if __name__ == "__main__":
    main()
