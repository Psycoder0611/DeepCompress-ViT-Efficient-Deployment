"""Break-even rank and per-layer cost calculator for DeepCompress-ViT (DeiT).

A dense Linear(in -> out) costs in*out multiply-accumulates (MACs) per token.
A factorized layer (x @ Z) @ D with rank r costs r*(in + out) MACs per token.
The factorized layer is cheaper only when r < in*out / (in + out)  (break-even rank).

Usage:
  python ours/breakeven.py                       # DeiT-S, global rank 277
  python ours/breakeven.py --rank 277 --model base
  python ours/breakeven.py --ranks qkv=288,proj=192,fc1=300,fc2=300
Note: "GFLOPs" in the paper = 2 x MACs (DeiT-S: ~4.6 GMACs = ~9.2 GFLOPs).
"""
import argparse

TOKENS = 197  # 196 patches + cls token at 224x224, patch 16


def layer_shapes(dim):
    return {"qkv": (dim, 3 * dim), "proj": (dim, dim), "fc1": (dim, 4 * dim), "fc2": (4 * dim, dim)}


def breakeven_rank(i, o):
    return i * o / (i + o)


def dense_macs(i, o):
    return i * o


def fact_macs(i, o, r):
    return r * (i + o)


def report(dim, ranks, blocks=12, tokens=TOKENS):
    shapes = layer_shapes(dim)
    print(f"{'layer':6}{'shape':>14}{'break-even r':>14}{'rank used':>11}{'dense MACs/tok':>16}{'fact MACs/tok':>15}{'change':>9}")
    tot_d = tot_f = 0
    for k, (i, o) in shapes.items():
        r = ranks[k]
        d, f = dense_macs(i, o), fact_macs(i, o, r)
        tot_d += d
        tot_f += f
        print(f"{k:6}{str((i, o)):>14}{breakeven_rank(i, o):>14.1f}{r:>11}{d:>16,}{f:>15,}{(f - d) / d * 100:>+8.1f}%")
    saved_macs = (tot_d - tot_f) * blocks * tokens
    print(f"\nPer block, all 4 layers: dense {tot_d:,} vs factorized {tot_f:,} MACs/token")
    print(f"Whole model ({blocks} blocks, {tokens} tokens): saved {saved_macs / 1e9:.3f} GMACs = {2 * saved_macs / 1e9:.3f} GFLOPs "
          f"(negative = factorized costs MORE)")
    over = [k for k, (i, o) in shapes.items() if ranks[k] >= breakeven_rank(i, o)]
    if over:
        print("Layers at/above break-even (cost more than dense):", ", ".join(over))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["small", "base"], default="small")
    p.add_argument("--rank", type=int, default=277, help="same rank for all layer types")
    p.add_argument("--ranks", type=str, default=None, help="per type, e.g. qkv=288,proj=192,fc1=300,fc2=300")
    a = p.parse_args()
    dim = 768 if a.model == "base" else 384
    ranks = {k: a.rank for k in layer_shapes(dim)}
    if a.ranks:
        for kv in a.ranks.split(","):
            k, v = kv.split("=")
            ranks[k.strip()] = int(v)
    report(dim, ranks)
