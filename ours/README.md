# Our work (DATA 255, Group 6) on top of DeepCompress-ViT

Base code: https://github.com/ML-Security-Research-LAB/DeepCompress-ViT (Ahmed et al., CVPR 2025). Everything in `ours/` is new;
files outside `ours/` are the authors' unless a commit says otherwise.

Goal (professor): keep CIFAR-10 accuracy within 1-2% of the uncompressed model (97.29%, i.e. >= ~95.3-96.3%),
and improve AND measure latency and memory.

## Files
| file | what it does | status |
|---|---|---|
| `breakeven.py` | break-even rank and per-layer MACs/FLOPs for any rank per layer type | tested (matches paper Table 3 numbers: 0.32 and 4.30 GFLOPs saved) |
| `factorized.py` | real factorized layers `(x@Z)@D + b` built from encoder/decoder | tested on CPU with random weights |
| `test_equivalence.py` | checks factorized model == decoded-dense model (authors' eval path) | passes, max diff ~6e-7 (random weights) |
| `benchmark.py` | latency + memory for dense / factorized, batch sizes 1,4,32, writes `results/benchmark.csv` | tested on CPU only |

Not tested yet: anything with the authors' real checkpoints, anything on a GPU, any training.

## Setup
```bash
git clone https://github.com/Psycoder0611/DeepCompress-ViT-Efficient-Deployment.git
cd DeepCompress-ViT-Efficient-Deployment
python -m venv .venv && source .venv/bin/activate        # authors used Python 3.8 / torch 2.3.1 / timm 1.0.7
pip install torch torchvision timm tqdm numpy            # do NOT pip install their full requirements.txt (conda paths, 6.5 KB of pins)
```
Authors' pins: PyTorch 2.3.1, torchvision 0.18.1, timm 1.0.7. Newer versions ran our tests, but if anything
behaves oddly, match the pins.

## Data
- CIFAR-10: downloads itself (torchvision). NOTE: the authors' loader hard-codes `root='/data/cifar10_data'`
  in `src/utils/data_utils.py`. Change it to `./data/cifar10` (or create `/data/cifar10_data`) before running.
- Dense CIFAR-10 DeiT-S checkpoint -> `saved_models/deit_small_cifar10_best.pth` (needed by `get_models`).
- Compressed rank-277 CIFAR-10 checkpoint -> `saved_models/small_rank_277_cifar10/deit_small_patch16_224.pth`.
  Both checkpoints: links are in the authors' README (Google Drive). Check that the dense one is there; if not, we fine-tune DeiT-S ourselves.
- Pretrained DeiT-S weights are downloaded by timm automatically on first use.
- ImageNet is NOT needed.

## Run order
```bash
python ours/breakeven.py                                   # 1. cost table (no GPU, no data)
python ours/test_equivalence.py                            # 2. should print PASS
python ours/benchmark.py --mode dense --device cpu         # 3. dense latency/memory baseline
python ours/benchmark.py --mode factorized-random --ranks qkv=277,proj=277,fc1=277,fc2=277 --device cpu
# 4. reproduce accuracy with the authors' script (needs a GPU: their script calls .cuda())
python inference.py --dataset cifar10 --rank 277 --batch_size 128 \
    --state_path saved_models/small_rank_277_cifar10/deit_small_patch16_224.pth
# 5. latency/memory of the real trained model
python ours/benchmark.py --mode factorized-ckpt --dense_ckpt saved_models/deit_small_cifar10_best.pth \
    --state_path saved_models/small_rank_277_cifar10/deit_small_patch16_224.pth --rank 277 --device cuda:0
```
Run each benchmark in its own process; add `--tag name` to label rows. For fair numbers: same machine, nothing else running,
same `--threads` on CPU.

## Known facts about the authors' code (read from the source)
- `main.py` always builds ImageNet loaders (`get_dataloaders_imagenet`); CIFAR-10 training needs a small edit to use `get_dataloaders_cifar10`.
- One `--rank` is used for all four layer types; encoder/decoder are shared across the 12 blocks, so ranks can vary per layer type (4 values), not per block.
- `inference.py` evaluates accuracy on decoded dense weights and calls `.cuda()`, so it needs a GPU. It counts FLOPs and parameter storage but never times anything.
