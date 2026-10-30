## DATA 255 Group 6: efficient deployment of DeepCompress-ViT
Fork of DeepCompress-ViT (CVPR 2025). Our additions are in `ours/` (see `ours/README.md`): real factorized
(X·Z)·D layers verified against the decoded-dense model, a latency/memory benchmark, and a break-even-aware per-layer-type rank search.
Target: CIFAR-10 accuracy within 1-2% of the uncompressed 97.29% with measured latency and memory gains.
Status: in progress. Results will be added to this table.

| model | CIFAR-10 acc | storage MB | peak mem MB | latency ms (bs 1) |
|---|---|---|---|---|
| dense DeiT-S | | | | |
| authors' rank 277 | | | | |
| ours | | | | |
