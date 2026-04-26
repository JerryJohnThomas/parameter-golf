# Records

| run_id | changes | steps | val_bpb (pre-quant) | val_bpb (post-quant) | artifact_MB | command | notes |
|--------|---------|-------|---------------------|----------------------|-------------|---------|-------|
| baseline_10min | none (baseline) | 1737 | 1.3441 | 1.3451 | 14.6MB | `MAX_WALLCLOCK_SECONDS=600 RUN_ID=baseline_10min torchrun --standalone --nproc_per_node=1 train_gpt.py` | 1x H100, 1 shard |
| exp1_layers11_warmdown | NUM_LAYERS=11, WARMDOWN_ITERS=3600, MATRIX_LR=0.05 | 1407 | 1.3459 | 1.3493 | 14.4MB | `MAX_WALLCLOCK_SECONDS=600 RUN_ID=exp1_layers11_warmdown NUM_LAYERS=11 WARMDOWN_ITERS=3600 MATRIX_LR=0.05 torchrun --standalone --nproc_per_node=1 train_gpt.py` | 1x H100, 1 shard |