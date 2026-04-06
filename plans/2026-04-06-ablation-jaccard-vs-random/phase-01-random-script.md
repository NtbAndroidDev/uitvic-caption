# Phase 01 — Tạo Random Pairing Script & Config

**Context:** [plan.md](./plan.md)
**Date:** 2026-04-06
**Priority:** High
**Status:** ✅ DONE — Step 2 complete - 2 files created, syntax+config verified

## Overview

Tạo 2 file mới cho variant ablation:
1. `scripts/generate_stage2_pairs_random.py` — clone generate_stage2_pairs.py, thay Jaccard → random
2. `configs/stage2_vit5_random.yaml` — clone stage2_vit5.yaml, đổi path input/output

## Key Insight

Thay đổi DUY NHẤT so với baseline:

```python
# BEFORE (Jaccard — baseline)
clean = best_matching_gt(noisy, gt_list).strip()

# AFTER (Random — variant)
import random
clean = random.choice(gt_list).strip()
```

Mọi thứ khác (BLIP model, batch_size, hyperparams ViT5) giữ nguyên để ablation công bằng.

## Related Code Files

- `scripts/generate_stage2_pairs.py` — nguồn để clone
- `configs/stage2_vit5.yaml` — config baseline (lr=5e-5, epochs=30, batch=16)
- `src/train_stage2.py` — không cần thay đổi gì

## Implementation Steps

### Step 1: Tạo `scripts/generate_stage2_pairs_random.py`

Clone toàn bộ `generate_stage2_pairs.py`, thực hiện 3 thay đổi nhỏ:

**Thay đổi 1** — Docstring (dòng 1–18):
```python
"""
Script sinh dữ liệu Stage 2 — RANDOM VARIANT (Ablation Study):
- Giống generate_stage2_pairs.py NGOẠI TRỪ:
  - Không dùng Jaccard similarity
  - Random chọn 1 trong 5 GT captions (random.choice)
- Dùng để so sánh ablation: Jaccard vs Random pairing

Output: stage2_pairs_random.jsonl
"""
```

**Thay đổi 2** — Xóa hàm `best_matching_gt()` (dòng 93–112), thay bằng:
```python
import random as _random

def random_gt(gt_captions: list) -> str:
    """Chọn ngẫu nhiên 1 GT caption (ablation: không dùng Jaccard)."""
    return _random.choice(gt_captions)
```

**Thay đổi 3** — Dòng 179 trong vòng lặp:
```python
# BEFORE:
clean = best_matching_gt(noisy, gt_list).strip()

# AFTER:
clean = random_gt(gt_list).strip()
```

**Thay đổi 4** — Output path mặc định (argparse, dòng 214):
```python
parser.add_argument("--output_path", type=str, default="data/stage2_pairs_random.jsonl")
```

### Step 2: Tạo `configs/stage2_vit5_random.yaml`

```yaml
model:
  name: "VietAI/vit5-base"

data:
  train_pairs: "/kaggle/working/data/stage2_pairs_random.jsonl"  # ← khác với baseline
  max_source_length: 64
  max_target_length: 64
  batch_size: 16
  num_workers: 2

training:
  num_epochs: 30
  lr: 5e-5
  weight_decay: 0.01
  device: "auto"
  grad_clip: 1.0
  early_stopping_patience: 5

logging:
  ckpt_dir: "/kaggle/working/outputs/stage2_random_checkpoints"  # ← khác với baseline
  log_every: 100
```

## Todo List

- [x] Tạo `scripts/generate_stage2_pairs_random.py`
- [x] Tạo `configs/stage2_vit5_random.yaml`
- [x] Verify: grep xác nhận không còn call `best_matching_gt` trong file random

## Success Criteria

- `generate_stage2_pairs_random.py` runs without error khi test với `--max_batches 2`
- `configs/stage2_vit5_random.yaml` đổi đúng 2 paths: `train_pairs` + `ckpt_dir`
- Không thay đổi bất kỳ hyperparameter nào để ablation công bằng

## Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| random.seed không fix → kết quả không reproduced | Medium | Thêm `random.seed(42)` trước vòng lặp |
| File output đè lên nhau | Low | Default output path khác nhau (_random.jsonl vs _blip.jsonl) |

## Next Steps

→ Phase 02: [phase-02-train-random.md](./phase-02-train-random.md)
