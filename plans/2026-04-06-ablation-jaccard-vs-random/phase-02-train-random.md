# Phase 02 — Chạy Kaggle Notebook: Random Pair Generation + ViT5 Training

**Context:** [plan.md](./plan.md) | [phase-01](./phase-01-random-script.md)
**Date:** 2026-04-06
**Priority:** High
**Status:** 🔲 Pending (cần phase-01 xong trước)

## Overview

Chạy toàn bộ pipeline random variant trên Kaggle:
1. Generate random pairs từ BLIP + annotation JSONs (~30–60min GPU)
2. Train ViT5 trên pairs mới (~4–6h GPU, early stopping)

## Kaggle Notebook Setup

### Datasets cần attach:
| Dataset | Kaggle path |
|---|---|
| BLIP Stage 1 checkpoint | `/kaggle/input/blipcaptionerv3/best_model.pt` |
| UITViC annotations + images | `/kaggle/input/uitvic/` |
| (Optional) Source code | Upload hoặc dùng Kaggle dataset |

### Notebook cells theo thứ tự:

**Cell 1 — Setup environment:**
```bash
pip install -q transformers torch pillow nltk
python -c "import nltk; nltk.download('punkt')"
```

**Cell 2 — Clone/copy source code:**
```bash
# Option A: Upload source qua Kaggle dataset
cp -r /kaggle/input/uitvic-source/* /kaggle/working/

# Option B: Git clone (nếu repo public)
git clone https://github.com/YOUR_REPO uitvic-captioning
cd uitvic-captioning
```

**Cell 3 — Generate random pairs:**
```bash
python scripts/generate_stage2_pairs_random.py \
    --blip_model Salesforce/blip-image-captioning-base \
    --checkpoint /kaggle/input/blipcaptionerv3/best_model.pt \
    --train_json /kaggle/input/uitvic/uitvic_captions_train2017.json \
    --train_images /kaggle/input/uitvic/coco_uitvic_train \
    --val_json /kaggle/input/uitvic/uitvic_captions_test2017.json \
    --val_images /kaggle/input/uitvic/coco_uitvic_test \
    --output_path /kaggle/working/data/stage2_pairs_random.jsonl \
    --batch_size 16
```

**Cell 4 — Train ViT5 (random variant):**
```bash
python -m src.train_stage2 --config configs/stage2_vit5_random.yaml
```

## Timeline Estimate

| Bước | Thời gian |
|---|---|
| Setup Kaggle, attach datasets | 15 phút |
| Generate random pairs (BLIP inference) | 30–60 phút |
| Train ViT5 (30 epochs, early stopping) | 4–6 giờ |
| Lưu output checkpoints | 5 phút |

**Tổng: ~6–8h Kaggle GPU session**

> ⚠️ Kaggle GPU limit: 30h/week. Hãy chắc chắn session được monitor.

## Outputs Cần Lưu

Sau khi training xong, download hoặc lưu vào Kaggle Output:
- `/kaggle/working/outputs/stage2_random_checkpoints/best_model.pt`
- `/kaggle/working/outputs/stage2_random_checkpoints/stage2_metrics.csv`
- `/kaggle/working/outputs/stage2_random_checkpoints/stage2_report.png`
- `/kaggle/working/data/stage2_pairs_random.jsonl` (optional, để verify)

## Verification

Sau khi training xong, check:
```python
import pandas as pd
df = pd.read_csv("/kaggle/working/outputs/stage2_random_checkpoints/stage2_metrics.csv")
print(df.sort_values("BLEU-4", ascending=False).head(3))
print(f"Best BLEU-4: {df['BLEU-4'].max():.4f}")
```

## Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| Kaggle GPU timeout (~9h limit) | Medium | Early stopping patience=5 giúp dừng sớm |
| BLIP output noisy khác lần trước | Low | seed=42 fix cho random, BLIP generate deterministic (num_beams) |
| Val BLEU-4 thực ra cao hơn Jaccard | Low | Vẫn là kết quả ablation hợp lệ, phân tích lý do trong báo cáo |

## Next Steps

→ Phase 03: [phase-03-compare-results.md](./phase-03-compare-results.md)
