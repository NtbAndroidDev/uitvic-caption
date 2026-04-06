# Phase 03 — Evaluation & So Sánh Kết Quả (Ablation Table)

**Context:** [plan.md](./plan.md) | [phase-02](./phase-02-train-random.md)
**Date:** 2026-04-06
**Priority:** Medium
**Status:** 🔲 Pending (cần phase-02 xong trước)

## Overview

Chạy end-to-end evaluation cho random variant, tổng hợp bảng ablation để đưa vào báo cáo đồ án.

## Evaluation Steps

### Step 1 — End-to-end evaluate (Kaggle)

Chạy `evaluate_pipeline.py` với ViT5 random checkpoint:

```bash
python scripts/evaluate_pipeline.py \
    --blip_model Salesforce/blip-image-captioning-base \
    --blip_checkpoint /kaggle/input/blipcaptionerv3/best_model.pt \
    --vit5_checkpoint /kaggle/working/outputs/stage2_random_checkpoints/best_model.pt \
    --test_json /kaggle/input/uitvic/uitvic_captions_test2017.json \
    --test_images /kaggle/input/uitvic/coco_uitvic_test \
    --output_path /kaggle/working/outputs/eval_random_results.json
```

### Step 2 — So sánh kết quả

Dùng `stage2_metrics.csv` của cả 2 model để so sánh:
- Val BLEU-4 best epoch của Jaccard vs Random
- End-to-end BLEU-1/2/3/4 trên full test set

## Bảng Ablation Study (Template cho báo cáo)

### Bảng 4.x: Ablation — Chiến lược ghép cặp Stage 2

| Phương pháp ghép cặp | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Random Selection | ? | ? | ? | ? |
| **Jaccard Similarity (đề xuất)** | **0.4552** | **0.3196** | **0.2493** | **0.1951** |
| Chênh lệch | ? | ? | ? | ? |

### Phân tích định tính (mẫu template)

> **Jaccard matching đảm bảo rằng cặp (noisy, clean) cùng mô tả nội dung ảnh**: noisy caption sinh bởi BLIP (không dấu) được ghép với GT caption có nội dung gần nhất. Điều này giúp ViT5 học task "khôi phục dấu câu" thay vì "học nội dung ngẫu nhiên".
> 
> Khi dùng random pairing, GT caption có thể mô tả một khía cạnh khác của ảnh (VD: BLIP mô tả người, GT random mô tả nền) → tạo noise trong training label → ViT5 khó học mapping chính xác.

## Success Criteria

- [ ] Có BLEU-4 end-to-end của random variant
- [ ] Jaccard BLEU-4 > Random BLEU-4 (kỳ vọng)
- [ ] Bảng ablation hoàn chỉnh điền vào báo cáo
- [ ] Có giải thích lý do định tính trong báo cáo

## Risk: Nếu Random BLEU-4 cao hơn Jaccard?

Không sao — vẫn là kết quả ablation hợp lệ. Phân tích có thể là:
- 5 GT captions của UITViC đều mô tả cùng 1 ảnh → random vẫn phù hợp về nội dung
- Jaccard trên caption không dấu (BLIP output) vs có dấu (GT) → word overlap bị ảnh hưởng bởi dấu câu
- → Suggest cải tiến: dùng Jaccard sau khi strip dấu cả 2 phía

## Output Files

Sau phase này:
- `outputs/eval_random_results.json` — BLEU scores random variant
- Bảng ablation điền vào section 4.5 báo cáo
