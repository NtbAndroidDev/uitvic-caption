# Phase 03 — Evaluation & Comparison

**Parent:** [plan.md](./plan.md)
**Priority:** High | **Status:** 🔲 Pending

## Overview

Đánh giá BLEU-4 của Qwen2-VL-7B trên full test set UIT-ViIC và so sánh với baseline pipeline BLIP+ViT5.

## Evaluation Script (`scripts/evaluate_qwen2vl.py`)

```python
# Tương tự evaluate_pipeline.py nhưng dùng Qwen2-VL trực tiếp
# Input: ảnh → Instruction prompt → Caption tiếng Việt
# Output: pipeline_evaluation.csv (cùng format với baseline)

from unsloth import FastVisionModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch, json, os

def evaluate_qwen2vl(model_dir, test_json, test_images, output_dir, batch_size=4):
    model, tokenizer = FastVisionModel.from_pretrained(model_dir, load_in_4bit=True)
    FastVisionModel.for_inference(model)

    hypotheses, references = [], []
    for image_path, gt_captions in test_data:
        pred = generate_caption(model, tokenizer, image_path)
        hypotheses.append(pred.split())
        references.append([gt.split() for gt in gt_captions])  # 5 GTs mỗi ảnh

    smooth = SmoothingFunction().method1
    bleu4 = corpus_bleu(references, hypotheses,
                        weights=(0.25,0.25,0.25,0.25),
                        smoothing_function=smooth)
    # Lưu CSV + qualitative examples
```

## Báo cáo so sánh cuối cùng

```
══════════════════════════════════════════════════════════
    SO SÁNH: BLIP+ViT5 (2-stage) vs Qwen2-VL-7B (end-to-end)
══════════════════════════════════════════════════════════
Phương pháp              BLEU-1  BLEU-2  BLEU-3  BLEU-4
──────────────────────────────────────────────────────────
BLIP Only (Stage 1)      0.0943  0.0193  0.0032  0.0007
BLIP+ViT5 (Jaccard) ←   0.4552  0.3196  0.2493  0.1951
Qwen2-VL-7B 4bit-QLoRA  ?       ?       ?       ?
══════════════════════════════════════════════════════════
```

## Kịch bản kết quả & Hành động

| Kết quả Qwen2-VL | Ý nghĩa | Hành động |
|---|---|---|
| BLEU-4 >> 0.1951 | VLM end-to-end vượt trội | Thảo luận trong báo cáo, đề xuất hướng tương lai |
| BLEU-4 ≈ 0.1951 | Tương đương nhưng đơn giản hơn | Nhấn mạnh efficiency của pipeline cũ |
| BLEU-4 < 0.1951 | Pipeline 2-stage hiệu quả hơn | Xác nhận approach hiện tại, đóng branch |

## Metric bổ sung (nếu có thời gian)

- **Qualitative examples:** 10 ảnh, so sánh caption 3 phương pháp
- **Inference speed:** thời gian sinh 1 caption (ms)
- **Model size:** disk size của checkpoint

## Todo

- [ ] Viết `scripts/evaluate_qwen2vl.py`
- [ ] Chạy eval trên full test set (~3,000 ảnh)
- [ ] So sánh BLEU-4 với baseline
- [ ] Viết kết quả vào `plans/2026-04-07-qwen2vl-finetune/reports/results.md`
- [ ] Quyết định: merge vào main hay giữ branch riêng

## Success Criteria

- [ ] Training hoàn thành không OOM trên Kaggle T4
- [ ] eval BLEU-4 được tính trên full test set (≥ 2,000 ảnh)
- [ ] Có kết quả so sánh rõ ràng với baseline 0.1951
- [ ] Kết quả được ghi lại trong plans/ để dùng cho báo cáo
