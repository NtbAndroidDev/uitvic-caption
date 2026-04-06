# Implementation Plan: Ablation Study — Jaccard vs Random Pairing (Stage 2)

**Date:** 2026-04-06
**Created By:** MAC <binh.nt@ots.vn>
**Status:** Draft
**Complexity:** Low
**Estimated Effort:** 1.5–2h code + ~5–7h Kaggle GPU

## Overview

Ablation study để chứng minh design choice: **Jaccard similarity matching tốt hơn random pairing** khi sinh training pairs cho Stage 2 (ViT5).

Baseline đã có: BLAP-ViC + ViT5 (Jaccard) → **BLEU-4 = 0.1951**

Variant cần làm: thay Jaccard bằng `random.choice()` → retrain ViT5 → so sánh BLEU-4.

## Key Insight

File `stage2_pairs_blip.jsonl` hiện tại chỉ lưu `{"noisy": "...", "clean": "..."}` — **không có image_id**.
→ Không thể tái sử dụng noisy captions để re-pair mà không chạy lại BLIP.
→ **Giải pháp:** Tạo script variant `generate_stage2_pairs_random.py` (clone generate_stage2_pairs.py),
   chỉ thay `best_matching_gt()` → `random.choice()`. Chạy lại trên Kaggle (~30–60min BLIP inference).

## Phases

| Phase | File | Mô tả | Ước tính |
|---|---|---|---|
| 01 | [phase-01-random-script.md](./phase-01-random-script.md) | Tạo script + config random variant | 30 phút code |
| 02 | [phase-02-train-random.md](./phase-02-train-random.md) | Chạy Kaggle notebook, train ViT5 random | ~6h GPU Kaggle |
| 03 | [phase-03-compare-results.md](./phase-03-compare-results.md) | Evaluation + bảng so sánh cho báo cáo | 30 phút |

## Expected Result

| Phương pháp | Kỳ vọng BLEU-4 | Ghi chú |
|---|---|---|
| Jaccard (baseline) | **0.1951** | Đã có |
| Random (variant) | < 0.1951 | Noise cao hơn → kỳ vọng tệ hơn |

→ Nếu Jaccard > Random: **validate design choice của đề tài**.

## Goals

- [x] Hiểu pipeline Stage 2
- [ ] Tạo `generate_stage2_pairs_random.py` (phase-01)
- [ ] Tạo `configs/stage2_vit5_random.yaml` (phase-01)
- [ ] Chạy random pair generation trên Kaggle (phase-02)
- [ ] Train ViT5 trên random pairs (phase-02)
- [ ] Evaluate + so sánh kết quả (phase-03)
- [ ] Viết bảng ablation vào báo cáo (phase-03)

## Next Steps

1. Review + approve plan
2. Implement phase 01: `/code plans/2026-04-06-ablation-jaccard-vs-random/phase-01-random-script.md`
3. Upload lên Kaggle, chạy training
4. Evaluate và viết báo cáo
