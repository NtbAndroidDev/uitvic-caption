# Implementation Plan: Qwen2-VL-7B Fine-tuning — UITViC Vietnamese Captioning

**Date:** 2026-04-07
**Created By:** MAC <binh.nt@ots.vn>
**Status:** Draft
**Complexity:** High
**Estimated Effort:** 4–6 giờ (setup + training + eval trên Kaggle)

## Overview

Thử nghiệm fine-tune **Qwen2-VL-7B** (multimodal LLM) end-to-end trực tiếp từ ảnh → caption tiếng Việt, thay thế pipeline 2-stage (BLIP → ViT5). Mục tiêu: kiểm tra xem một mô hình VLM hiện đại có thể vượt qua baseline **BLEU-4 = 0.1951** hay không.

**Approach:** QLoRA 4-bit + Unsloth trên Kaggle GPU (T4 16GB) — single GPU, đủ VRAM.

**Git branch:** `experiment/qwen2-vl` (tách từ `main`)

## Phases

| Phase | File | Mô tả | Status |
|---|---|---|---|
| Phase 01 | [phase-01-setup.md](./phase-01-setup.md) | Git branch + môi trường + dataset prep | 🔲 Pending |
| Phase 02 | [phase-02-training.md](./phase-02-training.md) | Training script + Kaggle cell | 🔲 Pending |
| Phase 03 | [phase-03-evaluation.md](./phase-03-evaluation.md) | Eval BLEU-4 + so sánh baseline | 🔲 Pending |

## Kiến trúc So Sánh

| | BLIP + ViT5 (baseline) | Qwen2-VL-7B (thử nghiệm) |
|---|---|---|
| Approach | 2-stage pipeline | End-to-end |
| Stage 1 | BLIP → noisy caption | — |
| Stage 2 | ViT5 → clean caption | Qwen2-VL → direct caption |
| Params | ~400M (BLIP) + ~250M (ViT5) | ~7B (QLoRA adapter ~50M) |
| Input | Image | Image + instruction prompt |
| Output | Vietnamese caption (có dấu) | Vietnamese caption |
| BLEU-4 | **0.1951** | ? |

## Rủi ro chính

- Kaggle T4 16GB: cần QLoRA 4-bit + batch=1 + gradient_accumulation=8
- Qwen2-VL sinh caption đa ngôn ngữ → cần prompt template rõ ràng bằng tiếng Việt
- Thời gian training ước tính: ~2–3 giờ/epoch (v.s. ~18 phút/epoch ViT5)
- Kaggle GPU quota: ~30h/week — cần train nhanh (ít epochs, early stopping)
