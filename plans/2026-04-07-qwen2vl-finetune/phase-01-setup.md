# Phase 01 — Setup: Git Branch + Môi trường + Dataset Preparation

**Parent:** [plan.md](./plan.md)
**Priority:** High | **Status:** 🔲 Pending

## Overview

Tạo nhánh git `experiment/qwen2-vl`, chuẩn bị dataset theo format Qwen2-VL, và thiết lập môi trường Kaggle.

## Implementation Steps

### Bước 1: Tạo git branch

```bash
git checkout main
git checkout -b experiment/qwen2-vl
git push -u origin experiment/qwen2-vl
```

Branch này hoàn toàn tách biệt với `main` → không ảnh hưởng pipeline hiện tại.

### Bước 2: Cấu trúc file mới (trong branch)

```
src/
  train_qwen2vl.py          ← training script mới (QLoRA + Unsloth)
scripts/
  prepare_qwen2vl_dataset.py ← convert UIT-ViIC → Qwen2-VL format
configs/
  qwen2vl_finetune.yaml     ← config: model, data, training params
```

### Bước 3: Dataset Format

Qwen2-VL yêu cầu format conversation (LLaVA-style):

```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<path_to_image>"},
        {"type": "text", "text": "Hãy mô tả hình ảnh này bằng tiếng Việt."}
      ]
    },
    {
      "role": "assistant",
      "content": "Một người đàn ông đang chơi tennis trên sân."
    }
  ]
}
```

Script `prepare_qwen2vl_dataset.py` sẽ:
1. Đọc `uitvic_captions_train2017.json` (COCO format)
2. Lấy 1 GT caption per ảnh (random với seed=42)
3. Xuất ra `data/qwen2vl_train.json` và `data/qwen2vl_test.json`

### Bước 4: Config file (`configs/qwen2vl_finetune.yaml`)

```yaml
model:
  name: "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

data:
  train_json: "/kaggle/working/data/qwen2vl_train.json"
  test_json: "/kaggle/working/data/qwen2vl_test.json"
  image_dir: "/kaggle/input/.../coco_uitvic_train"
  max_image_size: 768

training:
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8   # effective batch = 8
  num_epochs: 3
  learning_rate: 2e-4
  warmup_ratio: 0.03
  early_stopping_patience: 2       # BLEU-4 trên val

logging:
  ckpt_dir: "/kaggle/working/outputs/qwen2vl_checkpoints"
  log_every: 50
```

## Todo

- [ ] `git checkout -b experiment/qwen2-vl`
- [ ] Viết `scripts/prepare_qwen2vl_dataset.py`
- [ ] Viết `configs/qwen2vl_finetune.yaml`
- [ ] Push branch lên GitHub

## Risk Assessment

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| OOM trên Kaggle T4 16GB | Cao | Unsloth 4-bit + batch=1 + max_image=768 |
| Prompt tiếng Việt không stable | Trung bình | Test prompt trước với vài sample |
| Dataset format sai | Thấp | Validate script với 5 sample trước |
