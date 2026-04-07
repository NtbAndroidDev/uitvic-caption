# Phase 02 — Training: Kaggle Cell + QLoRA Fine-tuning

**Parent:** [plan.md](./plan.md)
**Priority:** High | **Status:** 🔲 Pending

## Overview

Viết script training và Kaggle notebook cell để fine-tune Qwen2-VL-7B với QLoRA 4-bit thông qua Unsloth.

## Kaggle Cell (chạy trong 1 cell duy nhất)

```python
# ================================================================
# EXPERIMENT: Qwen2-VL-7B Fine-tuning — UITViC Vietnamese Captioning
# Branch: experiment/qwen2-vl
# So sánh với baseline: BLIP+ViT5 BLEU-4=0.1951
# ================================================================
import os, sys, json, subprocess, glob, time

ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    r = subprocess.run(cmd, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"❌ Failed: {' '.join(str(c) for c in cmd)}")

# ── Paths ────────────────────────────────────────────────────────
REPO     = "/kaggle/working/uitvic-captioning"
BRANCH   = "experiment/qwen2-vl"
OUT_DIR  = "/kaggle/working/outputs/qwen2vl_checkpoints"
DATA_DIR = "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Pull branch ────────────────────────────────────────────────
if os.path.exists(REPO):
    subprocess.run(["git","-C",REPO,"fetch","origin"], check=True, env=ENV)
    subprocess.run(["git","-C",REPO,"checkout", BRANCH], check=True, env=ENV)
    subprocess.run(["git","-C",REPO,"reset","--hard",f"origin/{BRANCH}"], check=True, env=ENV)
else:
    subprocess.run(["git","clone","-b", BRANCH,
        "https://github.com/NtbAndroidDev/uitvic-caption", REPO], check=True, env=ENV)
os.chdir(REPO)
print("✓", subprocess.check_output(["git","log","--oneline","-2"]).decode().strip())

# ── 2. Install Unsloth (nếu chưa có) ────────────────────────────
try:
    import unsloth
    print("✓ Unsloth đã có")
except ImportError:
    print("Installing Unsloth...")
    subprocess.run([sys.executable, "-m", "pip", "install",
        "unsloth[kaggle-new]", "-q", "--no-deps"], check=True)
    print("✓ Unsloth installed")

# ── 3. Prepare dataset ───────────────────────────────────────────
import glob as g
TRAIN_JSON = g.glob("/kaggle/input/**/uitvic_captions_train2017.json", recursive=True)[0]
TEST_JSON  = g.glob("/kaggle/input/**/uitvic_captions_test2017.json",  recursive=True)[0]
TRAIN_IMGS = next(r for root,dirs,_ in os.walk("/kaggle/input")
                  for d in dirs if d=="coco_uitvic_train"
                  for r in [os.path.join(root,d)])
TEST_IMGS  = next(r for root,dirs,_ in os.walk("/kaggle/input")
                  for d in dirs if d=="coco_uitvic_test"
                  for r in [os.path.join(root,d)])

run([sys.executable, "scripts/prepare_qwen2vl_dataset.py",
    "--train_json", TRAIN_JSON, "--test_json", TEST_JSON,
    "--train_images", TRAIN_IMGS, "--test_images", TEST_IMGS,
    "--output_dir", DATA_DIR, "--seed", "42"])

# ── 4. Train ─────────────────────────────────────────────────────
t0 = time.time()
run([sys.executable, "-m", "src.train_qwen2vl",
    "--config", "configs/qwen2vl_finetune.yaml"])
print(f"✓ Training xong | {(time.time()-t0)/60:.1f} phút")

# ── 5. Evaluate ──────────────────────────────────────────────────
run([sys.executable, "scripts/evaluate_qwen2vl.py",
    "--model_dir", OUT_DIR,
    "--test_json", TEST_JSON, "--test_images", TEST_IMGS,
    "--output_dir", "/kaggle/working/outputs/eval_qwen2vl"])
```

## Training Script (`src/train_qwen2vl.py`) — Key Architecture

```python
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

model, tokenizer = FastVisionModel.from_pretrained(
    model_name  = cfg["model"]["name"],   # unsloth/Qwen2-VL-7B-Instruct-bnb-4bit
    max_seq_length = cfg["model"]["max_seq_length"],
    load_in_4bit = True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = False,   # chỉ fine-tune language layers
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = cfg["training"]["lora_rank"],
    lora_alpha = cfg["training"]["lora_alpha"],
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs = cfg["training"]["num_epochs"],
        learning_rate = cfg["training"]["learning_rate"],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        metric_for_best_model = "eval_bleu4",
        load_best_model_at_end = True,
    ),
    callbacks = [EarlyStoppingCallback(
        early_stopping_patience = cfg["training"]["early_stopping_patience"]
    )],
)
```

## Instruction Prompt Template

```python
PROMPT = """<|im_start|>user
<|vision_start|>{image}<|vision_end|>
Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt.<|im_end|>
<|im_start|>assistant
{caption}<|im_end|>"""
```

## Todo

- [ ] Viết `scripts/prepare_qwen2vl_dataset.py`
- [ ] Viết `src/train_qwen2vl.py` (Unsloth + SFTTrainer)
- [ ] Viết `configs/qwen2vl_finetune.yaml`
- [ ] Test trên 10 samples trước (`--max_samples 10`)
- [ ] Chạy full training trên Kaggle

## Risk & Mitigation

| Rủi ro | Xử lý |
|---|---|
| OOM T4 16GB | batch=1, image≤768px, gradient_accum=8 |
| Unsloth install fail | pip install với `--no-deps` fallback |
| Training > 9h Kaggle limit | Dùng 3 epochs thay 30, checkpoint mỗi epoch |
| Output không phải tiếng Việt | Prompt rõ ràng + validate 5 sample đầu |
