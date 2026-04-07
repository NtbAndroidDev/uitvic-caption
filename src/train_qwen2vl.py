#!/usr/bin/env python3
"""
train_qwen2vl.py
Fine-tune Qwen2-VL-7B trên UIT-ViIC dataset với QLoRA 4-bit via Unsloth.
Usage: python -m src.train_qwen2vl --config configs/qwen2vl_finetune.yaml
"""
import os, json, csv, argparse, random
import torch
import numpy as np
from pathlib import Path

try:
    from unsloth import FastVisionModel, is_bf16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("[WARN] Unsloth not available — install: pip install unsloth[kaggle-new]")

from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from PIL import Image


# ── BLEU eval ────────────────────────────────────────────────────
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────
class Qwen2VLDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, max_image_size: int = 768):
        with open(json_path, encoding="utf-8") as f:
            self.records = json.load(f)
        self.max_image_size = max_image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image"]).convert("RGB")
        # Resize để tiết kiệm VRAM
        w, h = image.size
        if max(w, h) > self.max_image_size:
            scale = self.max_image_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        caption = rec["conversations"][1]["content"]
        gt_captions = rec.get("_gt_captions", [caption])
        return {"image": image, "caption": caption, "gt_captions": gt_captions}


# ── BLEU Evaluation ───────────────────────────────────────────────
def evaluate_bleu(model, tokenizer, dataset, device, max_samples: int = 0,
                  print_samples: bool = False) -> dict:
    print(f"[EVAL] Evaluating on {'full' if max_samples == 0 else max_samples} samples...")
    FastVisionModel.for_inference(model)
    model.eval()

    hypotheses, references = [], []
    n = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            image = item["image"]
            gt_captions = item["gt_captions"]

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."},
                ],
            }]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, images=[image], return_tensors="pt",
                padding=True, truncation=True
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Cắt phần input khỏi output
            gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            if print_samples and i < 4:
                print(f"  [{i+1}] GT  : {gt_captions[0]}")
                print(f"       Pred: {pred}")

            if pred:
                hypotheses.append(pred.split())
                references.append([gt.split() for gt in gt_captions])

    FastVisionModel.for_training(model)

    if not hypotheses:
        print("[EVAL] Warning: No predictions generated")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    smooth = SmoothingFunction().method1
    return {
        "bleu1": corpus_bleu(references, hypotheses, weights=(1,0,0,0),          smoothing_function=smooth),
        "bleu2": corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0),       smoothing_function=smooth),
        "bleu3": corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0),  smoothing_function=smooth),
        "bleu4": corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth),
    }


# ── Main Training ─────────────────────────────────────────────────
def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {device}")
    assert UNSLOTH_AVAILABLE, "Unsloth required: pip install unsloth[kaggle-new]"

    # Load model
    print(f"[TRAIN] Loading {cfg['model']['name']}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name     = cfg["model"]["name"],
        max_seq_length = cfg["model"]["max_seq_length"],
        load_in_4bit   = cfg["model"]["load_in_4bit"],
    )

    # Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = cfg["model"]["finetune_vision_layers"],
        finetune_language_layers   = cfg["model"]["finetune_language_layers"],
        finetune_attention_modules = cfg["model"]["finetune_attention_modules"],
        finetune_mlp_modules       = cfg["model"]["finetune_mlp_modules"],
        r           = cfg["training"]["lora_rank"],
        lora_alpha  = cfg["training"]["lora_alpha"],
        lora_dropout= cfg["training"]["lora_dropout"],
    )
    print("[TRAIN] LoRA adapters applied ✓")

    # Datasets
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    max_img = dcfg.get("max_image_size", 768)
    train_ds = Qwen2VLDataset(dcfg["train_json"], max_img)
    val_ds   = Qwen2VLDataset(dcfg["val_json"],   max_img)
    test_ds  = Qwen2VLDataset(dcfg["test_json"],  max_img)
    print(f"[TRAIN] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Collate fn for SFTTrainer ────────────────────────────────
    def collate_fn(batch):
        messages_list = []
        for item in batch:
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text",  "text": "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."},
                ],
            }, {
                "role": "assistant",
                "content": item["caption"],
            }])

        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_list
        ]
        images = [item["image"] for item in batch]
        inputs = tokenizer(
            texts, images=images, return_tensors="pt",
            padding=True, truncation=True,
            max_length=cfg["model"]["max_seq_length"]
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    # Training args
    args = TrainingArguments(
        output_dir                  = ckpt_dir,
        num_train_epochs            = tcfg["num_epochs"],
        per_device_train_batch_size = tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps = tcfg["gradient_accumulation_steps"],
        learning_rate               = tcfg["learning_rate"],
        weight_decay                = tcfg["weight_decay"],
        warmup_ratio                = tcfg["warmup_ratio"],
        fp16                        = tcfg.get("fp16", True),
        bf16                        = False,
        logging_steps               = cfg["logging"]["log_every"],
        save_strategy               = "epoch",
        evaluation_strategy         = "no",   # manual eval per epoch
        remove_unused_columns       = False,
        report_to                   = "none",
        dataloader_num_workers      = 0,
    )

    # ── Manual train loop với BLEU early stopping ────────────────
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    train_loader = DataLoader(train_ds, batch_size=tcfg["per_device_train_batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)

    n_steps = len(train_loader) * tcfg["num_epochs"] // tcfg["gradient_accumulation_steps"]
    optimizer = AdamW(model.parameters(), lr=tcfg["learning_rate"],
                      weight_decay=tcfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(tcfg["warmup_ratio"] * n_steps),
        num_training_steps=n_steps
    )

    history = {"epoch": [], "train_loss": [], "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
    best_bleu4 = -1.0
    patience_counter = 0
    patience = tcfg["early_stopping_patience"]

    FastVisionModel.for_training(model)

    for epoch in range(1, tcfg["num_epochs"] + 1):
        print(f"\n=== Qwen2-VL Epoch {epoch}/{tcfg['num_epochs']} ===")
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / tcfg["gradient_accumulation_steps"]
            loss.backward()
            epoch_loss += outputs.loss.item()

            if step % tcfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg["logging"]["log_every"] == 0:
                print(f"  Step [{step}/{len(train_loader)}], Loss: {outputs.loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        bleu = evaluate_bleu(model, tokenizer, val_ds, device,
                             print_samples=(epoch == 1))

        print(f"[REPORT] Epoch {epoch}: Loss={avg_loss:.4f} | "
              f"BLEU-1={bleu['bleu1']:.4f} BLEU-2={bleu['bleu2']:.4f} "
              f"BLEU-3={bleu['bleu3']:.4f} BLEU-4={bleu['bleu4']:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        for k in ["bleu1","bleu2","bleu3","bleu4"]:
            history[k].append(bleu[k])

        # Save CSV
        csv_path = os.path.join(ckpt_dir, "qwen2vl_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Epoch","Train_Loss","BLEU-1","BLEU-2","BLEU-3","BLEU-4"])
            for i in range(len(history["epoch"])):
                w.writerow([history["epoch"][i], history["train_loss"][i],
                             history["bleu1"][i], history["bleu2"][i],
                             history["bleu3"][i], history["bleu4"][i]])

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"qwen2vl_epoch_{epoch}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"[SAVE] Checkpoint saved: {ckpt_path}")

        # Early stopping
        if bleu["bleu4"] > best_bleu4:
            best_bleu4 = bleu["bleu4"]
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"[EARLY STOPPING] (+) BLEU-4 improved to {best_bleu4:.4f}. Best model saved.")
        else:
            patience_counter += 1
            print(f"[EARLY STOPPING] (-) BLEU-4 no improvement (Best: {best_bleu4:.4f}). "
                  f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] *** Stopped at epoch {epoch}. ***")
                break

    best_ep = history["bleu4"].index(max(history["bleu4"])) + 1
    print(f"\n✓ Training done | Best epoch: {best_ep} | Val BLEU-4: {max(history['bleu4']):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
