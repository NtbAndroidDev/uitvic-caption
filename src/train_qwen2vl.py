#!/usr/bin/env python3
"""
train_qwen2vl.py — Qwen2-VL-7B QLoRA Fine-tuning (Unsloth)
Usage: python -m src.train_qwen2vl --config configs/qwen2vl_finetune.yaml
"""
import os, json, csv, argparse, random
import torch
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    UNSLOTH = True
except ImportError:
    UNSLOTH = False
    print("[WARN] Unsloth not found. Install: pip install unsloth[kaggle-new]")

from transformers import TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer, SFTConfig
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Instruction ─────────────────────────────────────────────────
INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."


def make_conversation(record: dict) -> dict:
    """Convert record → Unsloth conversation format."""
    image = Image.open(record["image"]).convert("RGB")
    caption = record["conversations"][1]["content"]
    return {
        "messages": [
            {"role": "user",      "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": INSTRUCTION},
            ]},
            {"role": "assistant", "content": caption},
        ],
        # Keep for BLEU evaluation
        "_image": image,
        "_gt_captions": record.get("_gt_captions", [caption]),
    }


def load_hf_dataset(json_path: str, max_image_size: int = 768):
    """Load JSON → list of records.
    messages内 image field lưu path string (không phải PIL) → HFDataset serialize được.
    PIL chỉ load trong evaluate_bleu.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for r in raw:
        img_path = r["image"]
        caption  = r["conversations"][1]["content"]
        records.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},   # ← path string, NOT PIL
                    {"type": "text",  "text": INSTRUCTION},
                ]},
                {"role": "assistant", "content": caption},
            ],
            "_image_path": img_path,
            "_gt_captions": r.get("_gt_captions", [caption]),
            "_max_image_size": max_image_size,
        })
    return records


# ─── BLEU Evaluator ──────────────────────────────────────────────
def _load_image(rec: dict) -> Image.Image:
    """Load + resize image from _image_path."""
    img = Image.open(rec["_image_path"]).convert("RGB")
    max_sz = rec.get("_max_image_size", 768)
    w, h = img.size
    if max(w, h) > max_sz:
        scale = max_sz / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def evaluate_bleu(model, tokenizer, records: list, device: str,
                  print_samples: bool = False, max_samples: int = 0) -> dict:
    """Compute BLEU-1/2/3/4 on val/test records."""
    n = len(records) if max_samples == 0 else min(max_samples, len(records))
    print(f"[EVAL] Evaluating on {n} samples...")
    FastVisionModel.for_inference(model)
    model.eval()

    hypotheses, references = [], []

    with torch.no_grad():
        for i, rec in enumerate(records[:n]):
            image   = _load_image(rec)              # always load from file path
            gt_caps = rec["_gt_captions"]

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": INSTRUCTION},
            ]}]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, images=[image], return_tensors="pt",
                padding=True, truncation=True
            ).to(device)

            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            if print_samples and i < 4:
                print(f"  [{i+1}]")
                print(f"    GT  : {gt_caps[0]}")
                print(f"    Pred: {pred}")

            if pred:
                hypotheses.append(pred.split())
                references.append([g.split() for g in gt_caps])

    FastVisionModel.for_training(model)

    if not hypotheses:
        print("[EVAL] ⚠️  No valid predictions!")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    smooth = SmoothingFunction().method1
    return {
        "bleu1": corpus_bleu(references, hypotheses, weights=(1,0,0,0),           smoothing_function=smooth),
        "bleu2": corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0),        smoothing_function=smooth),
        "bleu3": corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0),   smoothing_function=smooth),
        "bleu4": corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth),
    }


# ─── BLEU Early Stopping Callback ────────────────────────────────
import time as _time

class BLEUEarlyStoppingCallback(TrainerCallback):
    """Evaluate BLEU-4 sau mỗi epoch, dừng nếu không cải thiện. Có Kaggle time budget guard."""

    def __init__(self, model, tokenizer, val_records, ckpt_dir,
                 patience: int = 3, device: str = "cuda",
                 val_eval_samples: int = 0):
        self.model            = model
        self.tokenizer        = tokenizer
        self.val_records      = val_records
        self.ckpt_dir         = ckpt_dir
        self.patience         = patience
        self.device           = device
        self.val_eval_samples = val_eval_samples  # 0 = full val set
        self.best_bleu4       = -1.0
        self.counter          = 0
        self.history          = []
        self._csv_path        = os.path.join(ckpt_dir, "qwen2vl_metrics.csv")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch","Train_Loss","BLEU-1","BLEU-2","BLEU-3","BLEU-4"])

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch)
        print_s = (epoch == 1)

        bleu = evaluate_bleu(self.model, self.tokenizer, self.val_records,
                              self.device, print_samples=print_s,
                              max_samples=self.val_eval_samples)
        # Get latest train loss (search backwards for 'loss' key)
        train_loss = 0.0
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break
        self.history.append({"epoch": epoch, "train_loss": train_loss, **bleu})

        n_eval = len(self.val_records) if self.val_eval_samples == 0 else self.val_eval_samples
        print(f"[REPORT] Epoch {epoch}: Loss={train_loss:.4f} "
              f"| BLEU-1={bleu['bleu1']:.4f} BLEU-2={bleu['bleu2']:.4f} "
              f"BLEU-3={bleu['bleu3']:.4f} BLEU-4={bleu['bleu4']:.4f} "
              f"(eval on {n_eval} samples)")

        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, bleu["bleu1"], bleu["bleu2"],
                                    bleu["bleu3"], bleu["bleu4"]])

        # Save epoch checkpoint
        ep_path = os.path.join(self.ckpt_dir, f"qwen2vl_epoch_{epoch}")
        self.model.save_pretrained(ep_path)
        self.tokenizer.save_pretrained(ep_path)
        print(f"[SAVE] Checkpoint: {ep_path}")

        # Early stopping check
        if bleu["bleu4"] > self.best_bleu4:
            self.best_bleu4 = bleu["bleu4"]
            self.counter = 0
            best_path = os.path.join(self.ckpt_dir, "best_model")
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            print(f"[EARLY STOPPING] (+) BLEU-4 improved → {self.best_bleu4:.4f}. Best saved.")
        else:
            self.counter += 1
            print(f"[EARLY STOPPING] (-) No improvement (Best: {self.best_bleu4:.4f}). "
                  f"Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"[EARLY STOPPING] *** Stopping training. ***")
                control.should_training_stop = True

        return control


# ─── Main ─────────────────────────────────────────────────────────
def train(config_path: str):
    assert UNSLOTH, "Install Unsloth: pip install unsloth[kaggle-new]"
    cfg  = load_config(config_path)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Device: {device}")

    # Load model + LoRA
    print(f"[TRAIN] Loading {cfg['model']['name']}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name              = cfg["model"]["name"],
        max_seq_length          = cfg["model"]["max_seq_length"],
        load_in_4bit            = cfg["model"]["load_in_4bit"],
        use_gradient_checkpointing = "unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = cfg["model"]["finetune_vision_layers"],
        finetune_language_layers   = cfg["model"]["finetune_language_layers"],
        finetune_attention_modules = cfg["model"]["finetune_attention_modules"],
        finetune_mlp_modules       = cfg["model"]["finetune_mlp_modules"],
        r            = cfg["training"]["lora_rank"],
        lora_alpha   = cfg["training"]["lora_alpha"],
        lora_dropout = cfg["training"]["lora_dropout"],
        use_gradient_checkpointing = "unsloth",
    )
    print("[TRAIN] Model + LoRA ready ✓")

    # Load datasets
    dcfg     = cfg["data"]
    max_img  = dcfg.get("max_image_size", 768)
    tcfg     = cfg["training"]
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print("[TRAIN] Loading datasets...")
    train_records = load_hf_dataset(dcfg["train_json"], max_img)
    val_records   = load_hf_dataset(dcfg["val_json"],   max_img)
    print(f"  Train: {len(train_records)} | Val: {len(val_records)}")

    # Custom Dataset — tránh pyarrow mixed-type serialization bug
    class Qwen2VLDataset(torch.utils.data.Dataset):
        def __init__(self, records): self.records = records
        def __len__(self):           return len(self.records)
        def __getitem__(self, idx):  return {"messages": self.records[idx]["messages"]}

    train_ds = Qwen2VLDataset(train_records)
    print(f"[TRAIN] Dataset ready: {len(train_ds)} train samples")

    # Callbacks
    bleu_cb = BLEUEarlyStoppingCallback(
        model=model, tokenizer=tokenizer, val_records=val_records,
        ckpt_dir=ckpt_dir,
        patience=tcfg["early_stopping_patience"],
        device=device,
        val_eval_samples=tcfg.get("val_eval_samples", 0),  # 0 = full val set
    )

    # SFTConfig
    sft_args = SFTConfig(
        output_dir                  = ckpt_dir,
        num_train_epochs            = tcfg["num_epochs"],
        per_device_train_batch_size = tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps = tcfg["gradient_accumulation_steps"],
        learning_rate               = float(tcfg["learning_rate"]),
        weight_decay                = tcfg["weight_decay"],
        warmup_ratio                = tcfg["warmup_ratio"],
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = cfg["logging"]["log_every"],
        save_strategy               = "no",
        eval_strategy               = "no",
        remove_unused_columns       = False,
        report_to                   = "none",
        dataset_text_field          = "",
        dataset_kwargs              = {"skip_prepare_dataset": True},
        max_seq_length              = cfg["model"]["max_seq_length"],
        dataloader_num_workers      = 0,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_ds,
        args          = sft_args,
        callbacks     = [bleu_cb],
    )

    trainer.train()

    # Fallback: ensuite best_model est sauvegardé même si BLEU-4 = 0 à toutes les époques
    best_path = os.path.join(ckpt_dir, "best_model")
    if not os.path.exists(best_path):
        print("[SAVE] ⚠️  best_model not found — saving final model as best_model")
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        print(f"[SAVE] ✓ Saved final model to {best_path}")

    # Summary
    if bleu_cb.history:
        best_ep = max(bleu_cb.history, key=lambda x: x["bleu4"])
        print(f"\n✓ Training done | Best epoch: {best_ep['epoch']} | "
              f"Val BLEU-4: {best_ep['bleu4']:.4f}")
        print("\nTraining History:")
        print(f"  {'Epoch':>5}  {'BLEU-1':>7}  {'BLEU-2':>7}  {'BLEU-3':>7}  {'BLEU-4':>7}")
        for h in bleu_cb.history:
            print(f"  {h['epoch']:>5}  {h['bleu1']:.4f}  {h['bleu2']:.4f}  {h['bleu3']:.4f}  {h['bleu4']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
