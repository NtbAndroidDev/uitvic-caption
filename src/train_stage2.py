import os
import json
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from .stage2_dataset import TextCorrectionDataset
from .utils.helpers import load_config, save_checkpoint
from .utils.seed import set_seed

def evaluate_bleu(model, dataloader, tokenizer, device, max_samples=200):
    """
    Tính điểm BLEU-1,2,3,4 cho ViT5 trên tập Validation.
    """
    model.eval()
    references = []
    hypotheses = []

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    print(f"[EVAL] Evaluating ViT5 on {max_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples // dataloader.batch_size:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate corrected captions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Ground truth
            labels = batch["labels"].clone()
            labels[labels == -100] = pad_id
            gt = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            for p, g in zip(preds, gt):
                p_clean = p.strip()
                g_clean = g.strip()
                if p_clean and g_clean:  # Bỏ qua cặp rỗng
                    hypotheses.append(p_clean.split())
                    references.append([g_clean.split()])

    if not hypotheses:
        print("[EVAL] Warning: No valid predictions, returning 0 scores")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    return {"bleu1": b1, "bleu2": b2, "bleu3": b3, "bleu4": b4}

def train_stage2(config_path: str):
    cfg = load_config(config_path)
    set_seed(42)
    
    # Choose device
    if cfg["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["training"]["device"])
        
    print(f"[STAGE 2] Training ViT5 on device: {device}")

    # Load tokenizer qua sentencepiece trực tiếp để tránh bug KeyError: 0
    # trên Python 3.12 với VietAI/vit5-base
    from huggingface_hub import hf_hub_download
    from transformers import T5Tokenizer
    sp_model_path = hf_hub_download(repo_id=cfg["model"]["name"], filename="spiece.model")
    tokenizer = T5Tokenizer(vocab_file=sp_model_path, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["name"]).to(device)

    # Load pairs dataset
    full_dataset = TextCorrectionDataset(
        cfg["data"]["train_pairs"],
        tokenizer,
        cfg["data"]["max_source_length"],
        cfg["data"]["max_target_length"],
    )
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    num_training_steps = cfg["training"]["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

    history = {'train_loss': [], 'val_loss': [], 'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_bleu4 = -1.0
    patience = cfg["training"].get("early_stopping_patience", 5)
    patience_counter = 0
    
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        print(f"\n=== Stage 2 Epoch {epoch}/{cfg['training']['num_epochs']} ===")
        model.train()
        epoch_train_loss = 0.0
        
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            if cfg["training"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            if i % cfg["logging"]["log_every"] == 0:
                print(f"Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                epoch_val_loss += model(**batch).loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        bleu = evaluate_bleu(model, val_loader, tokenizer, device)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['bleu1'].append(bleu['bleu1'])
        history['bleu2'].append(bleu['bleu2'])
        history['bleu3'].append(bleu['bleu3'])
        history['bleu4'].append(bleu['bleu4'])
        
        # Save CSV Report
        csv_path = os.path.join(ckpt_dir, "stage2_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
            for e in range(len(history['train_loss'])):
                writer.writerow([e+1,
                                 history['train_loss'][e], history['val_loss'][e],
                                 history['bleu1'][e], history['bleu2'][e],
                                 history['bleu3'][e], history['bleu4'][e]])

        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        
        # Save Chart - 3 biểu đồ chuẩn báo cáo NCKH
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(18, 5))

        # Biểu đồ 1: Train Loss vs Validation Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], 'r-s', label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Biểu đồ 2: BLEU-4
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, history['bleu4'], 'g-o', label='BLEU-4')
        plt.title('Validation BLEU-4')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        # Biểu đồ 3: Tất cả BLEU-1,2,3,4
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, history['bleu1'], label='BLEU-1')
        plt.plot(epochs_range, history['bleu2'], label='BLEU-2')
        plt.plot(epochs_range, history['bleu3'], label='BLEU-3')
        plt.plot(epochs_range, history['bleu4'], label='BLEU-4')
        plt.title('BLEU-1/2/3/4 Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "stage2_report.png"), dpi=150)
        plt.close()
        
        print(f"[REPORT] Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, BLEU-1 {bleu['bleu1']:.4f}, BLEU-2 {bleu['bleu2']:.4f}, BLEU-3 {bleu['bleu3']:.4f}, BLEU-4 {bleu['bleu4']:.4f}")

        # --- Early Stopping ---
        if bleu['bleu4'] > best_bleu4:
            best_bleu4 = bleu['bleu4']
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_ckpt_path)
            patience_counter = 0
            print(f"[EARLY STOPPING] (+) BLEU-4 improved to {best_bleu4:.4f}. Best model saved.")
        else:
            patience_counter += 1
            print(f"[EARLY STOPPING] (-) BLEU-4 did not improve (Best: {best_bleu4:.4f}). Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"[EARLY STOPPING] *** Training stopped early at epoch {epoch}. ***")
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_vit5.yaml")
    args = parser.parse_args()
    train_stage2(args.config)
