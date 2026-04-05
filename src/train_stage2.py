import os
import json
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
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
    
    print(f"[EVAL] Evaluating ViT5 on {max_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples // dataloader.batch_size:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate corrected captions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Ground truth
            labels = batch["labels"].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            gt = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            for p, g in zip(preds, gt):
                hypotheses.append(p.strip().split())
                references.append([g.strip().split()])

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

    tokenizer = T5Tokenizer.from_pretrained(cfg["model"]["name"], legacy=False)
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

    history = {'train_loss': [], 'val_loss': [], 'val_bleu4': []}
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_bleu4 = 0.0
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
        history['val_bleu4'].append(bleu['bleu4'])
        
        # Save CSV Report (FOR THE USER REPORT)
        csv_path = os.path.join(ckpt_dir, "stage2_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
            for e in range(len(history['train_loss'])):
                writer.writerow([e+1, history['train_loss'][e], history['val_loss'][e],
                                 bleu['bleu1'], bleu['bleu2'], bleu['bleu3'], history['val_bleu4'][e]])

        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        
        # Save Chart
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['val_bleu4'], label='BLEU-4', color='green')
        plt.title('Validation BLEU-4')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "stage2_report.png"))
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
