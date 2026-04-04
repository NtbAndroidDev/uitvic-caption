import os
import json
import csv
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from .dataset import UITViCCOCODataset
from .models.blip_captioner import BlipViCaptioner
from .utils.helpers import load_config, save_checkpoint, load_checkpoint
from .utils.seed import set_seed

def evaluate_metrics(model, dataloader, processor, device, max_samples=100):
    """
    Tính toán các chỉ số BLEU-1, 2, 3, 4 cho tập Validation.
    """
    model.eval()
    references = []
    hypotheses = []
    
    print(f"[EVAL] Evaluating on {max_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples // dataloader.batch_size:
                break
                
            pixel_values = batch["pixel_values"].to(device)
            # Generate captions
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Ground truth
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            gt = processor.batch_decode(labels, skip_special_tokens=True)
            
            # Chuẩn bị dữ liệu cho corpus_bleu
            for p, g in zip(preds, gt):
                hypotheses.append(p.strip().split())
                references.append([g.strip().split()])

    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    return {"bleu1": b1, "bleu2": b2, "bleu3": b3, "bleu4": b4}

def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize
    captioner = BlipViCaptioner(cfg["model"]["name"])
    processor = captioner.get_processor()
    model = captioner.get_model().to(device)
    
    # Dataset & Dataloader
    train_dataset = UITViCCOCODataset(cfg["data"]["train_json"], cfg["data"]["train_image_root"], processor, cfg["data"]["max_length"])
    val_dataset = UITViCCOCODataset(cfg["data"]["val_json"], cfg["data"]["val_image_root"], processor, cfg["data"]["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    
    # Resume Checkpoint Logic
    start_epoch = 1
    history = {'train_loss': [], 'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load from config checkpoint (for manual resume)
    manual_ckpt = cfg["model"].get("checkpoint")
    if manual_ckpt and os.path.exists(manual_ckpt):
        start_epoch = load_checkpoint(manual_ckpt, model, optimizer, None, device) + 1
        print(f"[RESUME] Manually resumed from {manual_ckpt} at Epoch {start_epoch}")
        
    # Auto-load latest history if exists
    history_path = os.path.join(ckpt_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            start_epoch = len(history['train_loss']) + 1
            print(f"[RESUME] History found. Starting from Epoch {start_epoch}")

    num_training_steps = cfg["training"]["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * num_training_steps), num_training_steps=num_training_steps)

    print(f"[INFO] Starting training on {device}. Total Epochs: {cfg['training']['num_epochs']}")

    best_bleu4 = 0.0
    patience = cfg["training"].get("early_stopping_patience", 5)
    patience_counter = 0

    for epoch in range(start_epoch, cfg["training"]["num_epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['training']['num_epochs']} ===")
        model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            if i % cfg["logging"]["log_every"] == 0:
                print(f"Step [{i}/{len(train_loader)}], Current Loss: {loss.item():.4f}")
        
        # --- End of Epoch Evaluation ---
        avg_train_loss = epoch_loss / len(train_loader)
        metrics = evaluate_metrics(model, val_loader, processor, device, max_samples=200)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['bleu1'].append(metrics['bleu1'])
        history['bleu2'].append(metrics['bleu2'])
        history['bleu3'].append(metrics['bleu3'])
        history['bleu4'].append(metrics['bleu4'])
        
        # Save JSON history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
            
        # Save CSV metrics (For Excel Reporting)
        csv_path = os.path.join(ckpt_dir, "evaluation_report.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
            for e in range(len(history['train_loss'])):
                writer.writerow([
                    e + 1, 
                    history['train_loss'][e], 
                    history['bleu1'][e], 
                    history['bleu2'][e], 
                    history['bleu3'][e], 
                    history['bleu4'][e]
                ])

        # --- Early Stopping & Best Checkpoint Logic ---
        current_bleu4 = metrics['bleu4']
        if current_bleu4 > best_bleu4:
            best_bleu4 = current_bleu4
            print(f"[EARLY STOPPING] \u2705 BLEU-4 improved to {best_bleu4:.4f}. Resetting patience.")
            # Save the 'best' model checkpoint
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[EARLY STOPPING] \u26a0\ufe0f BLEU-4 did not improve (Best: {best_bleu4:.4f}). Patience: {patience_counter}/{patience}")

        # Regular periodic save
        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        
        # Plot and save charts
        plt.figure(figsize=(12, 5))
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-o', label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # Plot BLEU
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history['bleu4']) + 1), history['bleu4'], 'r-s', label='BLEU-4')
        plt.title('Validation BLEU-4')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "training_report.png"))
        plt.close()
        
        print(f"[REPORT] Epoch {epoch} complete. Loss: {avg_train_loss:.4f}, BLEU-4: {metrics['bleu4']:.4f}")
        print(f"[REPORT] Results saved in {ckpt_dir}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] \ud83d\udea8 Training stopped early at epoch {epoch} as BLEU-4 didn't improve for {patience} epochs.")
            break
