# src/train.py
import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .dataset import UITViCCOCODataset
from .models.blip_captioner import BlipViCaptioner
from .utils.helpers import load_config, save_checkpoint
from .utils.seed import set_seed


def train(config_path: str):
    cfg = load_config(config_path)
    set_seed(42)

    # xử lý device
    if cfg["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device(cfg["training"]["device"])
    
    # 1. Model + processor
    captioner = BlipViCaptioner(cfg["model"]["name"])
    processor = captioner.get_processor()
    model = captioner.get_model().to(device)

    # 2. Dataset & DataLoader
    train_dataset = UITViCCOCODataset(
        json_path=cfg["data"]["train_json"],
        image_root=cfg["data"]["train_image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
    )
    val_dataset = UITViCCOCODataset(
        json_path=cfg["data"]["val_json"],
        image_root=cfg["data"]["val_image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # 3. Optimizer + scheduler
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_training_steps = cfg["training"]["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0

    print("Using device:", device)
    print("Train size:", len(train_dataset), "Val size:", len(val_dataset))
    print("Batches per epoch:", len(train_loader))

    # History tracking
    history = {'train_loss': [], 'val_loss': []}
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['training']['num_epochs']} ===")
        running_loss = 0.0
        epoch_train_loss = 0.0
        
        model.train() # Ensure model is in train mode
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            if cfg["training"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["grad_clip"]
                )

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_train_loss += loss_val
            global_step += 1

            if i % cfg["logging"]["log_every"] == 0:
                avg_loss = running_loss / cfg["logging"]["log_every"]
                print(
                    f"Epoch [{epoch}/{cfg['training']['num_epochs']}], "
                    f"Step [{i}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

        # Calculate average train loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation Loop
        print("Running validation...")
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                epoch_val_loss += outputs.loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save history to JSON
        history_path = os.path.join(ckpt_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, "loss_chart.png"))
        plt.close()
        print(f"Saved loss chart to {os.path.join(ckpt_dir, 'loss_chart.png')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_blip.yaml")
    args = parser.parse_args()

    train(args.config)
