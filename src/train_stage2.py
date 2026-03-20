import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from .stage2_dataset import TextCorrectionDataset
from .utils.helpers import load_config, save_checkpoint
from .utils.seed import set_seed


def choose_device(cfg_device: str):
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(cfg_device)


def train_stage2(config_path: str):
    cfg = load_config(config_path)
    set_seed(42)

    device = choose_device(cfg["training"]["device"])
    print("Stage2 device:", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["name"]).to(device)

    # nếu có init_ckpt thì load trọng số từ checkpoint cũ
    init_ckpt = cfg["training"].get("init_ckpt", None)
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"[Stage2] Loading initial weights from {init_ckpt}")
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    elif init_ckpt:
        print(f"[WARNING] Initial checkpoint {init_ckpt} not found. Starting from scratch.")

    full_dataset = TextCorrectionDataset(
        cfg["data"]["train_pairs"],
        tokenizer,
        cfg["data"]["max_source_length"],
        cfg["data"]["max_target_length"],
    )
    
    # Split dataset into train (90%) and val (10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

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

    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    num_training_steps = cfg["training"]["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    global_step = 0
    
    # History tracking
    history = {'train_loss': [], 'val_loss': []}
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        print(f"\n=== Stage2 Epoch {epoch}/{cfg['training']['num_epochs']} ===")
        running_loss = 0.0
        epoch_train_loss = 0.0
        
        model.train()
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            if cfg["training"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_train_loss += loss_val
            global_step += 1

            if i % cfg["logging"]["log_every"] == 0:
                avg_loss = running_loss / cfg["logging"]["log_every"]
                print(f"Step {i}/{len(train_loader)}, loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Calculate average train loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation Loop
        print("Running validation...")
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                epoch_val_loss += outputs.loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save history
        history_path = os.path.join(ckpt_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(model, optimizer, epoch, ckpt_dir)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Stage 2 Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_dir, "loss_chart.png"))
        plt.close()
        print(f"Saved loss chart to {os.path.join(ckpt_dir, 'loss_chart.png')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_vit5.yaml")
    args = parser.parse_args()

    train_stage2(args.config)
