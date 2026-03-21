import os
import yaml
import torch

def load_config(path: str):
    """
    Load configuration from YAML file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch: int, ckpt_dir: str):
    """
    Saves a model checkpoint and deletes old checkpoints to save disk space.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 1. Tự động dọn dẹp các checkpoint cũ (Chỉ giữ lại mốc hiện tại và mốc trước đó)
    # Vì mỗi file 3GB, nếu để dồn anh sẽ bị "trên 20GB" và crash máy ngay lập tức.
    for f in os.listdir(ckpt_dir):
        if f.startswith("blip_epoch_") and f.endswith(".pt"):
            try:
                # Trích xuất số epoch từ tên file (ví dụ: blip_epoch_10.pt -> 10)
                file_epoch = int(f.split("_")[-1].split(".")[0])
                if file_epoch < epoch - 1:
                    os.remove(os.path.join(ckpt_dir, f))
            except:
                pass
    
    # 2. Lưu checkpoint mới
    checkpoint_path = os.path.join(ckpt_dir, f"blip_epoch_{epoch}.pt")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print(f"[HELPERS] Checkpoint saved successfully at: {checkpoint_path}")

def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads weights and optimizer state from a checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Checkpoint not found at: {ckpt_path}")
        return 0
    
    print(f"[HELPERS] Initializing model weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if available
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    return checkpoint.get("epoch", 0)
