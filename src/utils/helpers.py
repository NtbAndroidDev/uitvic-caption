import os
import yaml
import torch

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch: int, ckpt_dir: str):
    """
    Tự động lưu checkpoint và dọn dẹp các mốc cũ để tiết kiệm 20GB Disk Space.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 1. Xác định prefix (blip hoặc stage2) dựa trên ckpt_dir
    prefix = "stage2" if "stage2" in ckpt_dir.lower() else "blip"
    
    # 2. Dọn dẹp các checkpoint cũ (Chỉ giữ lại mốc hiện tại)
    for f in os.listdir(ckpt_dir):
        if f.startswith(f"{prefix}_epoch_") and f.endswith(".pt"):
            try:
                # Trích xuất số epoch từ file
                file_epoch = int(f.split("_")[-1].split(".")[0])
                if file_epoch < epoch: # Xóa sạch các bản cũ hơn
                    os.remove(os.path.join(ckpt_dir, f))
            except:
                pass
    
    # 3. Lưu mốc mới nhất
    checkpoint_path = os.path.join(ckpt_dir, f"{prefix}_epoch_{epoch}.pt")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print(f"[HELPERS] Saved: {checkpoint_path}")

def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Not found: {ckpt_path}")
        return 0
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)
