# src/inference.py
import os
import torch
from PIL import Image
import argparse

from .models.blip_captioner import BlipViCaptioner
from .utils.helpers import load_config
from .stage2_inference import CaptionFixer


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_blip_from_checkpoint(config_path: str, ckpt_path: str, device: torch.device):
    cfg = load_config(config_path)

    captioner = BlipViCaptioner(cfg["model"]["name"])
    processor = captioner.get_processor()
    model = captioner.get_model()

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"[BLIP] Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("[BLIP] Checkpoint not found, dùng pretrained BLIP (chưa fine-tune).")

    model = model.to(device)
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_blip.yaml")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn tới file ảnh .jpg cần caption",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/checkpoints/blip_epoch_1.pt",
        help="Checkpoint BLIP đã fine-tune",
    )

    # stage 2 options
    parser.add_argument(
        "--stage2_config",
        type=str,
        default=None,
        help="Config cho Stage2 (ViT5), vd: configs/stage2_vit5.yaml",
    )
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default=None,
        help="Checkpoint Stage2, vd: outputs/stage2_checkpoints/blip_epoch_3.pt",
    )

    args = parser.parse_args()

    device = choose_device()
    print("[BLIP] Using device for inference:", device)

    # load BLIP
    model, processor = load_blip_from_checkpoint(args.config, args.ckpt, device)

    # load ảnh
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {args.image}")

    image = Image.open(args.image).convert("RGB")

    # chuẩn bị input cho BLIP
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Stage 1: BLIP generate
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=40,
            num_beams=5,
            early_stopping=True,
        )

    noisy_caption = processor.decode(out[0], skip_special_tokens=True)
    print("\n=== BLIP raw caption ===")
    print(noisy_caption)

    # Stage 2: sửa câu nếu có config + ckpt
    if args.stage2_config is not None and args.stage2_ckpt is not None:
        fixer = CaptionFixer(args.stage2_config, args.stage2_ckpt)
        clean_caption = fixer.fix(noisy_caption)
        print("\n=== Stage2 fixed caption ===")
        print(clean_caption)
    else:
        print("\n[INFO] Không dùng Stage2. Nếu muốn, truyền thêm --stage2_config và --stage2_ckpt")


if __name__ == "__main__":
    main()
