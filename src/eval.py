# src/eval.py
import os
import json
import random

import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unidecode import unidecode

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


def load_coco(json_path):
    """Đọc COCO json và trả về list sample {file_name, caption}."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        caption = ann["caption"]
        file_name = id2file[img_id]
        samples.append({"image_id": img_id, "file_name": file_name, "caption": caption})
    return samples


def evaluate(
    blip_config_path: str,
    blip_ckpt_path: str,
    stage2_config_path: str,
    stage2_ckpt_path: str,
    max_samples: int = 200,
):
    print("[INFO] Bắt đầu evaluate BLIP & BLIP+Stage2...")
    cfg = load_config(blip_config_path)
    device = choose_device()
    print("[INFO] Eval device:", device)

    # ====== Load BLIP ======
    print("[BLIP] Load model & processor...")
    captioner = BlipViCaptioner(cfg["model"]["name"])
    processor = captioner.get_processor()
    blip_model = captioner.get_model()

    if not os.path.isfile(blip_ckpt_path):
        raise FileNotFoundError(f"[BLIP] Không tìm thấy checkpoint: {blip_ckpt_path}")
    print(f"[BLIP] Loading checkpoint from {blip_ckpt_path}")
    blip_state = torch.load(blip_ckpt_path, map_location=device)
    blip_model.load_state_dict(blip_state["model_state_dict"])
    blip_model.to(device)
    blip_model.eval()

    # ====== Load Stage2 fixer ======
    if stage2_config_path is None or stage2_ckpt_path is None:
        print("[Stage2] Không có config/ckpt, sẽ chỉ eval BLIP raw.")
        fixer = None
    else:
        print("[Stage2] Khởi tạo CaptionFixer...")
        fixer = CaptionFixer(stage2_config_path, stage2_ckpt_path)

    # ====== Load data (val set) ======
    val_json = cfg["data"]["val_json"]
    image_root = cfg["data"]["val_image_root"]
    print(f"[DATA] Đọc dữ liệu từ: {val_json}")
    samples = load_coco(val_json)
    print(f"[DATA] Tổng số mẫu trong val: {len(samples)}")

    random.shuffle(samples)
    samples = samples[:max_samples]
    print(f"[DATA] Đánh giá trên {len(samples)} mẫu")

    smooth = SmoothingFunction().method1

    bleu_raw = []
    bleu_raw_no = []
    bleu_fix = []
    bleu_fix_no = []

    # Lưu vài ví dụ để in ra cuối
    example_cases = []

    for idx, s in enumerate(samples, start=1):
        img_path = os.path.join(image_root, s["file_name"])
        gt = s["caption"]

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # ----- BLIP generate -----
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_length=40,
                num_beams=5,
                early_stopping=True,
            )
        pred_raw = processor.decode(out[0], skip_special_tokens=True)

        # ----- Stage2 sửa câu nếu có -----
        if fixer is not None:
            pred_fix = fixer.fix(pred_raw)
        else:
            pred_fix = pred_raw

        # ----- BLEU RAW -----
        bleu_r = sentence_bleu(
            [gt.split()],
            pred_raw.split(),
            smoothing_function=smooth,
        )
        bleu_raw.append(bleu_r)

        bleu_r_no = sentence_bleu(
            [unidecode(gt).split()],
            unidecode(pred_raw).split(),
            smoothing_function=smooth,
        )
        bleu_raw_no.append(bleu_r_no)

        # ----- BLEU FIXED -----
        bleu_f = sentence_bleu(
            [gt.split()],
            pred_fix.split(),
            smoothing_function=smooth,
        )
        bleu_fix.append(bleu_f)

        bleu_f_no = sentence_bleu(
            [unidecode(gt).split()],
            unidecode(pred_fix).split(),
            smoothing_function=smooth,
        )
        bleu_fix_no.append(bleu_f_no)

        # Lưu vài ví dụ (5 mẫu đầu)
        if len(example_cases) < 5:
            example_cases.append(
                {
                    "file_name": s["file_name"],
                    "gt": gt,
                    "raw": pred_raw,
                    "fix": pred_fix,
                }
            )

        if idx % 20 == 0:
            print(
                f"[PROGRESS] {idx}/{len(samples)} mẫu - "
                f"BLEU_raw: {sum(bleu_raw)/len(bleu_raw):.4f}, "
                f"BLEU_fix: {sum(bleu_fix)/len(bleu_fix):.4f}"
            )

    # ====== Kết quả cuối cùng ======
    bleu_raw_avg = sum(bleu_raw) / len(bleu_raw)
    bleu_raw_no_avg = sum(bleu_raw_no) / len(bleu_raw_no)
    bleu_fix_avg = sum(bleu_fix) / len(bleu_fix)
    bleu_fix_no_avg = sum(bleu_fix_no) / len(bleu_fix_no)

    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print(f"BLEU BLIP raw (giữ dấu)        : {bleu_raw_avg:.4f}")
    print(f"BLEU BLIP raw (bỏ dấu so sánh) : {bleu_raw_no_avg:.4f}")
    print(f"BLEU BLIP+Stage2 (giữ dấu)     : {bleu_fix_avg:.4f}")
    print(f"BLEU BLIP+Stage2 (bỏ dấu)      : {bleu_fix_no_avg:.4f}")

    print("\n===== VÍ DỤ MINH HỌA (5 mẫu) =====")
    for ex in example_cases:
        print(f"\n--- {ex['file_name']} ---")
        print(f"GT      : {ex['gt']}")
        print(f"BLIP    : {ex['raw']}")
        print(f"Stage2  : {ex['fix']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_blip.yaml",
        help="Config BLIP",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint BLIP đã fine-tune",
    )
    parser.add_argument(
        "--stage2_config",
        type=str,
        default="configs/stage2_vit5.yaml",
        help="Config Stage2 (ViT5)",
    )
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        required=True,
        help="Checkpoint Stage2 đã train",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Số mẫu lấy trong val để eval",
    )
    args = parser.parse_args()

    evaluate(
        args.config,
        args.ckpt,
        args.stage2_config,
        args.stage2_ckpt,
        args.max_samples,
    )
