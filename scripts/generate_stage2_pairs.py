"""
Script sinh dữ liệu Stage 2:
- Load BLIP best_model.pt (Stage 1)
- Chạy inference trên toàn bộ Train + Val Dataset
- Lưu ra file JSONL format: {"noisy": "<câu BLIP thô>", "clean": "<câu gốc tiếng Việt>"}
- File này sẽ được dùng để train ViT5 (Stage 2)

Usage (chạy trên Kaggle):
    python scripts/generate_stage2_pairs.py \
        --blip_model Salesforce/blip-image-captioning-base \
        --checkpoint /kaggle/input/YOUR_DATASET/best_model.pt \
        --train_json /kaggle/input/.../uitvic_captions_train2017.json \
        --train_images /kaggle/input/.../coco_uitvic_train/coco_uitvic_train \
        --val_json /kaggle/input/.../uitvic_captions_test2017.json \
        --val_images /kaggle/input/.../coco_uitvic_test/coco_uitvic_test \
        --output_path data/stage2_pairs_blip.jsonl \
        --batch_size 16
"""

import os
import sys
import json
import argparse
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import UITViCCOCODataset


def generate_pairs(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load processor + model
    print(f"[INFO] Loading BLIP model from: {args.blip_model}")
    processor = BlipProcessor.from_pretrained(args.blip_model)
    model = BlipForConditionalGeneration.from_pretrained(args.blip_model)

    # Load Stage 1 checkpoint (best_model.pt)
    print(f"[INFO] Loading Stage 1 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Hỗ trợ cả 2 format: state_dict thuần hoặc dict bọc ngoài
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully!")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    total_pairs = 0

    with open(args.output_path, "w", encoding="utf-8") as out_f:

        for split_name, json_path, image_root in [
            ("train", args.train_json, args.train_images),
            ("val",   args.val_json,   args.val_images),
        ]:
            print(f"\n[INFO] Processing {split_name} split: {json_path}")

            dataset = UITViCCOCODataset(
                json_path, image_root, processor, max_length=64
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2
            )

            print(f"[INFO] {split_name}: {len(dataset)} samples, {len(loader)} batches")

            with torch.no_grad():
                for step, batch in enumerate(loader, start=1):
                    pixel_values = batch["pixel_values"].to(device)

                    # BLIP generate ra câu "thô" (không dấu / broken Vietnamese)
                    generated_ids = model.generate(
                        pixel_values=pixel_values,
                        max_length=64,
                        num_beams=4
                    )
                    noisy_captions = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )

                    # Ground truth: câu "clean" tiếng Việt chuẩn
                    labels = batch["labels"].clone()
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    clean_captions = processor.batch_decode(
                        labels, skip_special_tokens=True
                    )

                    # Ghi ra file JSONL
                    for noisy, clean in zip(noisy_captions, clean_captions):
                        noisy = noisy.strip()
                        clean = clean.strip()
                        if noisy and clean:
                            out_f.write(json.dumps(
                                {"noisy": noisy, "clean": clean},
                                ensure_ascii=False
                            ) + "\n")
                            total_pairs += 1

                    if step % 50 == 0:
                        print(f"  [{split_name}] Step {step}/{len(loader)}, pairs so far: {total_pairs}")

    print(f"\n[DONE] Tổng cộng {total_pairs} cặp câu đã được lưu vào: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 2 training pairs")
    parser.add_argument("--blip_model",  type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--checkpoint",  type=str, required=True, help="Path to best_model.pt from Stage 1")
    parser.add_argument("--train_json",  type=str, required=True)
    parser.add_argument("--train_images",type=str, required=True)
    parser.add_argument("--val_json",    type=str, required=True)
    parser.add_argument("--val_images",  type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/stage2_pairs_blip.jsonl")
    parser.add_argument("--batch_size",  type=int, default=16)
    args = parser.parse_args()
    generate_pairs(args)
