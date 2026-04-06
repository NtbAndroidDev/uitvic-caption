"""
Script sinh dữ liệu Stage 2 — RANDOM VARIANT (Ablation Study):
- Giống generate_stage2_pairs.py NGOẠI TRỪ:
  - KHÔNG dùng Jaccard similarity để chọn GT caption
  - Dùng random.choice() để chọn ngẫu nhiên 1 trong 5 GT captions
- Mục đích: so sánh ablation Jaccard vs Random pairing
  → Chứng minh Jaccard matching là design choice đúng đắn

Output: stage2_pairs_random.jsonl

Usage (chạy trên Kaggle):
    python scripts/generate_stage2_pairs_random.py \\
        --blip_model Salesforce/blip-image-captioning-base \\
        --checkpoint /kaggle/input/YOUR_DATASET/best_model.pt \\
        --train_json /kaggle/input/.../uitvic_captions_train2017.json \\
        --train_images /kaggle/input/.../coco_uitvic_train \\
        --val_json /kaggle/input/.../uitvic_captions_test2017.json \\
        --val_images /kaggle/input/.../coco_uitvic_test \\
        --output_path data/stage2_pairs_random.jsonl \\
        --batch_size 16 \\
        --seed 42
"""

import os
import sys
import json
import random
import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────
# Dataset: 1 sample PER IMAGE (unique), trả về tất cả GT caption của ảnh đó
# ─────────────────────────────────────────
class UniqueImageDataset(Dataset):
    """
    Trả về MỖI ẢNH DUY NHẤT 1 lần kèm theo list tất cả GT captions của nó.
    """
    def __init__(self, json_path, image_root, processor, max_length=64):
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        # Gom tất cả GT captions theo image_id
        from collections import defaultdict
        img2captions = defaultdict(list)
        for ann in coco["annotations"]:
            img2captions[ann["image_id"]].append(ann["caption"])

        # Danh sách ảnh unique
        self.samples = []
        for img_id, captions in img2captions.items():
            self.samples.append({
                "image_id": img_id,
                "file_name": id2file[img_id],
                "captions": captions  # list 5 GT captions
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample["file_name"])
        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["captions"] = sample["captions"]  # list strings
        return encoding


def collate_fn(batch):
    """Collate: pixel_values thành tensor, captions thành list of lists."""
    import torch
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["captions"] for b in batch]  # list[list[str]]
    return {"pixel_values": pixel_values, "captions": captions}


# ─────────────────────────────────────────
# ABLATION: Random selection thay vì Jaccard
# ─────────────────────────────────────────
def random_gt(gt_captions: list) -> str:
    """
    Chọn NGẪU NHIÊN 1 GT caption từ danh sách.

    [ABLATION VARIANT] Khác với Jaccard (best_matching_gt) — không dùng
    word overlap để tìm caption gần nhất với BLIP output.
    random.seed() được set ở ngoài để đảm bảo reproducibility.
    """
    return random.choice(gt_captions)


def generate_pairs(args):
    # Fix seed để reproducibility
    random.seed(args.seed)
    print(f"[INFO] Random seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load processor
    print("[INFO] Loading BLIP processor...")
    processor = BlipProcessor.from_pretrained(args.blip_model)

    # Load BLIP architecture (config only, không download weights từ HF)
    print("[INFO] Loading BLIP architecture from config only...")
    from transformers import BlipConfig
    config = BlipConfig.from_pretrained(args.blip_model)
    model = BlipForConditionalGeneration(config)

    # Load Stage 1 checkpoint
    print(f"[INFO] Loading Stage 1 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("[INFO] Model loaded successfully!")

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)

    total_pairs = 0

    with open(args.output_path, "w", encoding="utf-8") as out_f:

        for split_name, json_path, image_root in [
            ("train", args.train_json, args.train_images),
            ("val",   args.val_json,   args.val_images),
        ]:
            print(f"\n[INFO] Processing {split_name} split: {json_path}")

            dataset = UniqueImageDataset(json_path, image_root, processor, max_length=64)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            print(f"[INFO] {split_name}: {len(dataset)} unique images, {len(loader)} batches")

            with torch.no_grad():
                for step, batch in enumerate(loader, start=1):
                    if args.max_batches > 0 and step > args.max_batches:
                        print(f"  [{split_name}] Reached max_batches={args.max_batches}, stopping early.")
                        break

                    pixel_values = batch["pixel_values"].to(device)
                    gt_captions_batch = batch["captions"]  # list[list[str]]

                    # BLIP generate câu "noisy"
                    generated_ids = model.generate(
                        pixel_values=pixel_values,
                        max_length=64,
                        num_beams=4
                    )
                    noisy_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    # [ABLATION] Với mỗi ảnh: RANDOM chọn 1 GT caption (không dùng Jaccard)
                    for noisy, gt_list in zip(noisy_captions, gt_captions_batch):
                        noisy = noisy.strip()
                        clean = random_gt(gt_list).strip()  # ← KEY DIFFERENCE vs Jaccard
                        if noisy and clean:
                            out_f.write(json.dumps(
                                {"noisy": noisy, "clean": clean},
                                ensure_ascii=False
                            ) + "\n")
                            total_pairs += 1

                    if step % 50 == 0:
                        print(f"  [{split_name}] Step {step}/{len(loader)}, pairs so far: {total_pairs}")

    print(f"\n[DONE] Tổng cộng {total_pairs} cặp câu (BLIP → Random GT) đã lưu vào: {args.output_path}")
    print("[NOTE] Ablation variant — dùng random.choice() thay Jaccard similarity")

    # In ra 5 cặp mẫu để verify
    print("\n[SAMPLE] 5 cặp đầu tiên:")
    import json as _json
    with open(args.output_path, encoding="utf-8") as _f:
        for i, line in enumerate(_f):
            if i >= 5:
                break
            obj = _json.loads(line)
            print(f"  noisy: {obj['noisy']}")
            print(f"  clean: {obj['clean']}")
            print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 training pairs — RANDOM VARIANT (Ablation Study)"
    )
    parser.add_argument("--blip_model",   type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--train_json",   type=str, required=True)
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--val_json",     type=str, required=True)
    parser.add_argument("--val_images",   type=str, required=True)
    parser.add_argument("--output_path",  type=str, default="data/stage2_pairs_random.jsonl")
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--max_batches",  type=int, default=0, help="0 = chạy full")
    parser.add_argument("--seed",         type=int, default=42, help="Random seed để reproducibility")
    args = parser.parse_args()
    generate_pairs(args)
