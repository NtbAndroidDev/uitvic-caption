"""
Script sinh dữ liệu Stage 2 (FIXED VERSION):
- Load BLIP best_model.pt (Stage 1)
- Với mỗi ảnh: BLIP generate 1 câu "noisy"
- Tìm GT caption GIỐNG NHẤT với BLIP output (word overlap)
  → đảm bảo cặp (noisy, clean) cùng nói về 1 nội dung
- Lưu ra file JSONL để train ViT5

Usage (chạy trên Kaggle):
    python scripts/generate_stage2_pairs.py \
        --blip_model Salesforce/blip-image-captioning-base \
        --checkpoint /kaggle/input/YOUR_DATASET/best_model.pt \
        --train_json /kaggle/input/.../uitvic_captions_train2017.json \
        --train_images /kaggle/input/.../coco_uitvic_train \
        --val_json /kaggle/input/.../uitvic_captions_test2017.json \
        --val_images /kaggle/input/.../coco_uitvic_test \
        --output_path data/stage2_pairs_blip.jsonl \
        --batch_size 16
"""

import os
import sys
import json
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


def best_matching_gt(blip_caption: str, gt_captions: list) -> str:
    """
    Tìm GT caption có word overlap cao nhất với BLIP output.
    Dùng Jaccard similarity trên tập từ (đơn giản, nhanh).
    """
    blip_words = set(blip_caption.lower().split())
    best_score = -1
    best_cap = gt_captions[0]
    for cap in gt_captions:
        gt_words = set(cap.lower().split())
        if not blip_words and not gt_words:
            score = 1.0
        elif not blip_words or not gt_words:
            score = 0.0
        else:
            score = len(blip_words & gt_words) / len(blip_words | gt_words)
        if score > best_score:
            best_score = score
            best_cap = cap
    return best_cap


def generate_pairs(args):
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

                    # Với mỗi ảnh: tìm GT caption giống BLIP nhất
                    for noisy, gt_list in zip(noisy_captions, gt_captions_batch):
                        noisy = noisy.strip()
                        clean = best_matching_gt(noisy, gt_list).strip()
                        if noisy and clean:
                            out_f.write(json.dumps(
                                {"noisy": noisy, "clean": clean},
                                ensure_ascii=False
                            ) + "\n")
                            total_pairs += 1

                    if step % 50 == 0:
                        print(f"  [{split_name}] Step {step}/{len(loader)}, pairs so far: {total_pairs}")

    print(f"\n[DONE] Tổng cộng {total_pairs} cặp câu (BLIP → Best GT) đã lưu vào: {args.output_path}")

    # In ra 5 cặp mẫu để verify clean captions có tiếng Việt có dấu không
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
    parser = argparse.ArgumentParser(description="Generate Stage 2 training pairs (BLIP → Best-match GT)")
    parser.add_argument("--blip_model",   type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--train_json",   type=str, required=True)
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--val_json",     type=str, required=True)
    parser.add_argument("--val_images",   type=str, required=True)
    parser.add_argument("--output_path",  type=str, default="data/stage2_pairs_blip.jsonl")
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--max_batches",  type=int, default=0, help="0 = chạy full")
    args = parser.parse_args()
    generate_pairs(args)
