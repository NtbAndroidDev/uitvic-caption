#!/usr/bin/env python3
"""
prepare_qwen2vl_dataset.py
Convert UIT-ViIC (COCO format) → Qwen2-VL conversation format.
Output: data/qwen2vl_train.json, data/qwen2vl_val.json, data/qwen2vl_test.json
"""
import json, os, random, argparse
from pathlib import Path


INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."


def build_dataset(ann_file: str, image_dir: str, seed: int = 42, val_split: float = 0.1):
    """Load COCO-style annotation, return list of conversation dicts."""
    random.seed(seed)
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    # image_id → file_name
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # image_id → list[caption]
    id2caps = {}
    for ann in coco["annotations"]:
        id2caps.setdefault(ann["image_id"], []).append(ann["caption"])

    records = []
    for image_id, captions in id2caps.items():
        fname = id2file.get(image_id)
        if not fname:
            continue
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue  # skip missing images

        # Pick one GT caption randomly per image
        caption = random.choice(captions).strip()

        records.append({
            "image": img_path,
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text",  "text": INSTRUCTION},
                    ],
                },
                {
                    "role": "assistant",
                    "content": caption,
                },
            ],
            # Keep GT list for BLEU eval
            "_gt_captions": captions,
            "_image_id": image_id,
        })

    return records


def split_train_val(records, val_split=0.1, seed=42):
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json",   required=True)
    parser.add_argument("--test_json",    required=True)
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--test_images",  required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--val_split",    type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build train + val split
    print("[PREP] Building train dataset...")
    all_train = build_dataset(args.train_json, args.train_images, seed=args.seed)
    train, val = split_train_val(all_train, val_split=args.val_split, seed=args.seed)
    print(f"  Train: {len(train)} | Val: {len(val)}")

    # Build test (keep all GTs for BLEU)
    print("[PREP] Building test dataset...")
    test = build_dataset(args.test_json, args.test_images, seed=args.seed)
    print(f"  Test: {len(test)}")

    # Save
    for name, data in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(args.output_dir, f"qwen2vl_{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[PREP] Saved: {out} ({len(data)} records)")

    # Sanity check — print first example
    ex = train[0]
    print(f"\n[SANITY] First train example:")
    print(f"  Image : {ex['image']}")
    print(f"  Caption: {ex['conversations'][1]['content']}")
    print(f"  GTs   : {ex['_gt_captions'][:2]}")
    print("\n✓ Dataset preparation done!")


if __name__ == "__main__":
    main()
