"""
Script re-pair Stage 2 — RANDOM VARIANT (không cần chạy lại BLIP).

- Đọc stage2_pairs_blip.jsonl (noisy từ BLIP + clean từ Jaccard)
- Với mỗi noisy caption: tìm image tương ứng trong annotation JSON
  (match qua clean caption → image_id → tất cả 5 GT captions)
- Random chọn 1 trong 5 GT → ghi ra stage2_pairs_random.jsonl

→ Tiết kiệm ~45 phút GPU so với chạy lại toàn bộ BLIP inference.

Usage:
    python scripts/repaire_random.py \\
        --input_pairs  /kaggle/input/.../stage2_pairs_blip.jsonl \\
        --train_json   /kaggle/input/.../uitvic_captions_train2017.json \\
        --val_json     /kaggle/input/.../uitvic_captions_test2017.json \\
        --output_pairs /kaggle/working/data/stage2_pairs_random.jsonl \\
        --seed         42
"""

import json
import random
import argparse
from collections import defaultdict


def build_lookup(json_path):
    """
    Trả về 2 dict từ annotation JSON:
      caption2imgid : clean_caption → image_id
      imgid2captions: image_id → [list 5 GT captions]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    imgid2captions = defaultdict(list)
    for ann in coco["annotations"]:
        imgid2captions[ann["image_id"]].append(ann["caption"])

    caption2imgid = {}
    for ann in coco["annotations"]:
        caption2imgid[ann["caption"].strip()] = ann["image_id"]

    return caption2imgid, imgid2captions


def repaire(args):
    random.seed(args.seed)
    print(f"[INFO] Random seed: {args.seed}")

    # Build lookup từ cả train + val annotation
    print("[INFO] Building caption lookup from annotation JSONs...")
    caption2imgid, imgid2captions = {}, defaultdict(list)
    for json_path in [args.train_json, args.val_json]:
        c2i, i2c = build_lookup(json_path)
        caption2imgid.update(c2i)
        for k, v in i2c.items():
            imgid2captions[k].extend(v)

    total_images = len(imgid2captions)
    print(f"[INFO] Loaded {total_images} unique images, {len(caption2imgid)} captions")

    # Đọc Jaccard pairs và re-pair với random GT
    print(f"[INFO] Reading Jaccard pairs: {args.input_pairs}")
    with open(args.input_pairs, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"[INFO] Loaded {len(pairs)} pairs")

    matched = 0
    fallback = 0
    output_pairs = []

    for pair in pairs:
        noisy = pair["noisy"].strip()
        clean_jaccard = pair["clean"].strip()

        # Tìm image_id qua clean caption (Jaccard đã chọn)
        img_id = caption2imgid.get(clean_jaccard)

        if img_id and imgid2captions[img_id]:
            # Random chọn 1 trong tất cả GT captions của ảnh đó
            clean_random = random.choice(imgid2captions[img_id]).strip()
            matched += 1
        else:
            # Fallback: nếu không match được → dùng chính clean_jaccard
            clean_random = clean_jaccard
            fallback += 1

        if noisy and clean_random:
            output_pairs.append({"noisy": noisy, "clean": clean_random})

    print(f"[INFO] Matched: {matched}  |  Fallback (giữ Jaccard): {fallback}")

    # Ghi output
    with open(args.output_pairs, "w", encoding="utf-8") as f:
        for pair in output_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(output_pairs)} random pairs → {args.output_pairs}")

    # In 5 mẫu verify
    print("\n[SAMPLE] 5 cặp đầu:")
    for i, p in enumerate(output_pairs[:5]):
        print(f"  noisy : {p['noisy']}")
        print(f"  clean : {p['clean']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-pair Stage 2 với Random GT (không cần BLIP inference)")
    parser.add_argument("--input_pairs",  type=str, required=True, help="stage2_pairs_blip.jsonl")
    parser.add_argument("--train_json",   type=str, required=True)
    parser.add_argument("--val_json",     type=str, required=True)
    parser.add_argument("--output_pairs", type=str, default="data/stage2_pairs_random.jsonl")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    repaire(args)
