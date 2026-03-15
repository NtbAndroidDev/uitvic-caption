import json
from unidecode import unidecode
import random
import re
import os

INPUT_JSON = "data/uitvic_dataset/uitvic_captions_train_new.json"
OUT_JSONL = "data/stage2_pairs.jsonl"

random.seed(42)

def random_noise(text: str) -> str:
    # 1) bỏ dấu toàn bộ
    base = unidecode(text)

    # 2) đôi khi viết thường hết
    base = base.strip()
    if random.random() < 0.5:
        base = base.lower()

    # 3) bỏ dấu câu .,!? cho giống kiểu BLIP hay lười
    base = re.sub(r"[.,!?]", "", base)

    return base

def main():
    if not os.path.isfile(INPUT_JSON):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    pairs = []
    for ann in coco["annotations"]:
        clean = ann["caption"].strip()
        noisy = random_noise(clean)
        pairs.append({"noisy": noisy, "clean": clean})

    print("Tổng số cặp:", len(pairs))

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("Đã lưu:", OUT_JSONL)

if __name__ == "__main__":
    main()
