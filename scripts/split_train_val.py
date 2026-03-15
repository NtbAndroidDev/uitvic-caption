import json
import random
import os

random.seed(42)

TRAIN_JSON = "data/uitvic_dataset/uitvic_captions_train2017.json"
OUT_TRAIN_JSON = "data/uitvic_dataset/uitvic_captions_train_new.json"
OUT_VAL_JSON   = "data/uitvic_dataset/uitvic_captions_val2017.json"

VAL_RATIO = 0.1  # 10% ảnh làm val

def main():
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * VAL_RATIO)
    val_ids = set(image_ids[:num_val])
    train_ids = set(image_ids[num_val:])

    train_images = [img for img in images if img["id"] in train_ids]
    val_images   = [img for img in images if img["id"] in val_ids]

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
    val_annotations   = [ann for ann in annotations if ann["image_id"] in val_ids]

    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images  : {len(val_images)}")

    base_info = {k: v for k, v in coco.items() if k not in ["images", "annotations"]}

    train_coco = {
        **base_info,
        "images": train_images,
        "annotations": train_annotations,
    }
    val_coco = {
        **base_info,
        "images": val_images,
        "annotations": val_annotations,
    }

    os.makedirs(os.path.dirname(OUT_TRAIN_JSON), exist_ok=True)
    with open(OUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)

    with open(OUT_VAL_JSON, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"Saved new train to {OUT_TRAIN_JSON}")
    print(f"Saved val to {OUT_VAL_JSON}")


if __name__ == "__main__":
    main()
