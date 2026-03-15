# scripts/build_stage2_from_blip.py
import sys
import os
import json
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.blip_captioner import BlipViCaptioner
from src.utils.helpers import load_config


INPUT_JSON = "data/uitvic_dataset/uitvic_captions_train_new.json"
IMAGE_ROOT = "data/uitvic_dataset/coco_uitvic_train/coco_uitvic_train"
BLIP_CONFIG = "configs/train_blip.yaml"
BLIP_CKPT = "outputs/checkpoints/blip_epoch_3.pt"  # dùng epoch mới nhất
OUT_JSONL = "data/stage2_pairs_blip.jsonl"


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_coco(json_path):
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


def main():
    device = choose_device()
    print("Device:", device)

    # load config + BLIP
    cfg = load_config(BLIP_CONFIG)
    captioner = BlipViCaptioner(cfg["model"]["name"])
    processor = captioner.get_processor()
    model = captioner.get_model()

    if not os.path.isfile(BLIP_CKPT):
        raise FileNotFoundError(f"Không tìm thấy BLIP ckpt: {BLIP_CKPT}")
    print(f"[BLIP] Loading checkpoint from {BLIP_CKPT}")
    state = torch.load(BLIP_CKPT, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # load COCO train
    samples = load_coco(INPUT_JSON)
    print("Tổng số annotation:", len(samples))

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    out_f = open(OUT_JSONL, "w", encoding="utf-8")

    for idx, s in enumerate(samples, start=1):
        img_path = os.path.join(IMAGE_ROOT, s["file_name"])
        gt_caption = s["caption"]
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=40,
                num_beams=5,
                early_stopping=True,
            )
        pred = processor.decode(out[0], skip_special_tokens=True)

        obj = {"noisy": pred, "clean": gt_caption}
        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if idx % 100 == 0:
            print(f"[PROGRESS] {idx}/{len(samples)}")

    print("Tổng số cặp đã tạo:", len(samples))
    out_f.close()
    print("Đã lưu:", OUT_JSONL)


if __name__ == "__main__":
    main()
