import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.blip_captioner import BlipViCaptioner
from src.utils.helpers import load_config

def main():
    parser = argparse.ArgumentParser(description="Tạo dữ liệu Stage 2 (Noisy pairs) từ BLIP-30")
    parser.add_argument("--input_json", type=str, required=True, help="Path to UITViC train JSON")
    parser.add_argument("--image_root", type=str, required=True, help="Path to COCO images")
    parser.add_argument("--blip_ckpt", type=str, required=True, help="Path to BLIP_30.pt")
    parser.add_argument("--out_jsonl", type=str, default="data/stage2_pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for faster generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREP] Device: {device}")

    # Load BLIP Model
    print(f"[PREP] Loading BLIP model and checkpoint: {args.blip_ckpt}")
    # Sử dụng config mặc định cho BLIP
    captioner = BlipViCaptioner("Salesforce/blip-image-captioning-base")
    processor = captioner.get_processor()
    model = captioner.get_model()
    
    # Load Weights
    state = torch.load(args.blip_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    # Load original COCO JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        caption = ann["caption"]
        file_name = id2file[img_id]
        samples.append({"image_id": img_id, "file_name": file_name, "caption": caption})

    print(f"[PREP] Total annotations: {len(samples)}")
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    # Batch processing for speed
    out_f = open(args.out_jsonl, "w", encoding="utf-8")
    
    for i in tqdm(range(0, len(samples), args.batch_size)):
        batch_samples = samples[i : i + args.batch_size]
        images = []
        for s in batch_samples:
            img_path = os.path.join(args.image_root, s["file_name"])
            images.append(Image.open(img_path).convert("RGB"))
            
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=40, num_beams=5)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)
            
        for s, p in zip(batch_samples, preds):
            obj = {"noisy": p.strip(), "clean": s["caption"].strip()}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"[DONE] Stage 2 pairs saved at: {args.out_jsonl}")

if __name__ == "__main__":
    main()
