import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
# Tắt Tqdm để dùng Print truyền thống cho Kaggle hông bị treo log
# from tqdm import tqdm

from src.models.blip_captioner import BlipViCaptioner
from src.utils.helpers import load_config

def main():
    parser = argparse.ArgumentParser(description="Tạo dữ liệu Stage 2 (Noisy pairs) từ BLIP-30")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--blip_ckpt", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default="data/stage2_pairs.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREP] Device: {device}")

    # Load BLIP Model
    print(f"[PREP] Loading base model and ckpt: {args.blip_ckpt}")
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

    print(f"[PREP] Tổng cộng {len(samples)} mẫu. Đang bắt đầu trích xuất...")
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    out_f = open(args.out_jsonl, "w", encoding="utf-8")
    
    total = len(samples)
    for i in range(0, total, args.batch_size):
        batch_samples = samples[i : i + args.batch_size]
        images = []
        for s in batch_samples:
            img_path = os.path.join(args.image_root, s["file_name"])
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except:
                continue # Bỏ qua ảnh lỗi
            
        if not images: continue
        
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=40, num_beams=5)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)
            
        for s, p in zip(batch_samples, preds):
            obj = {"noisy": p.strip(), "clean": s["caption"].strip()}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # IN LOG TRUYỀN THỐNG ĐỂ ANH THẤY NÓ CHẠY ✅✅✅
        if (i + args.batch_size) % 128 == 0 or (i + args.batch_size) >= total:
            print(f"[PROGRESS] Đã xong {min(i + args.batch_size, total)}/{total} ảnh ({(min(i+args.batch_size, total)/total)*100:.2f}%)")

    out_f.close()
    print(f"[DONE] File huấn luyện đã sẵn sàng: {args.out_jsonl}")

if __name__ == "__main__":
    main()
