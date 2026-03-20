# main.py
import io
import torch
import argparse
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel

# Import từ package src
from src.models.blip_captioner import BlipViCaptioner
from src.stage2_inference import CaptionFixer
from src.utils.helpers import load_config
from src.train import train as run_train
from src.eval import evaluate as run_eval

app = FastAPI(title="Vietnamese Image Captioning Demo")

# Global variables for the API
DEVICE = None
processor = None
blip_model = None
fixer = None

def init_models():
    global DEVICE, processor, blip_model, fixer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    
    print(f"[INFO] Khởi tạo hệ thống trên thiết bị: {DEVICE}")

    try:
        # 1. Load cấu hình
        BLIP_CFG = load_config("configs/train_blip.yaml")
        STAGE2_CFG_PATH = "configs/stage2_vit5_blip.yaml"

        # 2. Khởi tạo Stage 1: BLIP
        captioner = BlipViCaptioner(BLIP_CFG["model"]["name"])
        processor = captioner.get_processor()
        blip_model = captioner.get_model().to(DEVICE)

        # Load checkpoint BLIP đã fine-tune (nếu có)
        BLIP_CKPT = "outputs/checkpoints/blip_epoch_3.pt"
        if torch.os.path.exists(BLIP_CKPT):
            blip_state = torch.load(BLIP_CKPT, map_location=DEVICE)
            blip_model.load_state_dict(blip_state["model_state_dict"])
            print(f"[INFO] Đã load Stage 1 (BLIP) thành công.")
        else:
            print(f"[WARNING] Không tìm thấy checkpoint {BLIP_CKPT}. Dùng model gốc.")

        blip_model.eval()

        # 3. Khởi tạo Stage 2: ViT5 Fixer
        STAGE2_CKPT = "outputs/stage2_checkpoints_blip/blip_epoch_2.pt"
        if torch.os.path.exists(STAGE2_CKPT):
            fixer = CaptionFixer(STAGE2_CFG_PATH, STAGE2_CKPT)
            print(f"[INFO] Đã load Stage 2 (ViT5) thành công.")
        else:
            print(f"[WARNING] Không tìm thấy checkpoint Stage 2 {STAGE2_CKPT}.")

    except Exception as e:
        print(f"[ERROR] Lỗi khi khởi tạo model: {e}")
        # Không raise lỗi ở đây để tránh sập app khi đang ở mode train

class PredictionResponse(BaseModel):
    filename: str
    noisy_caption: str
    clean_caption: str

@app.get("/")
async def index():
    return {"status": "ready", "model": "BLIP + ViT5 Two-Stage"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if blip_model is None:
         raise HTTPException(status_code=503, detail="Model chưa được khởi tạo.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên phải là hình ảnh.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_length=40,
                num_beams=5,
                early_stopping=True
            )
        noisy_caption = processor.decode(out[0], skip_special_tokens=True)

        clean_caption = noisy_caption
        if fixer:
            clean_caption = fixer.fix(noisy_caption)

        return PredictionResponse(
            filename=file.filename,
            noisy_caption=noisy_caption,
            clean_caption=clean_caption
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="serve", choices=["train", "evaluate", "serve"])
    parser.add_argument("--config", type=str, default="configs/train_blip.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/blip_epoch_3.pt", help="Path to checkpoint (for evaluate)")
    
    args = parser.parse_args()

    if args.mode == "train":
        print(f"[INFO] Bắt đầu chế độ HUẤN LUYỆN với config: {args.config}")
        run_train(args.config)
    
    elif args.mode == "evaluate":
        print(f"[INFO] Bắt đầu chế độ ĐÁNH GIÁ")
        # Gọi hàm evaluate từ src/eval.py (cần truyền đủ args)
        run_eval(
            blip_config_path=args.config,
            blip_ckpt_path=args.checkpoint,
            stage2_config_path="configs/stage2_vit5_blip.yaml",
            stage2_ckpt_path="outputs/stage2_checkpoints_blip/blip_epoch_2.pt"
        )
    
    elif args.mode == "serve":
        init_models()
        uvicorn.run(app, host="0.0.0.0", port=8000)