# main.py
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from pydantic import BaseModel

# Import từ package src dựa trên cấu trúc cây thư mục của bạn
from src.models.blip_captioner import BlipViCaptioner
from src.stage2_inference import CaptionFixer
from src.utils.helpers import load_config

app = FastAPI(title="Vietnamese Image Captioning Demo")

# --- KHỞI TẠO HỆ THỐNG ---
# Tự động chọn thiết bị (CUDA, MPS cho Mac, hoặc CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"[INFO] Khởi động hệ thống trên thiết bị: {DEVICE}")

try:
    # 1. Load cấu hình BLIP và ViT5
    BLIP_CFG = load_config("configs/train_blip.yaml")
    STAGE2_CFG_PATH = "configs/stage2_vit5_blip.yaml"

    # 2. Khởi tạo Stage 1: BLIP
    captioner = BlipViCaptioner(BLIP_CFG["model"]["name"])
    processor = captioner.get_processor()
    blip_model = captioner.get_model().to(DEVICE)

    # Load checkpoint BLIP đã fine-tune (epoch 3)
    BLIP_CKPT = "outputs/checkpoints/blip_epoch_3.pt"
    blip_state = torch.load(BLIP_CKPT, map_location=DEVICE)
    blip_model.load_state_dict(blip_state["model_state_dict"])
    blip_model.eval()
    print(f"[INFO] Đã load Stage 1 (BLIP) thành công.")

    # 3. Khởi tạo Stage 2: ViT5 Fixer
    # Sử dụng checkpoint đã train với dữ liệu nhiễu thực tế
    STAGE2_CKPT = "outputs/stage2_checkpoints_blip/blip_epoch_2.pt"
    fixer = CaptionFixer(STAGE2_CFG_PATH, STAGE2_CKPT)
    print(f"[INFO] Đã load Stage 2 (ViT5) thành công.")

except Exception as e:
    print(f"[ERROR] Lỗi khi khởi tạo model: {e}")
    raise e


class PredictionResponse(BaseModel):
    filename: str
    noisy_caption: str
    clean_caption: str


@app.get("/")
async def index():
    return {"status": "ready", "model": "BLIP + ViT5 Two-Stage"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Kiểm tra định dạng file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên phải là hình ảnh.")

    try:
        # 1. Đọc và tiền xử lý ảnh
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Giai đoạn 1: Sinh mô tả thô (Noisy Caption) từ BLIP
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_length=40,
                num_beams=5,
                early_stopping=True
            )
        noisy_caption = processor.decode(out[0], skip_special_tokens=True)

        # 3. Giai đoạn 2: Tinh chỉnh bằng ViT5 (Clean Caption)
        clean_caption = fixer.fix(noisy_caption)

        return PredictionResponse(
            filename=file.filename,
            noisy_caption=noisy_caption,
            clean_caption=clean_caption
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

# Cách chạy: uvicorn main:app --reload