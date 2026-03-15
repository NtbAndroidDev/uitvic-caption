# UIT-ViC Image Captioning với BLIP

Dự án này triển khai mô hình **BLIP (Bootstrapping Language-Image Pre-training)** cho bài toán Sinh mô tả ảnh tự động (Image Captioning) bằng tiếng Việt, sử dụng tập dữ liệu **UIT-ViC (Vietnamese Image Captioning)**.

## 📌 Giới thiệu

Image Captioning là quá trình tự động tạo ra một câu mô tả tự nhiên cho một hình ảnh đầu vào. Dự án này tận dụng sức mạnh của mô hình BLIP, một trong những mô hình tiên tiến trong lĩnh vực Vision-Language, và tinh chỉnh (fine-tune) trên tập dữ liệu tiếng Việt để tạo ra các câu mô tả chính xác và tự nhiên nhất.

### 🌟 Tính năng chính
* Tinh chỉnh (Fine-tuning) mô hình BLIP trên tập dữ liệu UIT-ViC.
* Sinh câu mô tả (Inference) cho các hình ảnh mới bằng tiếng Việt.
* Đánh giá hiệu suất mô hình bằng các độ đo phổ biến: BLEU, METEOR, ROUGE-L, CIDEr.

## 🛠 Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python 3.8+ và PyTorch.

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/uitvic-blip-captioning.git
   cd uitvic-blip-captioning
   ```

2. **Tạo môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Linux/Mac
   # venv\Scripts\activate   # Trên Windows
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r ../requirements.txt
   ```

## 📂 Cấu trúc thư mục

```text
uitvic-captioning/
├── configs/            # Chứa các tệp cấu hình (YAML/JSON) cho huấn luyện và đánh giá
├── data/               # Thư mục chứa dữ liệu UIT-ViC (Không push lên Git)
├── notebooks/          # Jupyter notebooks cho EDA và thử nghiệm
├── outputs/            # Thư mục chứa model checkpoints và kết quả (Không push lên Git)
├── scripts/            # Các shell script hỗ trợ chạy các tác vụ
├── src/                # Mã nguồn chính của dự án
│   ├── dataset.py      # Xử lý dữ liệu và Dataloader
│   ├── model.py        # Định nghĩa/Khởi tạo mô hình BLIP
│   ├── train.py        # Quá trình huấn luyện mô hình
│   ├── evaluate.py     # Đánh giá mô hình
│   └── utils.py        # Các hàm tiện ích (metrics, logging,...)
├── main.py             # Script chính để chạy toàn bộ pipeline
└── README.md           # Tài liệu hướng dẫn
```

## 🚀 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Tải tập dữ liệu UIT-ViC và đặt vào thư mục `data/`. Cấu trúc thư mục dữ liệu nên như sau:
```text
data/
├── images/             # Chứa toàn bộ hình ảnh
├── train.json          # File annotations cho tập train
├── val.json            # File annotations cho tập validation
└── test.json           # File annotations cho tập test
```

### 2. Huấn luyện mô hình (Training)
Để bắt đầu quá trình huấn luyện từ đầu hoặc tinh chỉnh, chạy lệnh:
```bash
python main.py --mode train --config configs/train_config.yaml
```

### 3. Đánh giá mô hình (Evaluation)
Sau khi huấn luyện xong, chạy lệnh sau để đánh giá trên tập test:
```bash
python main.py --mode evaluate --config configs/eval_config.yaml --checkpoint outputs/best_model.pth
```

### 4. Dự đoán (Inference)
Để sinh câu mô tả cho một hình ảnh bất kỳ:
```bash
python main.py --mode inference --image_path path/to/your/image.jpg --checkpoint outputs/best_model.pth
```

## 📊 Kết quả đánh giá (Dự kiến)

| Metric  | Score |
|---------|-------|
| BLEU-4  | --    |
| METEOR  | --    |
| ROUGE-L | --    |
| CIDEr   | --    |

*(Kết quả chi tiết sẽ được cập nhật sau khi hoàn tất quá trình huấn luyện và đánh giá trên tập test của UIT-ViC).*

## 📚 Tài liệu tham khảo
* [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
* [UIT-ViIC: A Dataset for Vietnamese Image Captioning](https://arxiv.org/abs/2005.00392)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## ✍️ Tác giả
* [Tên của bạn] - Sinh viên trường Đại học Công nghệ Thông tin (UIT) - ĐHQG-HCM.
* Đồ án môn học / Khóa luận tốt nghiệp.
