# UIT-ViC Image Captioning with BLIP

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/leo040802/uitvic-dataset)

This project implements the **BLIP (Bootstrapping Language-Image Pre-training)** model for the Image Captioning task in Vietnamese, utilizing the **UIT-ViIC (Vietnamese Image Captioning)** dataset.

## 📌 Introduction

Image Captioning is the process of automatically generating a natural language description for an input image. This project leverages the power of the BLIP model, one of the state-of-the-art models in the Vision-Language domain, and fine-tunes it on a Vietnamese dataset to generate the most accurate and natural captions.

### 🌟 Key Features
* Fine-tuning the BLIP model on the UIT-ViIC dataset.
* Generating captions (Inference) for new images in Vietnamese.
* Evaluating model performance using common metrics: BLEU, METEOR, ROUGE-L, CIDEr.

## 🛠 Environment Setup

Ensure you have Python 3.8+ and PyTorch installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NtbAndroidDev/uitvic-caption.git
   cd uitvic-caption
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # venv\Scripts\activate   # On Windows
   ```

3. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Directory Structure

```text
uitvic-captioning/
├── configs/            # Configuration files (YAML/JSON) for training and evaluation
├── data/               # Directory containing the UIT-ViIC dataset (Not pushed to Git)
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── outputs/            # Directory containing model checkpoints and results (Not pushed to Git)
├── scripts/            # Shell scripts to support running tasks
├── src/                # Main source code of the project
│   ├── dataset.py      # Data processing and Dataloader
│   ├── model.py        # BLIP model definition/initialization
│   ├── train.py        # Model training process
│   ├── evaluate.py     # Model evaluation
│   └── utils.py        # Utility functions (metrics, logging, etc.)
├── main.py             # Main script to run the entire pipeline
└── README.md           # Documentation
```

## 🚀 Usage Guide

### 1. Data Preparation
Download the UIT-ViIC dataset on [Kaggle](https://www.kaggle.com/datasets/leo040802/uitvic-dataset) and place it in the `data/` directory. The data directory structure should be as follows:
```text
data/
├── images/             # Contains all images
├── train.json          # Annotations file for the training set
├── val.json            # Annotations file for the validation set
└── test.json           # Annotations file for the test set
```

### 2. Model Training
To start the training process from scratch or fine-tune, run the following command:
```bash
python main.py --mode train --config configs/train_config.yaml
```

### 3. Model Evaluation
After training is complete, run the following command to evaluate on the test set:
```bash
python main.py --mode evaluate --config configs/eval_config.yaml --checkpoint outputs/best_model.pth
```

### 4. Inference
To generate a caption for any image:
```bash
python main.py --mode inference --image_path path/to/your/image.jpg --checkpoint outputs/best_model.pth
```

## 📊 Evaluation Results (Expected)

| Metric  | Score |
|---------|-------|
| BLEU-4  | --    |
| METEOR  | --    |
| ROUGE-L | --    |
| CIDEr   | --    |

*(Detailed results will be updated after completing the training and evaluation process on the UIT-ViIC test set).*

## 📚 References
* [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
* [UIT-ViIC: A Dataset for Vietnamese Image Captioning (Paper)](https://arxiv.org/abs/2002.00175?utm_source=chatgpt.com)
* [UIT-ViIC Dataset on Kaggle](https://www.kaggle.com/datasets/leo040802/uitvic-dataset)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## ✍️ Author
* [Nguyen Thanh Binh](https://github.com/NtbAndroidDev) - Student at University of Information Technology (UIT) - VNU-HCM.
* Course Project / Graduation Thesis.