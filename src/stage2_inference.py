# src/stage2_inference.py
import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from .utils.helpers import load_config


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class CaptionFixer:
    def __init__(self, config_path: str, ckpt_path: str):
        cfg = load_config(config_path)
        self.device = choose_device()
        print("[Stage2] Using device:", self.device)

        model_name = cfg["model"]["name"]
        print(f"[Stage2] Loading tokenizer & model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        print(f"[Stage2] Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        # dùng đúng key khi save_checkpoint
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        self.max_source_length = cfg["data"]["max_source_length"]
        self.max_target_length = cfg["data"]["max_target_length"]

    def fix(self, noisy_caption: str) -> str:
        """Sửa câu noisy -> câu tiếng Việt sạch hơn."""
        inputs = self.tokenizer(
            noisy_caption,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=self.max_target_length,
                num_beams=5,
                early_stopping=True,
            )

        clean = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return clean
