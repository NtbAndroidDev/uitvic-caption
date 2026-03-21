# src/models/blip_captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class BlipViCaptioner:
    def __init__(self, model_name: str):
        self.processor = BlipProcessor.from_pretrained(model_name)
        # Ép máy nạp thẳng vào bộ nhớ, bỏ qua cơ chế materialization gây treo
        print(f"[BLIP] Loading base model weights from {model_name}...")
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False, # Tắt tính năng gây treo
            torch_dtype=torch.float32
        )

    def get_processor(self):
        return self.processor

    def get_model(self):
        return self.model
