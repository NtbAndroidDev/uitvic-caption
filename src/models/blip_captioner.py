# src/models/blip_captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipViCaptioner:
    def __init__(self, model_name: str):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def get_processor(self):
        return self.processor

    def get_model(self):
        return self.model
