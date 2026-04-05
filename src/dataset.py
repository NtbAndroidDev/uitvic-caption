import json
import os
from typing import Any, Dict, List
from PIL import Image
from torch.utils.data import Dataset


class UITViCCOCODataset(Dataset):
    def __init__(self, json_path: str, image_root: str, processor, max_length: int = 40):
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # map id ảnh -> file_name
        self.id2file: Dict[int, str] = {
            img["id"]: img["file_name"] for img in coco["images"]
        }

        self.samples: List[Dict[str, Any]] = []
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"]             # tiếng Việt có dấu
            file_name = self.id2file[img_id]
            self.samples.append(
                {"image_id": img_id, "file_name": file_name, "caption": caption}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_root, sample["file_name"])
        caption = sample["caption"]

        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        encoding["raw_caption"] = sample["caption"]  # raw string có dấu từ JSON
        return encoding
