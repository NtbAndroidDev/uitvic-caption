import json
from torch.utils.data import Dataset

class TextCorrectionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_source_length=64, max_target_length=64):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        noisy = item["noisy"]
        clean = item["clean"]

        source = self.tokenizer(
            noisy,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            clean,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = source["input_ids"].squeeze(0)
        attention_mask = source["attention_mask"].squeeze(0)
        labels = target["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
