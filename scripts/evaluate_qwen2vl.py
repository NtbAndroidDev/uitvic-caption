#!/usr/bin/env python3
"""
evaluate_qwen2vl.py
Đánh giá Qwen2-VL-7B trên full test set UIT-ViIC.
Output: pipeline_evaluation.csv (cùng format baseline để dễ so sánh)
Usage:
  python scripts/evaluate_qwen2vl.py \
    --model_dir /kaggle/working/outputs/qwen2vl_checkpoints/best_model \
    --test_json  /kaggle/working/data/qwen2vl_test.json \
    --output_dir /kaggle/working/outputs/eval_qwen2vl
"""
import os, json, csv, argparse
import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

try:
    from unsloth import FastVisionModel
    UNSLOTH = True
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    UNSLOTH = False

INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."


def load_model(model_dir: str, device: str):
    if UNSLOTH:
        model, tokenizer = FastVisionModel.from_pretrained(model_dir, load_in_4bit=True)
        FastVisionModel.for_inference(model)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16
        )
    return model.to(device).eval(), tokenizer


def generate_caption(model, tokenizer, image: Image.Image, device: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": INSTRUCTION},
        ],
    }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text, images=[image], return_tensors="pt",
        padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"[EVAL] Loading model from {args.model_dir}...")
    model, tokenizer = load_model(args.model_dir, device)
    print("[EVAL] Model loaded ✓")

    # Load test data
    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    print(f"[EVAL] Test samples: {len(test_data)}")

    hypotheses, references = [], []
    qualitative = []

    for i, rec in enumerate(test_data):
        image = Image.open(rec["image"]).convert("RGB")
        gt_captions = rec.get("_gt_captions", [rec["conversations"][1]["content"]])

        pred = generate_caption(model, tokenizer, image, device)
        if not pred:
            pred = gt_captions[0]  # fallback

        hypotheses.append(pred.split())
        references.append([gt.split() for gt in gt_captions])

        if i < 20:  # qualitative examples
            qualitative.append({
                "image": rec["image"],
                "ground_truth": gt_captions[0],
                "qwen2vl_output": pred,
            })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test_data)}] done...")

    # BLEU
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    print(f"\n[RESULT] BLEU-1={b1:.4f} | BLEU-2={b2:.4f} | BLEU-3={b3:.4f} | BLEU-4={b4:.4f}")
    print(f"[BASELINE] BLIP+ViT5 (Jaccard): BLEU-4=0.1951")
    if b4 > 0.1951:
        print(f"[RESULT] ✅ Qwen2-VL-7B vượt baseline! Δ=+{b4-0.1951:.4f}")
    else:
        print(f"[RESULT] ⚠️  Qwen2-VL-7B chưa vượt baseline. Δ={b4-0.1951:.4f}")

    # Save CSV — cùng format evaluate_pipeline.py
    csv_path = os.path.join(args.output_dir, "pipeline_evaluation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "BLIP_ViT5_Jaccard", "Qwen2VL_7B", "Delta"])
        for metric, baseline, score in [
            ("BLEU1", 0.4552, b1), ("BLEU2", 0.3196, b2),
            ("BLEU3", 0.2493, b3), ("BLEU4", 0.1951, b4),
        ]:
            w.writerow([metric, f"{baseline:.4f}", f"{score:.4f}", f"{score-baseline:+.4f}"])
    print(f"[SAVE] {csv_path}")

    # Save qualitative
    qual_path = os.path.join(args.output_dir, "qualitative_examples.json")
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump(qualitative, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {qual_path}")

    print("\n✓ Evaluation done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   required=True)
    parser.add_argument("--test_json",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
