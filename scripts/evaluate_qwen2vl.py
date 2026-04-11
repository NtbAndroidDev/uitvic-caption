#!/usr/bin/env python3
"""
evaluate_qwen2vl.py
Đánh giá Qwen2-VL-7B trên full test set UIT-ViIC.
Output: pipeline_evaluation.csv + qualitative_examples.json
Usage:
  python scripts/evaluate_qwen2vl.py \
    --model_dir /kaggle/working/outputs/qwen2vl_checkpoints/best_model \
    --test_json  /kaggle/working/data/qwen2vl_test.json \
    --output_dir /kaggle/working/outputs/eval_qwen2vl
"""
import os, json, csv, argparse, sys
import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

try:
    from unsloth import FastVisionModel
    UNSLOTH = True
except ImportError:
    UNSLOTH = False

INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."


def load_model(model_dir: str):
    """Load best_model (LoRA adapter) for inference."""
    assert os.path.exists(model_dir), f"❌ model_dir not found: {model_dir}"
    print(f"[EVAL] Loading model from {model_dir}...", flush=True)

    if UNSLOTH:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_dir, load_in_4bit=True
        )
        FastVisionModel.for_inference(model)
        # 4bit models are already on GPU — do NOT call .to(device)
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16,
            device_map="auto"
        )
    model.eval()
    print("[EVAL] Model loaded ✓", flush=True)
    return model, tokenizer


def generate_caption(model, tokenizer, image: Image.Image) -> str:
    """Generate a single caption for one image."""
    device = next(model.parameters()).device
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": INSTRUCTION},
        ],
    }]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    print(f"[EVAL] Device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model_dir)

    # Load test data
    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    print(f"[EVAL] Test samples: {len(test_data)}", flush=True)

    hypotheses, references = [], []
    qualitative  = []
    failed       = 0
    log_interval = max(1, len(test_data) // 10)  # log 10 times

    for i, rec in enumerate(test_data):
        try:
            image = Image.open(rec["image"]).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open image {rec['image']}: {e}", flush=True)
            failed += 1
            continue

        gt_captions = rec.get("_gt_captions", [rec["conversations"][1]["content"]])

        try:
            pred = generate_caption(model, tokenizer, image)
        except Exception as e:
            print(f"  [WARN] Generate failed at sample {i}: {e}", flush=True)
            pred = ""

        if not pred:
            pred = gt_captions[0]  # fallback — tránh empty pred ảnh hưởng BLEU

        hypotheses.append(pred.split())
        references.append([gt.split() for gt in gt_captions])

        if i < 20:
            qualitative.append({
                "image":          rec["image"],
                "ground_truth":   gt_captions[0],
                "qwen2vl_output": pred,
            })

        # Progress — log mỗi 10% hoặc sample cuối
        if (i + 1) % log_interval == 0 or i == len(test_data) - 1:
            print(f"  [{i+1:3d}/{len(test_data)}] latest pred: {pred[:60]}...",
                  flush=True)

    print(f"\n[EVAL] Done. Success={len(hypotheses)}, Failed/skipped={failed}",
          flush=True)

    if not hypotheses:
        print("❌ No valid predictions — cannot compute BLEU", flush=True)
        sys.exit(1)

    # ── BLEU scores ───────────────────────────────────────────────
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses,
                     weights=(1,0,0,0), smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses,
                     weights=(0.5,0.5,0,0), smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses,
                     weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses,
                     weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    print(f"\n[RESULT] BLEU-1={b1:.4f} | BLEU-2={b2:.4f} | "
          f"BLEU-3={b3:.4f} | BLEU-4={b4:.4f}", flush=True)
    print(f"[BASELINE] BLIP+ViT5 (Jaccard): BLEU-4=0.1951", flush=True)
    if b4 > 0.1951:
        print(f"[RESULT] ✅ Qwen2-VL-7B vượt baseline! Δ=+{b4-0.1951:.4f}",
              flush=True)
    else:
        print(f"[RESULT] ⚠️  Qwen2-VL-7B chưa vượt baseline. Δ={b4-0.1951:.4f}",
              flush=True)

    # ── Save CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "pipeline_evaluation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "BLIP_ViT5_Jaccard", "Qwen2VL_7B", "Delta"])
        for metric, baseline, score in [
            ("BLEU1", 0.4552, b1), ("BLEU2", 0.3196, b2),
            ("BLEU3", 0.2493, b3), ("BLEU4", 0.1951, b4),
        ]:
            w.writerow([metric, f"{baseline:.4f}", f"{score:.4f}",
                        f"{score - baseline:+.4f}"])
    print(f"[SAVE] {csv_path}", flush=True)

    # ── Save qualitative ──────────────────────────────────────────
    qual_path = os.path.join(args.output_dir, "qualitative_examples.json")
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump(qualitative, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {qual_path}", flush=True)

    print("\n✓ Evaluation done!", flush=True)


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
