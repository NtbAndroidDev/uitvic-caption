"""
Script đánh giá pipeline đầy đủ End-to-End:
  Ảnh → BLIP (Stage 1) → câu thô → ViT5 (Stage 2) → câu hoàn chỉnh
  → So sánh BLEU với ground truth

Kết quả:
  - BLEU-1,2,3,4 của BLIP alone
  - BLEU-1,2,3,4 của BLIP + ViT5 (pipeline đầy đủ)
  - Bảng so sánh CSV
  - Biểu đồ cột so sánh 2 mô hình
  - 10 ví dụ minh họa định tính

Usage:
    python scripts/evaluate_pipeline.py \
        --blip_model Salesforce/blip-image-captioning-base \
        --blip_ckpt  /kaggle/input/datasets/ntb1nh/stage1/best_model.pt \
        --vit5_model VietAI/vit5-base \
        --vit5_ckpt  /kaggle/working/uitvic-caption/outputs/stage2_checkpoints_blip/best_model.pt \
        --test_json  /kaggle/input/.../uitvic_captions_test2017.json \
        --test_images /kaggle/input/.../coco_uitvic_test \
        --output_dir outputs/evaluation \
        --max_samples 500
"""

import os
import sys
import json
import csv
import math
import argparse
from collections import Counter
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, BlipConfig,
    AutoModelForSeq2SeqLM, T5Tokenizer
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from huggingface_hub import hf_hub_download

try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score not installed. Run: pip install rouge-score")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import UITViCCOCODataset


def load_blip(blip_model_name, blip_ckpt_path, device):
    print("[PIPELINE] Loading BLIP processor...")
    processor = BlipProcessor.from_pretrained(blip_model_name)

    print("[PIPELINE] Loading BLIP architecture (config only)...")
    config = BlipConfig.from_pretrained(blip_model_name)
    model = BlipForConditionalGeneration(config)

    print(f"[PIPELINE] Loading BLIP checkpoint: {blip_ckpt_path}")
    ckpt = torch.load(blip_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("[PIPELINE] BLIP loaded!")
    return processor, model


def load_vit5(vit5_model_name, vit5_ckpt_path, device):
    print("[PIPELINE] Loading ViT5 tokenizer (local dir, skip tokenizer.json)...")
    import shutil
    _tok_dir = "/tmp/vit5_tok_eval"
    os.makedirs(_tok_dir, exist_ok=True)
    if os.path.isdir(vit5_model_name):
        # Local cache → copy trực tiếp
        for _f in ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]:
            _src = os.path.join(vit5_model_name, _f)
            if os.path.exists(_src):
                shutil.copy(_src, f"{_tok_dir}/{_f}")
    else:
        # HuggingFace repo ID → download
        for _f in ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]:
            shutil.copy(hf_hub_download(repo_id=vit5_model_name, filename=_f), f"{_tok_dir}/{_f}")
    tokenizer = T5Tokenizer.from_pretrained(_tok_dir, use_fast=False)

    print("[PIPELINE] Loading ViT5 model...")
    vit5 = AutoModelForSeq2SeqLM.from_pretrained(vit5_model_name)

    print(f"[PIPELINE] Loading ViT5 checkpoint: {vit5_ckpt_path}")
    state_dict = torch.load(vit5_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    vit5.load_state_dict(state_dict)
    vit5 = vit5.to(device).eval()
    print("[PIPELINE] ViT5 loaded!")
    return tokenizer, vit5



def correct_with_vit5(raw_captions, tokenizer, vit5_model, device, max_length=64):
    inputs = tokenizer(
        raw_captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    with torch.no_grad():
        outputs = vit5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Nếu output rỗng, fallback về input gốc (BLIP caption)
    result = []
    for dec, raw in zip(decoded, raw_captions):
        result.append(dec.strip() if dec.strip() else raw.strip())
    return result


def compute_bleu(references, hypotheses):
    smooth = SmoothingFunction().method1
    refs = [[r.strip().split()] for r in references]
    hyps = [h.strip().split() for h in hypotheses]
    b1 = corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=smooth)
    b2 = corpus_bleu(refs, hyps, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    b3 = corpus_bleu(refs, hyps, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    b4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    return {"bleu1": b1, "bleu2": b2, "bleu3": b3, "bleu4": b4}


def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_cider(hypotheses: list, references: list, n: int = 4) -> float:
    """Simplified CIDEr-D (TF-IDF cosine similarity, x10 scale)."""
    doc_freq = Counter()
    num_docs  = len(references)
    for refs in references:
        seen = set()
        for ref in refs:
            for k in range(1, n + 1):
                for ng in set(_ngrams(ref.split(), k)):
                    if ng not in seen:
                        doc_freq[ng] += 1
                        seen.add(ng)

    def get_tf(tokens):
        c = Counter()
        for k in range(1, n + 1):
            c.update(_ngrams(tokens, k))
        return c

    def tfidf(tf_dict):
        vec = {}
        for ng, cnt in tf_dict.items():
            idf = math.log((num_docs + 1.0) / (doc_freq.get(ng, 0) + 1.0))
            vec[ng] = cnt * idf
        return vec

    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_vec = tfidf(get_tf(hyp.split()))
        avg_ref_tf = Counter()
        for ref in refs:
            for k, v in get_tf(ref.split()).items():
                avg_ref_tf[k] += v / len(refs)
        ref_vec = tfidf(avg_ref_tf)
        dot    = sum(hyp_vec.get(k, 0) * v for k, v in ref_vec.items())
        norm_h = math.sqrt(sum(v ** 2 for v in hyp_vec.values())) + 1e-10
        norm_r = math.sqrt(sum(v ** 2 for v in ref_vec.values())) + 1e-10
        scores.append(dot / (norm_h * norm_r))
    return (sum(scores) / len(scores)) * 10.0 if scores else 0.0


def compute_extra_metrics(hypotheses_str: list, references_str: list) -> dict:
    """Compute METEOR, ROUGE-L, CIDEr for a list of hypotheses vs references."""
    # METEOR
    meteor_scores = []
    for hyp, refs in zip(hypotheses_str, references_str):
        meteor_scores.append(meteor_score([r.split() for r in refs], hyp.split()))
    meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # ROUGE-L
    if HAS_ROUGE:
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = []
        for hyp, refs in zip(hypotheses_str, references_str):
            best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
            rouge_scores.append(best)
        rougeL = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    else:
        rougeL = 0.0

    # CIDEr
    cider = compute_cider(hypotheses_str, references_str)
    return {"meteor": meteor, "rougeL": rougeL, "cider": cider}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PIPELINE] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    blip_processor, blip_model = load_blip(args.blip_model, args.blip_ckpt, device)
    vit5_tokenizer, vit5_model = load_vit5(args.vit5_model, args.vit5_ckpt, device)

    # Load test dataset
    print(f"[PIPELINE] Loading test dataset: {args.test_json}")
    dataset = UITViCCOCODataset(args.test_json, args.test_images, blip_processor, max_length=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[PIPELINE] Test: {len(dataset)} samples, {len(loader)} batches")

    blip_refs, blip_hyps, pipeline_hyps = [], [], []
    qualitative_examples = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if args.max_samples > 0 and step * args.batch_size >= args.max_samples:
                break

            pixel_values = batch["pixel_values"].to(device)

            # Step 1: BLIP generate raw caption
            generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=64, num_beams=4)
            blip_captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Step 2: ViT5 correct caption
            corrected_captions = correct_with_vit5(blip_captions, vit5_tokenizer, vit5_model, device)

            # Ground truth: dùng raw string từ JSON (có dấu), KHÔNG decode qua BLIP tokenizer
            gt_captions = batch["raw_caption"]  # list[str] có tiếng Việt đầy đủ dấu

            for blip_cap, corr_cap, gt_cap in zip(blip_captions, corrected_captions, gt_captions):
                blip_refs.append(gt_cap)
                blip_hyps.append(blip_cap)
                pipeline_hyps.append(corr_cap)

                # Chỉ lấy ví dụ từ ảnh KHÁC NHAU (tránh trùng 5 caption/ảnh)
                if len(qualitative_examples) < 10 and not any(
                    ex["blip_output"] == blip_cap for ex in qualitative_examples
                ):
                    qualitative_examples.append({
                        "ground_truth": gt_cap,
                        "blip_output": blip_cap,
                        "pipeline_output": corr_cap
                    })

            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{min(len(loader), args.max_samples // args.batch_size)}")

    # Compute BLEU
    print("\n[PIPELINE] Computing BLEU scores...")
    blip_bleu = compute_bleu(blip_refs, blip_hyps)
    full_bleu = compute_bleu(blip_refs, pipeline_hyps)

    # Compute extra metrics
    print("[PIPELINE] Computing METEOR / ROUGE-L / CIDEr...")
    blip_extra = compute_extra_metrics(blip_hyps,     [[r] for r in blip_refs])
    full_extra = compute_extra_metrics(pipeline_hyps, [[r] for r in blip_refs])

    # Print results
    print("\n" + "="*65)
    print("KẾT QUẢ ĐÁNH GIÁ PIPELINE")
    print("="*65)
    print(f"{'Metric':<12} {'BLIP Only':>14} {'BLIP + ViT5':>14} {'Improvement':>12}")
    print("-"*55)
    for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
        b = blip_bleu[k]; p = full_bleu[k]
        print(f"{k.upper():<12} {b:>14.4f} {p:>14.4f} {p-b:>+12.4f}")
    for label, bk, fk in [
        ("METEOR",  blip_extra["meteor"],  full_extra["meteor"]),
        ("ROUGE-L", blip_extra["rougeL"],  full_extra["rougeL"]),
        ("CIDEr",   blip_extra["cider"],   full_extra["cider"]),
    ]:
        print(f"{label:<12} {bk:>14.4f} {fk:>14.4f} {fk-bk:>+12.4f}")
    print("="*65)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "pipeline_evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "BLIP_Only", "BLIP_ViT5_Jaccard", "Improvement"])
        for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
            writer.writerow([k.upper(), f"{blip_bleu[k]:.4f}", f"{full_bleu[k]:.4f}", f"{full_bleu[k]-blip_bleu[k]:+.4f}"])
        for label, bk, fk in [
            ("METEOR",  blip_extra["meteor"],  full_extra["meteor"]),
            ("ROUGE-L", blip_extra["rougeL"],  full_extra["rougeL"]),
            ("CIDEr",   blip_extra["cider"],   full_extra["cider"]),
        ]:
            writer.writerow([label, f"{bk:.4f}", f"{fk:.4f}", f"{fk-bk:+.4f}"])
    print(f"\n[SAVED] CSV: {csv_path}")

    # Save qualitative examples
    qual_path = os.path.join(args.output_dir, "qualitative_examples.json")
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump(qualitative_examples, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Qualitative examples: {qual_path}")

    # Save chart - Bar chart so sánh 2 mô hình
    metrics  = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    blip_vals = [blip_bleu["bleu1"], blip_bleu["bleu2"], blip_bleu["bleu3"], blip_bleu["bleu4"]]
    full_vals = [full_bleu["bleu1"], full_bleu["bleu2"], full_bleu["bleu3"], full_bleu["bleu4"]]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], blip_vals, width, label="BLIP Only", color="#4C72B0")
    bars2 = ax.bar([i + width/2 for i in x], full_vals, width, label="BLIP + ViT5", color="#DD8452")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("So sánh BLEU: BLIP Only vs BLIP + ViT5 (Pipeline đầy đủ)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, "pipeline_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[SAVED] Chart: {chart_path}")

    # Print qualitative examples
    print("\n--- 5 VÍ DỤ ĐỊNH TÍNH ---")
    for i, ex in enumerate(qualitative_examples[:5]):
        print(f"\n[{i+1}] Ground Truth   : {ex['ground_truth']}")
        print(f"    BLIP Output    : {ex['blip_output']}")
        print(f"    Pipeline Output: {ex['pipeline_output']}")

    print(f"\n[DONE] Kết quả đã lưu tại: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip_model",  type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--blip_ckpt",   type=str, required=True)
    parser.add_argument("--vit5_model",  type=str, default="VietAI/vit5-base")
    parser.add_argument("--vit5_ckpt",   type=str, required=True)
    parser.add_argument("--test_json",   type=str, required=True)
    parser.add_argument("--test_images", type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="outputs/evaluation")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = full test set")
    args = parser.parse_args()
    evaluate(args)
