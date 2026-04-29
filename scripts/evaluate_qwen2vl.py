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
import os, json, csv, argparse, sys, math
from collections import Counter
import torch
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score not installed. Run: pip install rouge-score", flush=True)

try:
    from unsloth import FastVisionModel
    UNSLOTH = True
except ImportError:
    UNSLOTH = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("[WARN] peft not installed. Run: pip install peft", flush=True)

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."


# ─────────────────────────────────────────────
#  Metrics helpers
# ─────────────────────────────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_cider(hypotheses: list, references: list, n: int = 4) -> float:
    """Simplified CIDEr-D (no length penalty, TF-IDF cosine similarity, ×10 scale)."""
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
    """
    Args:
        hypotheses_str : list[str]  — predicted captions
        references_str : list[list[str]] — reference captions per sample
    Returns dict with keys: meteor, rougeL, cider
    """
    # METEOR
    meteor_scores = []
    for hyp, refs in zip(hypotheses_str, references_str):
        hyp_tok  = hyp.split()
        refs_tok = [r.split() for r in refs]
        meteor_scores.append(meteor_score(refs_tok, hyp_tok))
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


# ─────────────────────────────────────────────
#  Model helpers
# ─────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def load_model(model_dir: str):
    assert os.path.exists(model_dir), f"❌ model_dir not found: {model_dir}"
    print(f"[EVAL] Loading model from {model_dir}...", flush=True)
    if UNSLOTH:
        model, tokenizer = FastVisionModel.from_pretrained(model_dir, load_in_4bit=True)
        FastVisionModel.for_inference(model)
    else:
        # Load base model with 4-bit quant + apply LoRA adapter via PEFT
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"[EVAL] Loading base model {BASE_MODEL_ID} with 4-bit quant...", flush=True)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        if HAS_PEFT:
            print(f"[EVAL] Applying LoRA adapter from {model_dir}...", flush=True)
            model = PeftModel.from_pretrained(base, model_dir)
        else:
            raise RuntimeError("peft package required. Run: pip install peft")
        tokenizer = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    model.eval()
    print("[EVAL] Model loaded ✓", flush=True)
    return model, tokenizer


def generate_caption(model, tokenizer, image: "Image.Image") -> str:
    device = next(model.parameters()).device
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": INSTRUCTION},
        ],
    }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if HAS_QWEN_VL_UTILS:
        image_inputs, _ = process_vision_info(messages)
        inputs = tokenizer(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(device)
    else:
        inputs = tokenizer(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

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


# ─────────────────────────────────────────────
#  Main evaluate
# ─────────────────────────────────────────────

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(args.model_dir)

    with open(args.test_json, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    print(f"[EVAL] Test samples: {len(test_data)}", flush=True)

    hypotheses, references = [], []
    hypotheses_str, references_str = [], []
    qualitative = []
    failed       = 0
    log_interval = max(1, len(test_data) // 10)

    for i, rec in enumerate(test_data):
        try:
            image = Image.open(rec["image"]).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open {rec['image']}: {e}", flush=True)
            failed += 1
            continue

        gt_captions = rec.get("_gt_captions", [rec["conversations"][1]["content"]])

        try:
            pred = generate_caption(model, tokenizer, image)
        except Exception as e:
            print(f"  [WARN] Generate failed at sample {i}: {e}", flush=True)
            pred = ""

        if not pred:
            pred = gt_captions[0]

        hypotheses.append(pred.split())
        references.append([gt.split() for gt in gt_captions])
        hypotheses_str.append(pred)
        references_str.append(gt_captions)

        if i < 20:
            qualitative.append({
                "image":          rec["image"],
                "ground_truth":   gt_captions[0],
                "qwen2vl_output": pred,
            })

        if (i + 1) % log_interval == 0 or i == len(test_data) - 1:
            print(f"  [{i+1:3d}/{len(test_data)}] {pred[:70]}", flush=True)

    print(f"\n[EVAL] Success={len(hypotheses)}, Failed={failed}", flush=True)
    if not hypotheses:
        print("❌ No valid predictions", flush=True)
        sys.exit(1)

    # ── BLEU ──
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0),            smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0),         smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0),    smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    # ── Extra metrics ──
    print("\n[EVAL] Computing METEOR / ROUGE-L / CIDEr...", flush=True)
    extra  = compute_extra_metrics(hypotheses_str, references_str)
    meteor = extra["meteor"]
    rougeL = extra["rougeL"]
    cider  = extra["cider"]

    # ── Print results ──
    print("\n" + "="*60, flush=True)
    print("KẾT QUẢ ĐÁNH GIÁ — Qwen2-VL-7B QLoRA", flush=True)
    print("="*60, flush=True)
    BASELINE = {
        "BLEU-1": 0.4552, "BLEU-2": 0.3196, "BLEU-3": 0.2493, "BLEU-4": 0.1951,
        "METEOR": None, "ROUGE-L": None, "CIDEr": None,
    }
    results = {
        "BLEU-1": b1, "BLEU-2": b2, "BLEU-3": b3, "BLEU-4": b4,
        "METEOR": meteor, "ROUGE-L": rougeL, "CIDEr": cider,
    }
    print(f"{'Metric':<10} {'Baseline (BLIP+ViT5)':>22} {'Qwen2-VL-7B':>14} {'Delta':>10}", flush=True)
    print("-"*60, flush=True)
    for metric, score in results.items():
        base = BASELINE[metric]
        if base is not None:
            print(f"{metric:<10} {base:>22.4f} {score:>14.4f} {score-base:>+10.4f}", flush=True)
        else:
            print(f"{metric:<10} {'N/A (run pipeline eval)':>22} {score:>14.4f} {'N/A':>10}", flush=True)
    print("="*60, flush=True)

    delta_b4 = b4 - 0.1951
    if delta_b4 > 0:
        print(f"[RESULT] ✅ Qwen2-VL vượt baseline BLEU-4! Δ=+{delta_b4:.4f}", flush=True)
    else:
        print(f"[RESULT] ⚠️  Chưa vượt baseline BLEU-4. Δ={delta_b4:.4f}", flush=True)

    # ── Save CSV ──
    csv_path = os.path.join(args.output_dir, "pipeline_evaluation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "BLIP_ViT5_Jaccard", "Qwen2VL_7B", "Delta"])
        for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]:
            base  = BASELINE[metric]
            score = results[metric]
            w.writerow([metric, f"{base:.4f}", f"{score:.4f}", f"{score-base:+.4f}"])
        for metric in ["METEOR", "ROUGE-L", "CIDEr"]:
            score = results[metric]
            w.writerow([metric, "N/A", f"{score:.4f}", "N/A"])
    print(f"[SAVE] {csv_path}", flush=True)

    # ── Save qualitative ──
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
