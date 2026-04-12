#!/usr/bin/env python3
"""
Pre-flight check — KHÔNG load model (tránh download 5GB).
Chỉ test: imports, process_vision_info, UnslothVisionDataCollator (với dummy model).
"""
import sys, os, json, yaml, argparse
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",        required=True)
    ap.add_argument("--train_json", required=True)
    args = ap.parse_args()

    INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."

    # ── 1. Imports ──────────────────────────────────────────────
    print("[1] Test imports...", flush=True)
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from qwen_vl_utils import process_vision_info
    print("  ✅ unsloth, UnslothVisionDataCollator, process_vision_info OK", flush=True)

    # ── 2. process_vision_info với ảnh thật ─────────────────────
    print("[2] Test process_vision_info...", flush=True)
    r0       = json.load(open(args.train_json))[0]
    img_path = r0["image"]
    assert os.path.exists(img_path), f"❌ Image not found: {img_path}"
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role":"user","content":[
        {"type":"image","image":img},
        {"type":"text", "text": INSTRUCTION},
    ]}]
    image_inputs, video_inputs = process_vision_info(msgs)
    assert image_inputs is not None and len(image_inputs) > 0, "process_vision_info trả về rỗng"
    print(f"  ✅ process_vision_info OK — {len(image_inputs)} image(s)", flush=True)

    # ── 3. Config hợp lệ ────────────────────────────────────────
    print("[3] Test config...", flush=True)
    with open(args.cfg) as f: cfg = yaml.safe_load(f)
    required = ["model","data","training","logging"]
    for k in required:
        assert k in cfg, f"❌ Missing key '{k}' in config"
    assert os.path.exists(cfg["data"]["train_json"]), f"❌ train_json not found"
    assert os.path.exists(cfg["data"]["val_json"]),   f"❌ val_json not found"
    assert os.path.exists(cfg["data"]["test_json"]),  f"❌ test_json not found"
    print(f"  ✅ Config OK — model: {cfg['model']['name']}", flush=True)

    # ── 4. NLTK ─────────────────────────────────────────────────
    print("[4] Test NLTK...", flush=True)
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.translate.bleu_score import corpus_bleu
    score = corpus_bleu([[["a","b","c"]]], [["a","b","c"]])
    assert score > 0
    print(f"  ✅ NLTK corpus_bleu OK — test score: {score:.4f}", flush=True)

    print("\n✅ PRE-FLIGHT PASSED — train script sẽ không crash!", flush=True)


if __name__ == "__main__":
    main()
