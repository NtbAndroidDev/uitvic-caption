#!/usr/bin/env python3
"""Pre-flight check — runs as SUBPROCESS (fresh Python, no pyarrow conflict)."""
import sys, os, json, yaml, argparse, torch
from PIL import Image

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from qwen_vl_utils import process_vision_info

INSTRUCTION = "Hãy viết một câu mô tả ngắn gọn hình ảnh này bằng tiếng Việt."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",        required=True)
    ap.add_argument("--train_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PRE-FLIGHT] Device: {device}", flush=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f)

    # A. Load model
    print("[A] Load model + LoRA...", flush=True)
    model, tok = FastVisionModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers    = cfg["model"]["finetune_vision_layers"],
        finetune_language_layers  = cfg["model"]["finetune_language_layers"],
        finetune_attention_modules= cfg["model"]["finetune_attention_modules"],
        finetune_mlp_modules      = cfg["model"]["finetune_mlp_modules"],
        r            = cfg["training"]["lora_rank"],
        lora_alpha   = cfg["training"]["lora_alpha"],
        lora_dropout = cfg["training"]["lora_dropout"],
        use_gradient_checkpointing="unsloth",
    )
    print("  ✅ Model OK", flush=True)

    # B. Collator
    print("[B] Test collator (1 sample)...", flush=True)
    r0       = json.load(open(args.train_json))[0]
    img_path = r0["image"]
    cap      = r0["conversations"][1]["content"]
    msgs_tr  = [
        {"role":"user",      "content":[{"type":"image","image":img_path},{"type":"text","text":INSTRUCTION}]},
        {"role":"assistant", "content": cap},
    ]
    batch = UnslothVisionDataCollator(model, tok)([{"messages": msgs_tr}])
    print(f"  ✅ Collator OK — input_ids: {batch['input_ids'].shape}", flush=True)

    # C. Generate (tests process_vision_info + patched tokenizer call)
    print("[C] Test generate (process_vision_info)...", flush=True)
    FastVisionModel.for_inference(model); model.eval()
    img = Image.open(img_path).convert("RGB")
    msgs_inf = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":INSTRUCTION}]}]
    text     = tok.apply_chat_template(msgs_inf, tokenize=False, add_generation_prompt=True)
    img_inp, _ = process_vision_info(msgs_inf)
    inputs   = tok(text=[text], images=img_inp, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    pred = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"  ✅ Generate OK: '{pred[:70]}'", flush=True)

    print("\n✅ PRE-FLIGHT PASSED — train script sẽ không crash!", flush=True)


if __name__ == "__main__":
    main()
