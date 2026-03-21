import argparse
from src.train import train
from src.train_stage2 import train_stage2
from src.eval import evaluate

def main():
    parser = argparse.ArgumentParser(description="UITViC Image Captioning CLI")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "train_stage2", "evaluate", "serve"], 
                        help="Mode: train (Stage 1), train_stage2 (ViT5), evaluate, serve")
    parser.add_argument("--config", type=str, default="configs/train_blip.yaml", help="Path to config file")
    
    # Common evaluation/inference arguments
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config)
    elif args.mode == "train_stage2":
        train_stage2(args.config)
    elif args.mode == "evaluate":
        # logic evaluate ở src.eval đã có main riêng, nhưng mình có thể gọi ở đây nếu cần
        pass
    elif args.mode == "serve":
        print("Serve mode coming soon...")

if __name__ == "__main__":
    main()