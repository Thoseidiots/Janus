import torch
import sys
import os
import json

# Add Janus to path
sys.path.insert(0, "/home/ubuntu/Janus")

from core.config import JanusConfig
from core.model import JanusModel

def verify():
    ckpt_path = "/home/ubuntu/Janus/weights/janus_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    print("Keys in checkpoint:", ckpt.keys())
    
    # Check if 'config' and 'model_state_dict' exist
    if "config" not in ckpt or "model_state_dict" not in ckpt:
        print("Error: Checkpoint missing 'config' or 'model_state_dict'.")
        # Attempt to load with default config if missing
        config = JanusConfig()
        print("Using default JanusConfig for verification.")
    else:
        config = JanusConfig(**ckpt["config"])
        print("Loaded config from checkpoint:", ckpt["config"])

    try:
        model = JanusModel(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print("Successfully loaded model state dict!")
        
        # Test a dummy forward pass
        dummy_input = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _ = model(dummy_input)
        print("Forward pass successful! Output shape:", logits.shape)
        
    except Exception as e:
        print(f"Error during model loading or forward pass: {e}")

if __name__ == "__main__":
    verify()
