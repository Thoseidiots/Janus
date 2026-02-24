import torch
from core.model import JanusModel
from core.config import JanusConfig
import os

def init_and_save():
    config = JanusConfig()
    model = JanusModel(config)
    
    save_path = "/home/ubuntu/Janus/weights/janus_init.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model initialized with random weights and saved to {save_path}")
    
    # Also save a config.json for compatibility
    import json
    from dataclasses import asdict
    config_path = "/home/ubuntu/Janus/core/config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    init_and_save()
