import torch
import sys
import os
import math
from data_loader import get_dataloader

# Add Janus to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.config import JanusConfig
from core.model import JanusModel

def evaluate():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    
    # Load checkpoint
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/janus_best.pt"))
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = JanusConfig(**ckpt["config"])
    
    # Model initialization
    model = JanusModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Data loading
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/val_data.pt"))
    dataloader = get_dataloader(data_path, batch_size, config.block_size, shuffle=False)
    
    # Evaluation loop
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Evaluating model on {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    
    print(f"Evaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    evaluate()
