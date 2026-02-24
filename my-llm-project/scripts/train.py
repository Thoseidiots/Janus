import torch
import torch.optim as optim
import sys
import os
import time
from data_loader import get_dataloader

# Add Janus to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.config import JanusConfig
from core.model import JanusModel

def train():
    # Configuration
    config = JanusConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 1
    learning_rate = 3e-4
    
    # Model initialization
    model = JanusModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Data loading
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/train_data.pt"))
    dataloader = get_dataloader(data_path, batch_size, config.block_size)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/janus_checkpoint.pt"))
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config.__dict__
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
