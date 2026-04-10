import torch
from torch.utils.data import Dataset, DataLoader
import os

class JanusDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data_path = data_path
        self.block_size = block_size
        
        if not os.path.exists(data_path):
            # Create a dummy dataset if none exists for testing
            print(f"Warning: {data_path} not found. Creating dummy data for testing.")
            self.data = torch.randint(0, 50304, (10000,))
        else:
            # In a real scenario, we would load and tokenize the data here
            # For now, we'll assume the data is a pre-tokenized tensor file
            self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(data_path, batch_size, block_size, shuffle=True):
    dataset = JanusDataset(data_path, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
