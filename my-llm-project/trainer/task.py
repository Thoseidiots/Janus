import torch
import torch.nn as nn
import random

# All your curriculum functions (Pre-K, Elementary, etc.)
def generate_pre_k_dataset(samples=1000):
    dataset = []
    items = ["keys", "phone", "wallet", "toy", "apple"]
    containers_small = ["blue box", "small bag", "wooden crate"]
    containers_large = ["red suitcase", "backpack", "cardboard box"]
    locations = ["car", "park", "kitchen", "garage"]
    for _ in range(samples):
        item, c1, c2, loc = random.choice(items), random.choice(containers_small), random.choice(containers_large), random.choice(locations)
        prompt = f"I put the {item} in the {c1}. I put the {c1} inside the {c2}. I move the {c2} to the {loc}. Where is the {item}?"
        answer = f"The {item} is in the {c1}, which is inside the {c2} in the {loc}."
        dataset.append((prompt, answer))
    return dataset

# Minimal Model Logic for Training
class GroundedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 10) # Placeholder for the architecture
    def forward(self, x): return self.net(x)

print("Curriculum script initialized. Starting Pre-K stage...")
