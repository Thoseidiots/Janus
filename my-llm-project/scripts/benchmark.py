import torch
import time
import sys
import os

# Add Janus to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.config import JanusConfig
from core.model import JanusModel

def benchmark_inference():
    config = JanusConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JanusModel(config).to(device)
    model.eval()
    
    # Generate dummy input
    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size)).to(device)
    
    print(f"Benchmarking inference on {device}...")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)
            
    # Benchmark
    start_time = time.time()
    n_iters = 20
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(input_ids)
            
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iters
    tokens_per_sec = config.block_size / avg_time
    
    print(f"Inference Results:")
    print(f"  Average Time per Forward Pass: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_inference()
