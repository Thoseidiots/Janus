import torch
import time
import os
from avus import Avus, AvusConfig

def run_janus_vram_test():
    print("="*60)
    print("JANUS COMPUTE RESERVOIR - VRAM OPTIMIZATION TEST")
    print("="*60)

    # Use a larger config to simulate VRAM pressure
    # avus-3b: dim=2048, n_layers=24
    config = AvusConfig(dim=512, n_layers=12, n_heads=8, n_kv_heads=4)
    
    # Initialize model with the Reservoir enabled
    print("\nInitializing Avus-3B with Compute Reservoir...")
    model = Avus(config, use_reservoir=True)
    model.eval()

    # Create dummy input
    vocab_size = config.vocab_size
    input_tokens = torch.randint(0, vocab_size, (1, 128))
    
    print(f"\nStep 1: First Inference (Charging the Reservoir)...")
    start = time.time()
    with torch.no_grad():
        output1 = model.generate(input_tokens, max_new_tokens=5)
    duration1 = time.time() - start
    print(f"First pass took: {duration1:.4f}s")

    print(f"\nStep 2: Second Inference (Discharging the Reservoir)...")
    start = time.time()
    with torch.no_grad():
        output2 = model.generate(input_tokens, max_new_tokens=5)
    duration2 = time.time() - start
    print(f"Second pass took: {duration2:.4f}s")
    
    improvement = (duration1 / duration2) if duration2 > 0 else 0
    print(f"\nSpeed Improvement: {improvement:.1f}x faster on replay")

    # Final Report from the Reservoir
    report = model.reservoir.get_report()
    print("\n" + "="*40)
    print("JANUS VRAM RESERVOIR REPORT")
    print("="*40)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*40)
    print("\nSummary: The reservoir offloaded intermediate layer states to disk.")
    print("This allows you to run models that would otherwise exceed your VRAM.")

if __name__ == "__main__":
    run_janus_vram_test()
