"""
Toy task example demonstrating HBM usage, training, spawning, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from holographic_brain_memory import HolographicBrainMemory, PhaseBrainLayer, SpawningBrain
from holographic_brain_memory.visualization import MemoryTraceVisualizer


def toy_task_example():
    """Run a toy task demonstrating HBM capabilities."""
    
    print("=" * 60)
    print("Holographic Brain Memory - Toy Task Example")
    print("=" * 60)
    
    # Configuration
    in_features = 16
    memory_dim = 256
    num_steps = 100
    
    # Initialize components
    memory = HolographicBrainMemory(dim=memory_dim)
    layer = PhaseBrainLayer(in_dim=in_features, memory=memory)
    
    # Dummy task: predict next value in sequence
    criterion = nn.MSELoss()
    optimizer = optim.Adam(layer.parameters(), lr=0.01)
    
    # Training loop
    memory_magnitudes = []
    losses = []
    
    print(f"\nTraining for {num_steps} steps...")
    for step in range(num_steps):
        # Generate random input and target
        x = torch.randn(1, in_features)
        y = torch.randn(1, layer.out_dim)
        
        # Forward pass
        output = layer(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        memory_magnitudes.append(memory.get_magnitude())
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Loss = {loss.item():.4f}, Memory Magnitude = {memory_magnitudes[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Spawning Brain Example")
    print("=" * 60)
    
    # Initialize spawning brain
    brain = SpawningBrain(in_dim=in_features, memory_dim=memory_dim, initial_layers=1)
    print(f"Initial neurons: {brain.get_logical_capacity()}")
    print(f"Physical memory footprint: {brain.get_memory_footprint()} bytes")
    
    # Simulate spawning
    print(f"\nSimulating spawning with threshold=0.05...")
    for i in range(10):
        dummy_input = torch.randn(1, in_features)
        if brain.check_and_spawn(dummy_input, threshold=0.05):
            print(f"  Step {i}: New neuron spawned! Total neurons: {brain.get_logical_capacity()}")
    
    print(f"\nFinal neurons: {brain.get_logical_capacity()}")
    print(f"Capacity ratio: {brain.get_capacity_ratio():.2f} neurons/byte")
    
    # Visualization
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent.parent / "demo_frames"
    output_dir.mkdir(exist_ok=True)
    
    # Plot training metrics
    metrics_path = output_dir / "training_metrics.png"
    MemoryTraceVisualizer.plot_training_metrics(
        {'Loss': losses, 'Memory Magnitude': memory_magnitudes},
        save_path=str(metrics_path)
    )
    print(f"Saved training metrics to {metrics_path}")
    
    # Plot memory phase distribution
    phase_path = output_dir / "memory_phase_distribution.png"
    MemoryTraceVisualizer.plot_memory_phase_distribution(
        memory.memory,
        save_path=str(phase_path)
    )
    print(f"Saved phase distribution to {phase_path}")
    
    # Plot memory magnitude distribution
    mag_path = output_dir / "memory_magnitude_distribution.png"
    MemoryTraceVisualizer.plot_memory_magnitude_distribution(
        memory.memory,
        save_path=str(mag_path)
    )
    print(f"Saved magnitude distribution to {mag_path}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    toy_task_example()
