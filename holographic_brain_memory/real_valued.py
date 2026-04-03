# Holographic Brain Memory (HBM) Library

This repository contains a PyTorch-based implementation of a Holographic Brain Memory (HBM) system, inspired by Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA). The core idea is to enable neural network-like capabilities where logical model capacity can scale significantly while maintaining a fixed or even shrinking physical byte footprint.

## Key Features

*   **Fixed-Size Memory with Growing Capacity:** Utilizes holographic superposition to store numerous logical items (e.g., 
neurons, associations, hierarchies) within a single, fixed-size complex vector.
*   **Phase-Only Weights:** Leverages phase-only weights for efficient binding and unbinding operations via Fast Fourier Transform (FFT), making it hardware-friendly.
*   **Graceful Degradation:** The system degrades softly under high load, providing robust performance.
*   **Singularity Spawning:** Dynamically adds new 
logical components ("neurons") into the fixed memory buffer when certain conditions (e.g., low output magnitude) are met, simulating brain-like growth without increasing physical memory.
*   **Byte Compression:** Includes mechanisms to quantize and fold the memory to fit within strict byte budgets, ideal for edge devices and microcontrollers.
*   **Real-Valued HRR:** Provides an alternative real-valued implementation for enhanced hardware compatibility.
*   **Visualization Tools:** Offers utilities to visualize the dynamic evolution of the holographic memory trace.

## Installation

To install the HBM library and its dependencies, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/holographic_brain_memory.git
    cd holographic_brain_memory
    ```
2.  **Install PyTorch:**
    HBM requires PyTorch. Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions specific to your system and desired CUDA version. For CPU-only environments, you can typically install it using pip:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
3.  **Install other dependencies:**
    ```bash
    pip install matplotlib numpy
    ```

## Usage

### Core Components

The `HolographicBrainMemory` class manages the holographic memory, while `PhaseBrainLayer` represents a computational unit that interacts with this memory.

```python
import torch
from holographic_brain_memory.core import HolographicBrainMemory, PhaseBrainLayer

# Initialize holographic memory
memory_dim = 1024
hbm = HolographicBrainMemory(dim=memory_dim)

# Create a PhaseBrainLayer that uses the HBM
in_features = 64
layer = PhaseBrainLayer(in_dim=in_features, memory=hbm)

# Example input
input_data = torch.randn(1, in_features) # Batch size 1

# Forward pass: encodes input, writes to HBM, and reads back
output = layer(input_data)

print(f"Output shape: {output.shape}")
```

### Singularity Spawning

The `SpawningBrain` class demonstrates how logical capacity can grow dynamically.

```python
import torch
from holographic_brain_memory.spawning import SpawningBrain

in_features = 16
memory_dim = 1024

brain = SpawningBrain(in_dim=in_features, memory_dim=memory_dim, initial_layers=1)
print(f"Initial neurons: {len(brain.layers)}")

# Simulate some processing and check for spawning
for _ in range(5):
    dummy_input = torch.randn(1, in_features)
    _ = brain(dummy_input) # Process input
    if brain.check_and_spawn(dummy_input, threshold=0.05):
        print(f"New neuron spawned! Total neurons: {len(brain.layers)}")

print(f"Final neurons: {len(brain.layers)}")
print(f"Physical memory footprint: {brain.get_memory_footprint()} bytes (fixed)")
```

### Real-Valued HRR

For scenarios where complex numbers are not desired or for specific hardware optimizations, a real-valued HRR implementation is available.

```python
import torch
from holographic_brain_memory.real_valued import RealHolographicMemory, RealPhaseBrainLayer

real_mem_dim = 512
real_hbm = RealHolographicMemory(dim=real_mem_dim)

real_in_features = 32
real_layer = RealPhaseBrainLayer(in_dim=real_in_features, memory=real_hbm)

real_input_data = torch.randn(1, real_in_features)
real_output = real_layer(real_input_data)

print(f"Real-valued HBM output shape: {real_output.shape}")
```

## Examples

Refer to the `examples/toy_task.py` script for a complete demonstration, including a training loop, singularity spawning, and visualization generation.

To run the example:

```bash
python holographic_brain_memory/examples/toy_task.py
```

This will generate `training_metrics.png` and a series of animation frames in the `demo_frames/` directory, visualizing the holographic memory trace over time.

## Testing

Unit tests are provided to ensure the correctness of the core components. To run the tests:

```bash
python holographic_brain_memory/tests/test_core.py
```

## Project Structure

```
holographic_brain_memory/
├── __init__.py
├── core.py             # Contains HolographicBrainMemory and PhaseBrainLayer (complex-valued HRR)
├── real_valued.py      # Contains RealHolographicMemory and RealPhaseBrainLayer (real-valued HRR)
├── spawning.py         # Contains SpawningBrain for dynamic layer creation
├── visualization.py    # Utilities for visualizing memory traces and training metrics
├── examples/           # Example scripts demonstrating usage
│   └── toy_task.py
└── tests/
    └── test_core.py    # Unit tests for core and real-valued HRR components
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: LICENSE file is not included in this response, but would be a standard addition in a real project.)

## Acknowledgments

Inspired by the concepts of Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA), and the user's vision of a "temporal dynamo" and "brain-like folding" in fixed physical space. Special thanks to the original prompt for the insightful analogies and initial code sketch.
