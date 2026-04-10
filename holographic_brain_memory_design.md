# Holographic Brain Memory Library Design Document

## 1. Introduction

This document outlines the design for transforming the provided `HolographicBrainMemory` and `PhaseBrainLayer` code snippets into a production-grade Python library. The core concept revolves around Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA), enabling a neural network-like system where logical model capacity can scale up while the physical byte footprint remains fixed or even shrinks. This is achieved through holographic superposition, where multiple 
logical items are superposed into a single fixed-size vector. The library will be built using PyTorch, leveraging its capabilities for tensor operations and neural network modules.

## 2. Core Concepts: Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA)

At the heart of this system are HRR and VSA, which utilize a holographic superposition principle. This allows for the encoding of numerous associations, role-filler pairs, and hierarchical structures within a single, fixed-size memory buffer. Unlike traditional neural networks where adding more neurons or layers leads to a linear or quadratic increase in stored weights and bytes, HRR/VSA allows for logical capacity growth with minimal physical memory expansion. Retrieval noise increases gracefully with load, ensuring a soft degradation rather than catastrophic failure.

Key aspects include:

*   **Fixed-size Memory:** Information is stored in a single complex vector, whose size (e.g., 512-4096 complex numbers) remains constant regardless of the number of logical items stored.
*   **Phase-only Weights:** The use of phase-only weights ( `w = e^{i\theta}` ) is crucial. These phases act as "keys" for binding and unbinding operations. The Fast Fourier Transform (FFT) enables extremely fast and hardware-friendly circular convolution, which is the core binding operation.
*   **Graceful Degradation:** As more information is superposed, retrieval noise increases gradually, allowing the system to maintain functionality even under high load.

## 3. Core Components

The primary component will be the `HolographicBrainMemory` class, which manages the holographic memory. The `PhaseBrainLayer` will serve as an example of how to interact with this memory, representing a "neuron" or layer that encodes and decodes information.

### 3.1. `HolographicBrainMemory` Class

This class will encapsulate the holographic memory and its fundamental operations.

**Attributes:**

*   `dim` (int): The dimensionality of the holographic memory vector. This determines the fixed physical byte footprint.
*   `decay` (float): A decay factor applied to the memory during write operations, allowing older information to gradually fade, preventing saturation.
*   `memory` ( `nn.Parameter` ): The core complex-valued tensor representing the holographic memory.
*   `proj` ( `nn.Linear` ): An optional linear projection layer for cleanup and projection, which can significantly boost capacity and stability.

**Methods:**

*   `__init__(self, dim: int = 1024, decay: float = 0.98)`: Initializes the memory with a given dimensionality and decay rate.
*   `bind(self, signal: torch.Tensor, theta: torch.Tensor) -> torch.Tensor`:
    *   **Purpose:** Performs the binding operation, phase-modulating a `signal` with a `theta` (phase vector) using circular convolution via FFT.
    *   **Inputs:**
        *   `signal`: The input signal tensor.
        *   `theta`: The phase vector tensor.
    *   **Output:** The bound (phase-modulated and convolved) tensor.
*   `write(self, signal: torch.Tensor, theta: torch.Tensor)`:
    *   **Purpose:** Superposes new information into the holographic memory. It binds the `signal` with `theta` and adds it to the existing memory, applying a decay factor.
    *   **Inputs:**
        *   `signal`: The signal to be written.
        *   `theta`: The phase key for the signal.
    *   **Internal Logic:**
        1.  Binds the `signal` and `theta` using the `bind` method.
        2.  Updates `self.memory.data` by applying the `decay` and adding the new bound information.
        3.  Normalizes the memory and applies the optional `proj` layer for cleanup and capacity restoration.
*   `read(self, theta: torch.Tensor) -> torch.Tensor`:
    *   **Purpose:** Retrieves information from the holographic memory using an inverse phase key.
    *   **Inputs:**
        *   `theta`: The phase key to unbind the desired information.
    *   **Output:** The unbound (retrieved) signal.
    *   **Internal Logic:**
        1.  Creates an inverse phase vector from `theta`.
        2.  Performs circular convolution (via FFT) of the memory with the inverse phase vector.
        3.  Returns the real part of the unbound tensor (or its absolute value for similarity/cleanup).
*   `compress_bytes(self, target_bytes: int)`:
    *   **Purpose:** Quantizes and potentially folds the memory to fit within a specified byte budget, demonstrating the fixed-byte footprint capability.
    *   **Inputs:**
        *   `target_bytes`: The desired maximum byte size for the memory.
    *   **Internal Logic:**
        1.  Quantizes the complex memory to `int8`.
        2.  If the current byte size exceeds `target_bytes`, it folds/aliases dimensions to reduce the footprint, leading to more aggressive overlap.

### 3.2. `PhaseBrainLayer` Class

This class will represent a conceptual "layer" or "neuron" that interacts with the `HolographicBrainMemory`.

**Attributes:**

*   `in_dim` (int): Input dimensionality for the encoder.
*   `memory` ( `HolographicBrainMemory` ): A reference to the shared holographic memory instance.
*   `encoder` ( `nn.Linear` ): A linear layer to encode input `x` into the memory's dimensionality.
*   `theta` ( `nn.Parameter` ): A learnable phase key associated with this specific layer/neuron.

**Methods:**

*   `__init__(self, in_dim: int, memory: HolographicBrainMemory)`: Initializes the layer with an input dimension and a shared `HolographicBrainMemory` instance.
*   `forward(self, x: torch.Tensor) -> torch.Tensor`:
    *   **Purpose:** Processes an input `x`, encodes it, writes it to the shared holographic memory, and then reads from the memory using its own phase key.
    *   **Inputs:**
        *   `x`: The input tensor for the layer.
    *   **Output:** The retrieved output from the holographic memory.
    *   **Internal Logic:**
        1.  Encodes the input `x` using `self.encoder`.
        2.  Calls `memory.write()` with the encoded signal and `self.theta`.
        3.  Calls `memory.read()` with `self.theta` to retrieve information.

## 4. Extended Features and Enhancements

To make this a production-grade library, the following features will be added:

### 4.1. Singularity Spawning (Brain-Like Growth Without Byte Growth)

This feature simulates the dynamic creation of new "neurons" or patterns within the fixed holographic memory. It will be triggered by specific conditions, such as low output magnitude or high interference, indicating a need for new representational capacity.

**Implementation Details:**

*   **Detection:** Monitor the output of `PhaseBrainLayer` (e.g., `torch.abs(y).mean() < threshold`).
*   **Spawning Mechanism:** When a singularity is detected, a new perturbed phase key (`new_theta`) and a small `seed_signal` will be generated and written into the *same* `HolographicBrainMemory` instance. This demonstrates that new patterns can emerge without allocating new parameters.

### 4.2. Complete Runnable Script with Training Loop

A comprehensive example script will be provided, demonstrating how to train a network built with `HolographicBrainMemory` and `PhaseBrainLayer` on a toy task.

**Components:**

*   **Dataset:** A simple synthetic dataset for demonstration (e.g., a classification or regression task).
*   **Model:** An instantiation of `HolographicBrainMemory` and one or more `PhaseBrainLayer` instances.
*   **Loss Function and Optimizer:** Standard PyTorch components.
*   **Training Logic:** Iterative training loop, including forward pass, loss calculation, backward pass, and optimizer step.
*   **Singularity Spawning Integration:** The training loop will include logic to trigger singularity spawning based on predefined conditions.

### 4.3. Real-Valued Version (Binary/Sparse HRR)

To enhance hardware friendliness and explore alternative representations, a real-valued version of HRR will be implemented. This might involve using binary or sparse vectors instead of complex numbers, potentially simplifying computations and reducing memory footprint further.

**Implementation Details:**

*   **Alternative Binding/Unbinding:** Adapt the `bind` and `read` operations to work with real-valued vectors, possibly using XOR or other real-valued convolution approximations.
*   **Quantization:** Explore more aggressive quantization schemes suitable for real-valued representations.

### 4.4. Visualization of the Growing Holographic Trace

An animated visualization will be created to illustrate the "glowing spiral/gear math art" concept, showing how new patterns (neurons) are folded into the fixed holographic memory.

**Implementation Details:**

*   **Data Collection:** During training or inference, capture snapshots of the `HolographicBrainMemory` state.
*   **Visualization Technique:** Use libraries like Matplotlib or Plotly to generate animated plots that represent the complex-valued memory as a spiral or similar geometric pattern, with new information causing denser, self-folding patterns.

### 4.5. Integration with OverlapLinear Code (if applicable)

If the user provides the "earlier OverlapLinear code," this library will demonstrate how to integrate `HolographicBrainMemory` with it, showcasing potential synergies.

## 5. Testing Strategy

Comprehensive unit tests will be developed to ensure the correctness and robustness of each component.

**Test Cases:**

*   **`HolographicBrainMemory`:**
    *   Initialization with different dimensions and decay rates.
    *   Correctness of `bind`, `write`, and `read` operations (e.g., writing and retrieving a known signal).
    *   Behavior of `compress_bytes` under various `target_bytes` values.
    *   Graceful degradation under high load (superposing many items).
*   **`PhaseBrainLayer`:**
    *   Correct forward pass logic.
    *   Interaction with `HolographicBrainMemory`.
*   **Singularity Spawning:**
    *   Verification that new patterns are spawned under specified conditions.
    *   Confirmation that memory footprint remains fixed after spawning.
*   **Training Loop:**
    *   End-to-end training on a toy task, verifying loss reduction.
    *   Integration of singularity spawning within the training process.

## 6. Project Structure

The project will be organized into a standard Python package structure:

```
holographic_brain_memory/
├── __init__.py
├── core.py             # Contains HolographicBrainMemory and PhaseBrainLayer
├── real_valued.py      # Contains real-valued HRR implementation
├── spawning.py         # Contains singularity spawning logic
├── visualization.py    # Contains visualization utilities
├── examples/           # Example scripts
│   └── toy_task.py
└── tests/
    ├── test_core.py
    ├── test_real_valued.py
    └── test_spawning.py
```

## 7. Future Considerations

*   **Cleanup Memory:** Implement a small auto-associative network for denoising retrieved vectors.
*   **Data-Dependent Theta:** Explore mechanisms for generating phase keys (`theta`) based on input data, similar to attention mechanisms.
*   **Hardware Deployment:** Investigate deployment on microcontrollers, edge devices, or in-memory computing hardware, leveraging the hardware-friendly nature of FFT and quantized representations.

This design document provides a roadmap for developing a robust and feature-rich Holographic Brain Memory library, addressing the user's request to transform the initial concept into a production-ready solution. The next steps will involve implementing these components and features, followed by thorough testing and documentation.
