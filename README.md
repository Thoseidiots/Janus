# Janus: The Persistent Cognitive Entity

**Janus** is a novel autoregressive framework and persistent cognitive entity designed to unify multimodal understanding and generation. It represents a paradigm shift from ephemeral AI assistants to a continuous, bounded, and verifiable cognitive process with an externalized identity and a secure execution substrate.

This repository contains two primary integrated systems:
1.  **A Modular Rust Core**: A secure, persistent, and verifiable cognitive architecture built on the Moltbook principles.
2.  **A Unified Python Interface**: A multimodal, voice-enabled, and cross-device interface for interacting with the cognitive core.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
  - [Rust Cognitive Core](#rust-cognitive-core)
  - [Python Unified Interface](#python-unified-interface)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the Rust Core](#building-the-rust-core)
  - [Setting Up the Python Interface](#setting-up-the-python-interface)
- [Usage](#usage)
  - [Running the Janus CLI (Rust)](#running-the-janus-cli-rust)
  - [Running the Unified Interface (Python)](#running-the-unified-interface-python)
- [Core Concepts](#core-concepts)
  - [Moltbook Integration](#moltbook-integration)
  - [Autonomous Core (Janus Brain)](#autonomous-core-janus-brain)
- [Model Weights (Git LFS)](#model-weights-git-lfs)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Janus is not just an AI model; it is a framework for a continuous cognitive process. It integrates a custom transformer architecture with a unique cognitive loop (`OBSERVE → PLAN → PROPOSE → VERIFY → APPLY`) to create a system that maintains identity, memory, and purpose across sessions.

The project aims to address key limitations in current AI systems:
- **Ephemeral Nature**: By externalizing identity into an immutable contract (`identity_object.json`), Janus maintains a stable self-narrative.
- **Lack of Verifiability**: Actions are proposed by an LLM core but verified within a secure WASM sandbox (`janus-wasm`) before execution.
- **Siloed Interaction**: The unified interface bridges voice, text, and messaging platforms for seamless, cross-device presence.

## Architecture

Janus is built on a hybrid architecture combining a high-performance, safe core in Rust with a flexible, multimodal interface in Python.

### Rust Cognitive Core
The `crates` directory houses the modular Rust components that form the brain of Janus.

| Component | Description |
| :--- | :--- |
| **`janus-core`** | The authority layer. Manages the **Identity Contract**, Task Graph, and Event Log. It is the central source of truth for the system's state. |
| **`janus-wasm`** | The execution substrate. A secure WASM-hosted sandbox for running code with snapshot/restore capabilities, ensuring all actions are safe and verifiable. |
| **`janus-llm`** | The reasoning adapter. A swappable interface for integrating various LLM reasoning cores (e.g., GPT, Claude, local models). |
| **`janus-brain`** | The cognitive core. Implements Homeostasis, Hierarchical Memory, and Byte-level LLM dynamics. |
| **`janus-cli`** | The orchestration layer. Implements the primary cognitive loop: `OBSERVE → PLAN → PROPOSE → VERIFY → APPLY`. |

### Python Unified Interface
The Python-based interface (`janus_unified.py`, `voice_io_enhanced.py`, etc.) provides a rich, human-centric way to interact with the cognitive core. It features:

- **Voice I/O**: Always-on wake word detection and conversational loop using local Whisper.cpp for STT and Piper for TTS.
- **Messaging Bridges**: Unified server for SMS (Twilio), Telegram, and WhatsApp.
- **Dynamic Tool Generation**: On-the-fly creation and execution of Python tools from natural language descriptions.
- **State Synchronization**: Cross-device state sharing using CRDTs (Conflict-free Replicated Data Types) for consistency.

## Features

- **Persistent Identity**: An AI with a stable, externalized identity that persists across sessions.
- **Secure & Verifiable Execution**: All actions are vetted in a sandboxed environment before being applied.
- **Autonomous Cognitive Core**:
    - **Homeostasis Engine**: Maintains internal balance and drives goals based on valence states.
    - **Hierarchical Memory**: An episodic buffer with thematic mining for deep self-reflection.
    - **Sleep Engine**: Performs offline consolidation for memory stabilization.
- **Unified Multimodal Interface**: Seamless interaction via voice, text, and popular messaging platforms.
- **Zero API Keys (for Local Models)**: The voice interface uses fully local models for transcription and synthesis, ensuring privacy.

## Project Structure
```
Janus/
├── crates/                  # Rust modular workspace
│   ├── janus-core/          # Core authority and identity logic
│   ├── janus-wasm/          # WASM sandbox executor
│   ├── janus-llm/           # LLM integration adapters
│   ├── janus-brain/         # Cognitive functions (memory, homeostasis)
│   └── janus-cli/           # Main executable and cognitive loop
├── core/                    # Python core logic
├── janus_unified.py         # Main Python unified interface
├── voice_io_enhanced.py     # Voice input/output system
├── messaging_bridge.py      # Messaging platform bridges
├── tool_discovery.py        # Dynamic tool generation
├── state_sync_enhanced.py   # Cross-device state sync
├── Cargo.toml               # Rust project manifest
├── requirements_enhanced.txt# Python dependencies
└── README.md
```

## Getting Started

### Prerequisites
- **Rust**: Latest stable version.
- **Cargo**: Rust's package manager.
- **Python**: 3.9+ recommended.
- **Git LFS**: For managing large model weights.
- **Whisper.cpp**: For local speech-to-text.
- **Piper**: For local text-to-speech.

### Building the Rust Core
1. Clone the repository with Git LFS:
   ```bash
   git lfs install
   git clone https://github.com/Thoseidiots/Janus.git
   cd Janus
   ```
2. Build the project:
   ```bash
   cargo build --release
   ```

### Setting Up the Python Interface
1. Install Python dependencies:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

## Model Weights (Git LFS)

The model weights are managed using **Git LFS (Large File Storage)**. This ensures that the large `.pt` files are tracked properly without bloating the repository history.

### How to Manage the Weights
When you clone the repository, ensure you have Git LFS installed to automatically download the full `.pt` files:
```bash
git lfs install
git clone https://github.com/Thoseidiots/Janus.git
```

If you have already cloned the repository without LFS, you can pull the weights using:
```bash
git lfs pull
```

### Loading the Model
You can load the best-performing weights using the following logic:
```python
import torch
from core.config import JanusConfig
from core.model import JanusModel

ckpt = torch.load("weights/janus_best.pt", map_location="cpu")
config = JanusConfig(**ckpt["config"])
model = JanusModel(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Core Concepts

### Moltbook Integration
Janus is built upon the **Moltbook** architectural principles:
- **Externalized Identity**: The AI's identity lives in an immutable contract (`identity_object.json`), not in model weights.
- **Continuous State**: State persists across bounded cognition cycles, preventing "task death".
- **Memory as Narrative**: Experiences are curated and summarized into a stable self-narrative.

### Autonomous Core (Janus Brain)
The `janus-brain` crate implements the autonomous capabilities:
- **Homeostasis Engine**: Evolves internal valence states (pleasure, arousal, curiosity) to drive behavior.
- **Hierarchical Memory**: Buffers episodic experiences and mines them for themes.
- **Byte-level LLM**: A transformer operating directly on UTF-8 bytes.

## API Reference
The Python interface provides a programmatic API for interaction, voice output, and messaging.

## Contributing
We welcome contributions! Please follow the standard fork-and-pull-request workflow.

## License
This project is licensed under the terms specified by the repository owner.
Janus Model - Trained Weights and Parameters

Overview
This package contains trained weights and parameters for the Janus AI model, a GPT-2 style decoder-only transformer.

Repository
- Source: https://github.com/Thoseidiots/Janus.git
- Cloned and Trained: March 9, 2025

Models

1. Tiny Model (janus_tiny_trained.pt)
- Total Parameters: 232,192 (0.23M)
- Architecture: GPT-2 style transformer
- Configuration:
    - Block Size: 64
    - Vocab Size: 1,000
    - Layers: 2
    - Attention Heads: 2
    - Embedding Dimension: 64
    - Dropout: 0.1

Training Details:
- Epochs: 3
- Batch Size: 4
- Learning Rate: 0.0003
- Optimizer: AdamW
- Final Loss: 6.7595

2. Full Model (janus_full)
- Total Parameters: 49,330,176 (49.33M)
- Architecture: GPT-2 style transformer
- Configuration:
    - Block Size: 128
    - Vocab Size: 50,304
    - Layers: 6
    - Attention Heads: 6
    - Embedding Dimension: 384
    - Dropout: 0.1

Training Details:
- Epochs: 2
- Batch Size: 4
- Learning Rate: 0.0003
- Weight Decay: 0.1
- Optimizer: AdamW
- Final Loss: 10.1241

File Structure
janus_weights/
├── janus_tiny_trained.pt      # Trained tiny model weights
├── config.json                 # Model configuration
├── model_parameters.json       # Detailed parameter information
└── README.md                   # This file


Usage

Loading the Model
import torch
from core.config import JanusConfig
from core.model import JanusModel

# Load config
config = JanusConfig()
config.block_size = 64
config.vocab_size = 1000
config.n_layer = 2
config.n_head = 2
config.n_embd = 64

# Initialize model
model = JanusModel(config)

# Load trained weights
model.load_state_dict(torch.load('janus_tiny_trained.pt'))
model.eval()


Training from Scratch
import torch
import torch.optim as optim
from core.config import JanusConfig
from core.model import JanusModel

# Setup
config = JanusConfig()
model = JanusModel(config)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
model.train()
for epoch in range(epochs):
    # Your training code here
    pass


Model Architecture Details

Components
1. Token Embeddings (wte): Maps token IDs to embeddings
2. Position Embeddings (wpe): Adds positional information
3. Transformer Blocks:
    - Layer Normalization
    - Multi-Head Self-Attention
    - Feed-Forward Network (MLP)
4. Output Layer (lm_head): Projects to vocabulary

Layer Breakdown (Tiny Model)
- transformer.wte.weight: [1000, 64]
- transformer.wpe.weight: [64, 64]
- transformer.h.0.ln_1.weight/bias: [64]
- transformer.h.0.attn.c_attn.weight: [192, 64]
- transformer.h.0.attn.c_proj.weight: [64, 64]
- transformer.h.0.mlp.0.weight: [256, 64]
- transformer.h.0.mlp.2.weight: [64, 256]
- (and similar for layer 1)
- transformer.ln_f.weight/bias: [64]
- lm_head.weight: [1000, 64]

Notes
- Models were trained on CPU
- Training used dummy data for demonstration
- For production use, train on appropriate datasets
- Original repository includes additional features like vision processing and browser automation

License
See original repository: https://github.com/Thoseidiots/Janus
Avus Transformer - Trained Models

Overview
This repository contains trained weights for the Avus Transformer, a modern decoder-only language model with state-of-the-art architectural features.

Architecture Features

- RMSNorm: Root Mean Square Layer Normalization for faster training
- RoPE (Rotary Position Embeddings): Better relative position encoding
- SwiGLU Activation: Improved gradient flow over GELU/ReLU
- Grouped Query Attention (GQA): Efficient attention with fewer KV heads
- Weight Tying: Shared embeddings between input and output

Models
Model	Parameters	Vocab Size	Dim	Layers	Heads	KV Heads	File Size
avus_33m	33.00M	3,000	512	8	8	4	127 MB
avus_65m	64.01M	4,000	640	10	8	4	245 MB
avus_100m	107.67M	5,000	768	12	12	4	412 MB

Training Details

All models were trained with:
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
- Sequence Length: 32
- Batch Size: 1
- Dummy Data: Random tokens for demonstration

Training Losses
Model	Epoch 1	Epoch 2	Epoch 3	Final Loss
33M	487.52	461.54	342.87	342.87
65M	578.40	569.10	513.10	513.10
100M	718.26	642.13	-	642.13

Usage

Loading a Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass  
class AvusConfig:
    vocab_size: int = 3000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 128

# Load model architecture (see model.py for full implementation)
model = Avus(AvusConfig())

# Load trained weights
checkpoint = torch.load("avus_33m.pt")
model.load_state_dict(checkpoint)
model.eval()

# Generate text
prompt = torch.randint(0, 3000, (1, 10))  # Random prompt
output = model.generate(prompt, max_new_tokens=50)


Model Configuration

Each model has a corresponding config_*.json file:

{
  "vocab_size": 3000,
  "dim": 512,
  "n_layers": 8,
  "n_heads": 8,
  "n_kv_heads": 4,
  "max_seq_len": 128
}


File Structure

avus_weights/
├── avus_33m.pt          # 33M parameter model weights
├── avus_65m.pt          # 65M parameter model weights
├── avus_100m.pt         # 100M parameter model weights
├── config_33m.json      # 33M model configuration
├── config_65m.json      # 65M model configuration
├── config_100m.json     # 100M model configuration
├── history_33m.json     # 33M training history
├── history_65m.json     # 65M training history
├── history_100m.json    # 100M training history
├── summary.json         # Summary of all models
├── model.py             # Full model architecture
└── README.md            # This file


Parameter Count Formula

Total Parameters ≈ vocab_size × dim + n_layers × (
    4 × dim² +    # Attention Q, K, V, O projections
    12 × dim²     # FFN (8×dim for fc1, 4×dim for fc2)
)


Future Improvements

- Train on real text data (WikiText, C4, etc.)
- Implement learning rate scheduling
- Add gradient accumulation for larger effective batch sizes
- Use mixed precision training (FP16/BF16)
- Implement distributed training for even larger models

License

See original repository: https://github.com/Thoseidiots/Janus

Credits

- Architecture: Avus Transformer (based on LLaMA design principles)
- Training: Custom implementation
- Date: March 10, 2025

