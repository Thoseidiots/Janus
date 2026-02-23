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
