# Janus: The Most Advanced AI in the World

Janus is a persistent, bounded, and verifiable cognitive entity. It represents a shift from ephemeral AI assistants to a continuous cognitive process with an externalized identity and a verifiable execution substrate.

## 🏗 Architecture

Janus is built on a modular Rust-based workspace, emphasizing security, persistence, and verifiability.

### Core Components

| Component | Description |
| :--- | :--- |
| **`janus-core`** | The authority layer. Manages the **Identity Contract**, Task Graph, and Event Log. |
| **`janus-wasm`** | The execution substrate. A secure WASM-hosted sandbox for running code with snapshot/restore capabilities. |
| **`janus-llm`** | The reasoning adapter. A swappable interface for integrating various LLM reasoning cores. |
| **`janus-brain`** | The cognitive core. Implements Homeostasis, Hierarchical Memory, and Byte-level LLM dynamics. |
| **`janus-cli`** | The orchestration layer. Implements the `OBSERVE → PLAN → PROPOSE → VERIFY → APPLY` loop. |

## 🦞 Moltbook Integration

Janus integrates the **Moltbook** architectural principles:
- **Externalized Identity**: The AI's identity lives in an immutable contract (`identity_object.json`), not in model weights.
- **Continuous State**: State persists across bounded cognition cycles, ensuring identity continuity without "task death."
- **Memory as Narrative**: Experiences are curated and summarized into a stable self-narrative.

## 🧠 Autonomous Core

Janus now features an **Autonomous Core** (housed in `janus-brain`) that implements:
- **Homeostasis Engine**: A recurrent core that evolves valence (pleasure, arousal, curiosity, etc.) over time.
- **Hierarchical Memory**: Episodic buffer with thematic mining for self-reflection.
- **Byte-level LLM**: A transformer operating directly on UTF-8 bytes, removing tokenization overhead.
- **Sleep Engine**: Offline consolidation for memory stabilization.

## 🚀 Getting Started

### Prerequisites
- Rust (latest stable)
- Cargo

### Building the Project
```bash
cargo build
```

### Running the Janus CLI
```bash
cargo run -p janus-cli
```

## 🛠 Development Workflow

Janus operates in a strict loop:
1.  **OBSERVE**: Capture inputs and environmental data.
2.  **PLAN**: Decompose goals into a structured task graph.
3.  **PROPOSE**: Generate actions and rationales via the LLM core.
4.  **VERIFY**: Validate actions within the WASM sandbox.
5.  **APPLY**: Execute verified actions and update the persistent state.

## 📜 License
This project is licensed under the terms of the repository owner.
