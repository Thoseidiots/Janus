# Janus

An autonomous AI system built from scratch. No API keys. No cloud dependencies. Everything runs locally.

## What Janus Is

Janus is a full-stack autonomous agent combining a custom transformer (Avus), persistent holographic memory, video comprehension, speech synthesis, and a tool execution layer. The goal is a system that can perceive, reason, remember, and act — continuously, without external services.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Janus System                    │
├─────────────────┬───────────────────────────────────┤
│   Perception    │  screen_interpreter.py            │
│                 │  video_observer.py                │
│                 │  janus_video_comprehension.py     │
│                 │  voice_io_enhanced.py             │
├─────────────────┼───────────────────────────────────┤
│   Reasoning     │  avus.py          (transformer)   │
│                 │  avus_inference.py                │
│                 │  avus_brain.py    (high-level API) │
├─────────────────┼───────────────────────────────────┤
│   Memory        │  holographic_brain_memory/        │
│                 │    core.py        (complex HRR)   │
│                 │    real_valued.py (real HRR)      │
│                 │    spawning.py    (dynamic growth) │
├─────────────────┼───────────────────────────────────┤
│   Action        │  updated_janus_capability_hub.py  │
│                 │  video_capability.py              │
│                 │  autonomy_capability.py           │
│                 │  browser_automation.py            │
├─────────────────┼───────────────────────────────────┤
│   Speech        │  speech_synthesis.py              │
│                 │  voice_io_enhanced.py             │
│                 │  janus_voip.py                    │
├─────────────────┼───────────────────────────────────┤
│   Training      │  train.py         (unified)       │
│                 │  avus_cognitive_dataset_generator_v3.py │
│                 │  language_dataset_generator_v2.py │
└─────────────────┴───────────────────────────────────┘
```

## Core Components

### Avus — The Transformer
Custom transformer architecture with RoPE, GQA, SwiGLU, and RMSNorm. Scales from 1B to 70B parameters.

| Model | Dim | Layers | Heads | KV Heads | Est. Params |
|-------|-----|--------|-------|----------|-------------|
| avus-1b | 768 | 12 | 12 | 4 | ~1B |
| avus-3b | 2048 | 24 | 16 | 8 | ~3B |
| avus-7b | 4096 | 32 | 32 | 8 | ~7B |
| avus-13b | 5120 | 40 | 40 | 8 | ~13B |
| avus-34b | 7168 | 48 | 56 | 8 | ~34B |
| avus-70b | 8192 | 80 | 64 | 8 | ~70B |

### Holographic Brain Memory (HBM)
Persistent memory using Holographic Reduced Representations. Stores knowledge in a fixed-size vector via circular convolution — survives between sessions, grows capacity dynamically via `SpawningBrain` without increasing physical memory.

### Speech Synthesis
Source-filter voice model implementing the full Human Speech Synthesis Diagnostic Framework — glottal source, formant filtering, prosody, micro-variations (jitter, shimmer, breathiness), emotional expression, and room acoustics. No external TTS API.

### Video Comprehension
Captures any window (browser, video player), runs motion detection, subtitle scanning, and scene analysis. Feeds observations into HBM via `AvusBrain`.

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Bootstrap the full system
python Janus-main/bootstrap_video_observer.py

# Train Avus (starts with 1B by default)
python Janus-main/train.py --model avus --size 1b --epochs 10

# Test speech synthesis
python Janus-main/speech_synthesis.py

# Test video comprehension
python Janus-main/janus_video_comprehension.py --watch "YouTube" --window --duration 120
```

## Training

```bash
# Train any model size
python train.py --model avus --size 7b --epochs 5 --resume

# Train holographic memory
python train.py --model hbm --epochs 20

# Train everything
python train.py --model all --size 1b --epochs 10
```

## Weight Merging

Three methods for combining trained checkpoints:

```bash
# SLERP: smooth blend between two checkpoints
python train.py --merge slerp --a weights_a.pt --b weights_b.pt --out merged.pt

# DARE: merge specialist models via task arithmetic
python train.py --merge dare --inputs base.pt specialist_3d.pt specialist_lang.pt --out merged.pt

# Model Soup: average N checkpoints
python train.py --merge soup --inputs epoch1.pt epoch2.pt epoch3.pt --out merged.pt
```

## Design Principles

- **No API keys** — everything runs locally on your hardware
- **No external LLMs** — Avus is our own model, built from scratch
- **Persistent memory** — HBM survives between sessions; standard transformers don't
- **Modular** — each component works standalone and composes cleanly
- **Scalable** — same codebase from 1B to 70B parameters

## Project Structure

```
Janus-main/
├── avus.py                          # Transformer architecture
├── avus_inference.py                # Inference wrapper
├── avus_brain.py                    # High-level reasoning API
├── avus_tokenizer.py                # Tokenizer
├── train.py                         # Unified training + weight merging
├── speech_synthesis.py              # Human speech synthesis engine
├── screen_interpreter.py            # Screen/pixel understanding
├── video_observer.py                # Video frame capture
├── janus_video_comprehension.py     # Video understanding pipeline
├── voice_io_enhanced.py             # Wake word + conversation loop
├── bootstrap_video_observer.py      # System startup
├── updated_janus_capability_hub.py  # Tool registry
├── holographic_brain_memory/        # HBM package
│   ├── core.py                      # Complex-valued HRR
│   ├── real_valued.py               # Real-valued HRR
│   ├── spawning.py                  # Dynamic neuron spawning
│   ├── visualization.py             # Memory trace visualization
│   ├── examples/toy_task.py
│   └── tests/test_core.py
├── config_avus_1b.json              # Model configs
├── config_avus_3b.json
├── config_avus_7b.json
├── config_avus_13b.json
├── config_avus_34b.json
└── config_avus_70b.json
```
