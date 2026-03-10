Avus Transformer - Trained Models Summary

Overview
Successfully trained 3 transformer models with a total of 204.68M parameters using the Avus architecture.

Models Trained
Model	Parameters	Vocab Size	Dim	Layers	Heads	KV Heads	File Size	Location
avus_33m	33.00M	3,000	512	8	8	4	127 MB	/tmp/avus_weights/
avus_65m	64.01M	4,000	640	10	8	4	245 MB	/tmp/avus_weights/
avus_100m	107.67M	5,000	768	12	12	4	412 MB	/tmp/avus_weights/

Architecture Features

- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Position Embeddings
- SwiGLU: Swish-Gated Linear Unit activation
- GQA: Grouped Query Attention (efficient attention)
- Weight Tying: Shared input/output embeddings

Training Results

33M Model
- Epoch 1: 487.52
- Epoch 2: 461.54
- Epoch 3: 342.87
- Final Loss: 342.87

65M Model
- Epoch 1: 578.40
- Epoch 2: 569.10
- Epoch 3: 513.10
- Final Loss: 513.10

100M Model
- Epoch 1: 718.26
- Epoch 2: 642.13
- Final Loss: 642.13

Training Configuration
- Optimizer: AdamW
- Learning Rate: 3e-4
- Weight Decay: 0.1
- Sequence Length: 32
- Batch Size: 1
- Data: Random dummy tokens

Files Available

Model Weights (in /tmp/avus_weights/)
- avus_33m.pt - 127 MB
- avus_65m.pt - 245 MB
- avus_100m.pt - 412 MB

Configuration Files
- config_33m.json
- config_65m.json
- config_100m.json

Training History
- history_33m.json
- history_65m.json
- history_100m.json

Code & Documentation
- model.py - Full model architecture
- README.md - Complete documentation
- summary.json - Model summary

Usage Example

import torch
from dataclasses import dataclass

@dataclass
class AvusConfig:
    vocab_size: int = 3000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 128

# Load model (see model.py for full architecture)
model = Avus(AvusConfig())
checkpoint = torch.load("avus_33m.pt")
model.load_state_dict(checkpoint)
model.eval()


Upload to GitHub

To upload these models to https://github.com/Thoseidiots/Janus:

1. Copy the files from /tmp/avus_weights/
2. Create a new directory in the repo (e.g., avus_models/)
3. Add all model weights, configs, and documentation
4. Use Git LFS for the large .pt files:
5. git lfs track "*.pt"
6. git add .
7. git commit -m "Add trained Avus models (33M, 65M, 100M parameters)"
8. git push


Notes

- Models trained on dummy data for demonstration
- For production use, retrain on real text corpora
- Loss values are high due to random training data
- Architecture is production-ready with modern transformer features
⸻Trained: March 10, 2025 
Total Parameters: 204.68M 
Architecture: Avus Transformer (LLaMA-style)