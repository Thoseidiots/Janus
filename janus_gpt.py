# janus_gpt.py

“””
JanusGPT: GPT-2-style decoder-only transformer that exactly matches
the architecture Manus trained (janus-v1).

Config from janus_training_summary.json:
vocab_size : 50304
block_size : 128
n_layer    : 6
n_head     : 6
n_embd     : 384
dropout    : 0.1

Weights live in my-llm-project/weights/janus_best.pt
Uses tiktoken (GPT-2 tokenizer) to match the training tokenizer.
Falls back to a simple character tokenizer if tiktoken is unavailable.
“””

from **future** import annotations

import math
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class JanusConfig:
block_size : int   = 128
vocab_size : int   = 50304
n_layer    : int   = 6
n_head     : int   = 6
n_embd     : int   = 384
dropout    : float = 0.1
model_type : str   = “janus-v1”

```
@classmethod
def from_summary(cls, path: str) -> "JanusConfig":
    with open(path) as f:
        data = json.load(f)
    cfg = data.get("config", {})
    return cls(
        block_size = cfg.get("block_size", 128),
        vocab_size = cfg.get("vocab_size", 50304),
        n_layer    = cfg.get("n_layer",    6),
        n_head     = cfg.get("n_head",     6),
        n_embd     = cfg.get("n_embd",     384),
        dropout    = cfg.get("dropout",    0.1),
        model_type = cfg.get("model_type", "janus-v1"),
    )
```

# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class JanusTokenizer:
“””
Uses tiktoken (GPT-2 encoding) to match training tokenizer.
Falls back to simple ASCII encoding if tiktoken not installed.
“””
def **init**(self):
self._enc = None
try:
import tiktoken
self._enc = tiktoken.get_encoding(“gpt2”)
except ImportError:
pass

```
def encode(self, text: str) -> list[int]:
    if self._enc is not None:
        return self._enc.encode(text)
    # Fallback: ASCII byte encoding mapped into 0-255
    return [min(b, 255) for b in text.encode("utf-8")]

def decode(self, tokens: list[int]) -> str:
    if self._enc is not None:
        # Filter out any out-of-range tokens
        valid = [t for t in tokens if 0 <= t < 50257]
        try:
            return self._enc.decode(valid)
        except Exception:
            return ""
    return bytes([t for t in tokens if t < 256]).decode("utf-8", errors="replace")
```

# ── Model components ──────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
def **init**(self, config: JanusConfig):
super().**init**()
assert config.n_embd % config.n_head == 0
self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd)
self.c_proj  = nn.Linear(config.n_embd, config.n_embd)
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
self.n_head  = config.n_head
self.n_embd  = config.n_embd
self.register_buffer(
“bias”,
torch.tril(torch.ones(config.block_size, config.block_size))
.view(1, 1, config.block_size, config.block_size)
)

```
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))
```

class MLP(nn.Module):
def **init**(self, config: JanusConfig):
super().**init**()
# Matches Manus training: Sequential with Linear(embd, 4*embd), GELU, Linear(4*embd, embd)
self.mlp = nn.Sequential(
nn.Linear(config.n_embd, 4 * config.n_embd),
nn.GELU(),
nn.Linear(4 * config.n_embd, config.n_embd),
nn.Dropout(config.dropout),
)

```
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(x)
```

class Block(nn.Module):
def **init**(self, config: JanusConfig):
super().**init**()
self.ln_1 = nn.LayerNorm(config.n_embd)
self.attn = CausalSelfAttention(config)
self.ln_2 = nn.LayerNorm(config.n_embd)
self.mlp  = MLP(config)

```
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

# ── JanusGPT ──────────────────────────────────────────────────────────────────

class JanusGPT(nn.Module):
“””
GPT-2-style model matching janus-v1 training architecture exactly.
Layer names mirror the checkpoint keys so torch.load() works directly.
“””

```
def __init__(self, config: JanusConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte  = nn.Embedding(config.vocab_size, config.n_embd),
        wpe  = nn.Embedding(config.block_size, config.n_embd),
        drop = nn.Dropout(config.dropout),
        h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # Weight tying
    self.transformer.wte.weight = self.lm_head.weight

    self._tokenizer = JanusTokenizer()

@classmethod
def from_checkpoint(cls, weights_path: str,
                    summary_path: Optional[str] = None,
                    device: str = "cpu") -> "JanusGPT":
    """
    Load a trained checkpoint.  Automatically reads config from
    janus_training_summary.json if summary_path provided.
    """
    config = JanusConfig()
    if summary_path and os.path.exists(summary_path):
        config = JanusConfig.from_summary(summary_path)

    model = cls(config)

    checkpoint = torch.load(weights_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        state = (checkpoint.get("model_state_dict")
                 or checkpoint.get("model")
                 or checkpoint.get("state_dict")
                 or checkpoint)
    else:
        state = checkpoint

    # Strip 'module.' prefix if trained with DataParallel
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[JanusGPT] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[JanusGPT] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.to(device)
    model.eval()
    print(f"[JanusGPT] Loaded from {weights_path} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    return model

def forward(self, idx: torch.Tensor,
            targets: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, T = idx.shape
    assert T <= self.config.block_size, \
        f"Sequence length {T} exceeds block_size {self.config.block_size}"

    pos = torch.arange(T, device=idx.device)
    x = self.transformer.drop(
        self.transformer.wte(idx) + self.transformer.wpe(pos)
    )
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
    return logits, loss

@torch.no_grad()
def generate(self, prompt: str, max_new: int = 200,
             temperature: float = 0.8, top_k: int = 50) -> str:
    """Generate text continuing from prompt."""
    self.eval()
    ids = self._tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new):
        # Crop to block_size
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    generated_ids = idx[0, len(ids):].tolist()
    return self._tokenizer.decode(generated_ids)

def get_embedding(self, text: str) -> torch.Tensor:
    """
    Returns a pooled embedding vector for text.
    Used by HomeostasisEngine as stimulus input.
    """
    ids = self._tokenizer.encode(text)[:self.config.block_size]
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        pos = torch.arange(len(ids), device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
    # Mean pool → shape (n_embd,) = (384,)
    return x[0].mean(dim=0)
```

# ── Weight loader utility ─────────────────────────────────────────────────────

def load_janus_brain(
weights_dir: str = “my-llm-project/weights”,
prefer_best: bool = True,
device: str = “cpu”,
) -> JanusGPT:
“””
Convenience function: loads janus_best.pt or janus_final.pt
from the weights directory, reading config from training summary.

```
Usage:
    from janus_gpt import load_janus_brain
    llm = load_janus_brain()
    print(llm.generate("What is memory management?"))
"""
fname    = "janus_best.pt" if prefer_best else "janus_final.pt"
weights  = os.path.join(weights_dir, fname)
summary  = os.path.join(weights_dir, "janus_training_summary.json")

if not os.path.exists(weights):
    raise FileNotFoundError(
        f"Weights not found at '{weights}'. "
        f"Clone the repo and ensure my-llm-project/weights/ exists."
    )

return JanusGPT.from_checkpoint(weights, summary, device=device)
```