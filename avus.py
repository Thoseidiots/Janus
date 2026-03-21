
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AvusConfig:
    vocab_size:  int   = 50304
    dim:         int   = 384     # optimal Avus-v1 dimension
    n_layers:    int   = 12      # Increased layers for enhanced capacity
    n_heads:     int   = 6       # matches trained weights
    n_kv_heads:  Optional[int] = None # Number of KV heads for GQA. If None, defaults to n_heads (MHA).
    max_seq_len: int   = 1024    # maximum context window
    dropout:     float = 0.1
    eps:         float = 1e-6

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster and simpler than LayerNorm — no mean subtraction.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

# ── SwiGLU ───────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Better gradient flow than ReLU or GELU.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# ── RoPE Positional Encoding ──────────────────────────────────────────────────

class RoPE(nn.Module):
    """
    Rotary Position Embedding.

    Encodes position by rotating query/key vectors.
    Advantages:
      - Works on any sequence length (even longer than trained on)
      - Relative positions naturally encoded
      - No extra parameters
    """
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        # Precompute rotation frequencies
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, theta)
        # Store as cos/sin cache
        self.register_buffer("cos", freqs.cos()[None, None, :, :])  # (1,1,T,dim/2)
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:, :, offset:offset + seq_len, :]
        sin = self.sin[:, :, offset:offset + seq_len, :]
        
        # Duplicate cos/sin to match the full head_dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

# ── Causal Self-Attention with KV Cache and Grouped-Query Attention ───────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
    - RoPE positional encoding
    - Causal mask (can’t attend to future tokens)
    - KV cache for fast autoregressive generation
    - Grouped-Query Attention (GQA) for efficiency
    """
    def __init__(self, config: AvusConfig):
        super().__init__()
        assert config.dim % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.dim // config.n_heads
        self.dim        = config.dim
        self.num_groups = self.n_heads // self.n_kv_heads

        # Q, K, V projections for GQA
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out    = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope    = RoPE(self.head_dim, config.max_seq_len)

        # Causal mask — lower triangular
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

        # KV cache (populated during generation)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len=T, offset=cache_offset)

        # KV cache — append new k/v to cached k/v during generation
        if use_cache:
            if self._cache_k is not None:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            self._cache_k = k
            self._cache_v = v

        # Repeat K and V heads for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask (only during training / full forward pass)
        if not use_cache or cache_offset == 0:
            seq_len = scores.shape[-1]
            scores  = scores.masked_fill(
                self.mask[:, :, :T, :seq_len] == 0,
                float("-inf")
            )

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        output = torch.matmul(attn, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(output)

    def clear_cache(self):
        self._cache_k = None
        self._cache_v = None

# ── Feed-Forward Network ──────────────────────────────────────────────────────

class AvusFFN(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, config: AvusConfig):
        super().__init__()
        self.fc1     = nn.Linear(config.dim, 4 * config.dim * 2, bias=False)  # *2 for SwiGLU gate
        self.act     = SwiGLU()
        self.fc2     = nn.Linear(4 * config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))

# ── Avus Block ────────────────────────────────────────────────────────────────

class AvusBlock(nn.Module):
    """Single Avus transformer block: attention + FFN with pre-norm."""
    def __init__(self, config: AvusConfig):
        super().__init__()
        self.ln1  = RMSNorm(config.dim, config.eps)
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config.dim, config.eps)
        self.ffn  = AvusFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        # Pre-norm attention (normalize before attention, not after)
        normed = self.ln1(x)
        x = x + self.attn(normed, use_cache=use_cache, cache_offset=cache_offset)
        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))
        return x

    def clear_cache(self):
        self.attn.clear_cache()

# ── Avus — Main Model ─────────────────────────────────────────────────────────

class Avus(nn.Module):
    """
    Avus transformer — Avus's own architecture.

    Custom-built high-performance transformer.
    """

    def __init__(self, config: Optional[AvusConfig] = None):
        super().__init__()
        self.config = config or AvusConfig()
        c = self.config

        self.tok_emb = nn.Embedding(c.vocab_size, c.dim)
        self.dropout = nn.Dropout(c.dropout)
        self.blocks  = nn.ModuleList([AvusBlock(c) for _ in range(c.n_layers)])
        self.ln_f    = RMSNorm(c.dim, c.eps)
        self.head    = nn.Linear(c.dim, c.vocab_size, bias=False)

        # Weight tying — share token embedding and output weights
        # Reduces parameters and improves training
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        x = self.dropout(self.tok_emb(idx))

        for block in self.blocks:
            x = block(x, use_cache=use_cache, cache_offset=cache_offset)

        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.

        Args:
            idx: (B, T) tensor of prompt token ids
            max_new_tokens: how many tokens to generate
            temperature: higher = more random, lower = more focused
            top_k: only sample from top K most likely tokens
        """
        self.eval()
        self.clear_cache()

        # Process prompt (cache all prompt k/v)
        _, _ = self.forward(idx, use_cache=True, cache_offset=0)
        prompt_len = idx.shape[1]

        for step in range(max_new_tokens):
            # Only feed the last token (rest is in cache)
            last = idx[:, -1:]
            logits, _ = self.forward(
                last,
                use_cache=True,
                cache_offset=prompt_len + step,
            )
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)

        self.clear_cache()
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Avus":
        """Load Avus from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config     = AvusConfig(**checkpoint.get("config", {}))
        model      = cls(config)
        # Handle both raw state_dict and wrapped checkpoints
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def save_checkpoint(self, path: str):
        """Save Avus checkpoint."""
        torch.save({
            "config": {
                "vocab_size":  self.config.vocab_size,
                "dim":         self.config.dim,
                "n_layers":    self.config.n_layers,
                "n_heads":     self.config.n_heads,
                "n_kv_heads":  self.config.n_kv_heads,
                "max_seq_len": self.config.max_seq_len,
                "dropout":     self.config.dropout,
                "eps":         self.config.eps,
            },
            "model_state_dict": self.state_dict(),
        }, path)
        print(f"Avus saved to {path}")

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = AvusConfig(
        vocab_size  = 50304,
        dim         = 384,
        n_layers    = 6,
        n_heads     = 6,
        n_kv_heads  = 3, # Example: 3 KV heads for GQA
        max_seq_len = 1024,
    )

    model = Avus(config)
    print(f"Avus initialized.")
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randint(0, config.vocab_size, (1, 16))
    logits, _   = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated sequence length: {output.shape[1]}")
    print("Avus is working.")
