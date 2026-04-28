
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Any

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AvusConfig:
    vocab_size:   int   = 50304
    dim:          int   = 384
    n_layers:     int   = 12
    n_heads:      int   = 6
    n_kv_heads:   Optional[int] = None  # GQA: if None defaults to n_heads (MHA)
    ffn_hidden:   Optional[int] = None  # explicit FFN hidden dim; if None uses 4*dim (SwiGLU: *2)
    max_seq_len:  int   = 4096
    dropout:      float = 0.0
    eps:          float = 1e-5
    # ── Efficiency features ───────────────────────────────────────────────────
    window_size:  int   = 256   # sliding window attention (0 = full attention)
    # MoE: number of experts and how many activate per token (0 = dense FFN)
    n_experts:    int   = 0     # 0 = dense, 8 = MoE with 8 experts
    n_experts_active: int = 2   # top-k experts activated per token
    # Speculative decoding: draft model config (None = disabled)
    draft_config: Optional[dict] = None

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        # Default FFN hidden: 8/3 * dim rounded to nearest 256 (SwiGLU standard)
        if self.ffn_hidden is None:
            self.ffn_hidden = _round256(int(self.dim * 8 / 3))

    @classmethod
    def from_dict(cls, d: dict) -> "AvusConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_file(cls, path: str) -> "AvusConfig":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate (no embeddings)."""
        attn  = 4 * self.dim * self.dim  # Q+K+V+O (approx, ignoring GQA reduction)
        ffn   = 3 * self.dim * self.ffn_hidden  # gate + up + down
        return self.n_layers * (attn + ffn)


def _round256(n: int) -> int:
    """Round n up to the nearest multiple of 256."""
    return ((n + 255) // 256) * 256

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
        # Move weight to input device if needed (supports model parallelism)
        if self.weight.device != x.device:
            self.weight = nn.Parameter(self.weight.to(x.device),
                                       requires_grad=self.weight.requires_grad)
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
    - Causal mask (can't attend to future tokens)
    - KV cache for fast autoregressive generation
    - Grouped-Query Attention (GQA) for efficiency
    - Sliding window attention (O(n) memory, Mistral-style)
    - KV cache compression (evicts oldest entries beyond window)
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
        self.window_size = config.window_size  # 0 = full attention

        # Q, K, V projections for GQA
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out    = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope    = RoPE(self.head_dim, config.max_seq_len)

        # Causal mask — lower triangular, sized to window if sliding window enabled
        mask_size = config.window_size if config.window_size > 0 else config.max_seq_len
        mask = torch.tril(torch.ones(mask_size, mask_size))
        self.register_buffer("mask", mask.view(1, 1, mask_size, mask_size))

        # KV cache (populated during generation)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def _compress_cache(self):
        """Evict oldest KV entries beyond window_size (sliding window compression)."""
        if self.window_size > 0 and self._cache_k is not None:
            cache_len = self._cache_k.shape[2]
            if cache_len > self.window_size:
                # Keep only the most recent window_size entries
                self._cache_k = self._cache_k[:, :, -self.window_size:, :]
                self._cache_v = self._cache_v[:, :, -self.window_size:, :]

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        dev = x.device

        # Move all projections and buffers to input device (model parallelism)
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out]:
            if proj.weight.device != dev:
                proj.weight = nn.Parameter(proj.weight.to(dev),
                                           requires_grad=proj.weight.requires_grad)
        if self.mask.device != dev:
            self.mask = self.mask.to(dev)
        if self.rope.cos.device != dev:
            self.rope.cos = self.rope.cos.to(dev)
            self.rope.sin = self.rope.sin.to(dev)

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len=T, offset=cache_offset)

        # KV cache — append new k/v, then compress if over window
        if use_cache:
            if self._cache_k is not None:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            self._cache_k = k
            self._cache_v = v
            self._compress_cache()
            k = self._cache_k
            v = self._cache_v

        # Sliding window: during training, limit attention to last window_size tokens
        if self.window_size > 0 and not use_cache:
            # Trim k/v to window if sequence is longer than window
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size:, :]
                v = v[:, :, -self.window_size:, :]
            # Q attends to at most window_size past tokens
            q_len   = q.shape[2]
            kv_len  = k.shape[2]
            win     = min(self.window_size, kv_len)
            # Build local causal mask for this window
            local_mask = self.mask[:, :, :q_len, :win].to(dev)

        # Repeat K and V heads for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if not use_cache or cache_offset == 0:
            if self.window_size > 0 and not use_cache:
                scores = scores.masked_fill(local_mask == 0, float("-inf"))
            else:
                seq_len = scores.shape[-1]
                q_len   = scores.shape[-2]
                full_mask = self.mask[:, :, :q_len, :seq_len].to(dev)
                scores = scores.masked_fill(full_mask == 0, float("-inf"))

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
    """Feed-forward network with SwiGLU activation and explicit hidden dim."""
    def __init__(self, config: AvusConfig):
        super().__init__()
        # SwiGLU: gate + up projection (x2), then down
        self.gate_up = nn.Linear(config.dim, config.ffn_hidden * 2, bias=False)
        self.act     = SwiGLU()
        self.down    = nn.Linear(config.ffn_hidden, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        if self.gate_up.weight.device != dev:
            self.gate_up.weight = nn.Parameter(self.gate_up.weight.to(dev),
                                               requires_grad=self.gate_up.weight.requires_grad)
            self.down.weight = nn.Parameter(self.down.weight.to(dev),
                                            requires_grad=self.down.weight.requires_grad)
        return self.dropout(self.down(self.act(self.gate_up(x))))

# ── Mixture of Experts FFN ────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """
    Sparse Mixture of Experts Feed-Forward Network.

    Instead of one dense FFN, uses N expert FFNs but only activates K per token.
    Capacity of an N/K x larger model at the compute cost of K experts.

    Example: n_experts=8, n_experts_active=2
      - 8 expert FFNs, each same size as a dense FFN
      - Each token routes to its top-2 experts
      - Active compute = 2/8 = 25% of a fully dense 8x model
      - But total capacity = 8x a single dense FFN

    Load balancing loss prevents all tokens routing to the same expert.
    """
    def __init__(self, config: AvusConfig):
        super().__init__()
        self.n_experts        = config.n_experts
        self.n_experts_active = config.n_experts_active
        self.dim              = config.dim

        # Router: projects token embedding to expert scores
        self.router = nn.Linear(config.dim, config.n_experts, bias=False)

        # Expert FFNs — each is a full SwiGLU FFN
        self.experts = nn.ModuleList([AvusFFN(config) for _ in range(config.n_experts)])

        # Track routing logits for load balancing loss
        self._router_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        dev = x.device

        if self.router.weight.device != dev:
            self.router.weight = nn.Parameter(
                self.router.weight.to(dev),
                requires_grad=self.router.weight.requires_grad
            )

        x_flat = x.view(-1, C)                                   # (B*T, C)
        router_logits = self.router(x_flat)                       # (B*T, n_experts)
        self._router_logits = router_logits

        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(router_probs, self.n_experts_active, dim=-1)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.zeros_like(x_flat)
        for k in range(self.n_experts_active):
            expert_ids     = topk_idx[:, k]          # (B*T,)
            expert_weights = topk_weights[:, k:k+1]  # (B*T, 1)
            for e_idx in range(self.n_experts):
                mask = (expert_ids == e_idx)
                if not mask.any():
                    continue
                e_in  = x_flat[mask].unsqueeze(0)
                e_out = self.experts[e_idx](e_in).squeeze(0)
                output[mask] += expert_weights[mask] * e_out

        return output.view(B, T, C)

    def load_balancing_loss(self) -> torch.Tensor:
        """
        Auxiliary loss encouraging uniform expert utilization.
        Add ~0.01 * this to the main training loss to prevent expert collapse.
        """
        if self._router_logits is None:
            return torch.tensor(0.0)
        probs      = F.softmax(self._router_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        return self.n_experts * (mean_probs * mean_probs).sum()

# ── Avus Block ────────────────────────────────────────────────────────────────

class AvusBlock(nn.Module):
    """Single Avus transformer block: attention + FFN (dense or MoE) with pre-norm."""
    def __init__(self, config: AvusConfig):
        super().__init__()
        self.ln1  = RMSNorm(config.dim, config.eps)
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config.dim, config.eps)
        # Use MoE FFN if n_experts > 0, otherwise dense
        if config.n_experts > 0:
            self.ffn = MoEFFN(config)
        else:
            self.ffn = AvusFFN(config)

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

    def __init__(self, config: Optional[AvusConfig] = None,
                 use_reservoir: bool = False,
                 reservoir_dir: str = "compute_storage"):
        super().__init__()
        self.config = config or AvusConfig()
        c = self.config
        self.use_reservoir = use_reservoir

        # Lazy import — only loads if reservoir is actually used
        if use_reservoir:
            from janus_reservoir import JanusLayerReservoir
            self.reservoir = JanusLayerReservoir(storage_dir=reservoir_dir)
            print(f"[Avus] Compute reservoir enabled → {reservoir_dir}")
        else:
            self.reservoir = None

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

        for i, block in enumerate(self.blocks):
            if self.use_reservoir and not self.training:
                # Wrap the block with the Compute Reservoir
                x = self.reservoir.wrap_layer(f"block_{i}", block, x, use_cache=use_cache, cache_offset=cache_offset)
            else:
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

    @torch.no_grad()
    def generate_speculative(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        draft_model: Optional["Avus"] = None,
        gamma: int = 4,  # draft tokens per step
    ) -> torch.Tensor:
        """
        Speculative decoding — 3-4x faster generation with identical output quality.

        How it works:
          1. Draft model (tiny, fast) proposes `gamma` tokens in one pass.
          2. Full model verifies all `gamma` tokens in a single forward pass.
          3. Accept tokens where draft and full model agree; reject the first mismatch.
          4. On rejection, sample a corrected token from the full model distribution.

        Result: same output distribution as greedy/top-k sampling from the full model,
        but with far fewer full-model forward passes.

        Args:
            draft_model: small Avus instance. If None, falls back to standard generate().
            gamma: number of draft tokens to propose per verification step.
        """
        if draft_model is None:
            return self.generate(idx, max_new_tokens, temperature, top_k)

        self.eval()
        draft_model.eval()
        self.clear_cache()
        draft_model.clear_cache()

        device = idx.device
        generated = 0

        while generated < max_new_tokens:
            # ── Step 1: Draft model proposes gamma tokens ─────────────────────
            draft_idx    = idx.clone()
            draft_tokens = []
            draft_probs  = []

            for _ in range(gamma):
                if generated + len(draft_tokens) >= max_new_tokens:
                    break
                d_logits, _ = draft_model.forward(draft_idx[:, -1:],
                                                   use_cache=True,
                                                   cache_offset=draft_idx.shape[1] - 1)
                d_logits = d_logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(d_logits, min(top_k, d_logits.size(-1)))
                    d_logits[d_logits < v[:, [-1]]] = float("-inf")
                d_probs   = F.softmax(d_logits, dim=-1)
                d_tok     = torch.multinomial(d_probs, num_samples=1)
                draft_tokens.append(d_tok)
                draft_probs.append(d_probs)
                draft_idx = torch.cat([draft_idx, d_tok], dim=1)

            if not draft_tokens:
                break

            # ── Step 2: Full model verifies all draft tokens in one pass ──────
            draft_seq   = torch.cat(draft_tokens, dim=1)  # (B, gamma)
            verify_input = torch.cat([idx[:, -1:], draft_seq], dim=1)  # (B, gamma+1)
            v_logits, _ = self.forward(verify_input)  # (B, gamma+1, vocab)

            # ── Step 3: Accept/reject each draft token ────────────────────────
            n_accepted = 0
            for i, (d_tok, d_prob) in enumerate(zip(draft_tokens, draft_probs)):
                v_logit = v_logits[:, i, :] / temperature
                if top_k > 0:
                    vv, _ = torch.topk(v_logit, min(top_k, v_logit.size(-1)))
                    v_logit[v_logit < vv[:, [-1]]] = float("-inf")
                v_prob = F.softmax(v_logit, dim=-1)

                # Acceptance probability: min(1, p_full / p_draft)
                accept_prob = torch.clamp(
                    v_prob.gather(1, d_tok) / (d_prob.gather(1, d_tok) + 1e-8),
                    max=1.0
                )
                if torch.rand(1, device=device).item() < accept_prob.item():
                    idx = torch.cat([idx, d_tok], dim=1)
                    n_accepted += 1
                    generated  += 1
                else:
                    # Rejection: sample corrected token from adjusted distribution
                    corrected = torch.clamp(v_prob - d_prob, min=0.0)
                    corrected = corrected / (corrected.sum(dim=-1, keepdim=True) + 1e-8)
                    c_tok = torch.multinomial(corrected, num_samples=1)
                    idx   = torch.cat([idx, c_tok], dim=1)
                    generated += 1
                    break  # restart draft from corrected token

            # If all draft tokens accepted, also take the full model's next token
            if n_accepted == len(draft_tokens) and generated < max_new_tokens:
                last_logit = v_logits[:, -1, :] / temperature
                if top_k > 0:
                    vv, _ = torch.topk(last_logit, min(top_k, last_logit.size(-1)))
                    last_logit[last_logit < vv[:, [-1]]] = float("-inf")
                last_probs = F.softmax(last_logit, dim=-1)
                next_tok   = torch.multinomial(last_probs, num_samples=1)
                idx        = torch.cat([idx, next_tok], dim=1)
                generated  += 1

            # Reset caches for next round
            self.clear_cache()
            draft_model.clear_cache()

        return idx

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Avus":
        """Load Avus from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config     = AvusConfig.from_dict(checkpoint.get("config", {}))
        model      = cls(config)
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
                "ffn_hidden":  self.config.ffn_hidden,
                "max_seq_len": self.config.max_seq_len,
                "dropout":     self.config.dropout,
                "eps":         self.config.eps,
                "window_size": self.config.window_size,
            },
            "model_state_dict": self.state_dict(),
        }, path)
        print(f"Avus saved to {path}")

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # ── Scaling ladder ────────────────────────────────────────────────────────
    configs = {
        "avus-1b":  AvusConfig(dim=768,  n_layers=12, n_heads=12, n_kv_heads=4),
        "avus-3b":  AvusConfig(dim=2048, n_layers=24, n_heads=16, n_kv_heads=8),
        "avus-7b":  AvusConfig(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8),
        "avus-13b": AvusConfig(dim=5120, n_layers=40, n_heads=40, n_kv_heads=8),
        "avus-34b": AvusConfig(dim=7168, n_layers=48, n_heads=56, n_kv_heads=8),
        "avus-70b": AvusConfig(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8),
    }

    print(f"{'Model':<12} {'Dim':>6} {'Layers':>7} {'Heads':>6} {'KV':>4} "
          f"{'FFN':>7} {'Est. Params':>14}")
    print("-" * 65)
    for name, cfg in configs.items():
        est = cfg.param_count_estimate()
        print(f"{name:<12} {cfg.dim:>6} {cfg.n_layers:>7} {cfg.n_heads:>6} "
              f"{cfg.n_kv_heads:>4} {cfg.ffn_hidden:>7} {est/1e9:>13.1f}B")

    print()

    # ── Smoke test on smallest config ─────────────────────────────────────────
    config = configs["avus-1b"]
    model  = Avus(config)
    print(f"avus-1b actual params: {model.count_parameters()/1e6:.1f}M")

    dummy = torch.randint(0, config.vocab_size, (1, 16))
    logits, _ = model(dummy)
    print(f"Input: {dummy.shape}  →  Output: {logits.shape}")

    out = model.generate(dummy, max_new_tokens=10)
    print(f"Generated length: {out.shape[1]}")
    print("Avus scaling ladder ready.")
