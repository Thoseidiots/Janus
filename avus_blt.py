"""
avus_blt.py
===========
Byte Latent Transformer (BLT) variant of Avus.

Architecture (inspired by Meta's BLT paper, 2024):

  Raw bytes (0-255)
       │
  ┌────▼────────────────────────────────┐
  │  Local Encoder  (tiny transformer)  │  processes bytes → patches
  │  patch_size bytes → 1 patch vector  │
  └────────────────┬────────────────────┘
                   │  patch embeddings  (seq_len / patch_size)
  ┌────────────────▼────────────────────┐
  │  Global Model  (full Avus + MoE)    │  reasons over patches
  │  sliding window, GQA, MoE FFN       │
  └────────────────┬────────────────────┘
                   │  patch hidden states
  ┌────────────────▼────────────────────┐
  │  Local Decoder  (tiny transformer)  │  patch hidden → byte logits
  │  predicts next patch_size bytes     │
  └─────────────────────────────────────┘

Why this beats pure tokenization:
  - Zero OOV: handles any byte sequence (binary, JSON, code, images)
  - Learned patches: model discovers optimal groupings from data
  - Shorter global sequence: 8x fewer positions for the expensive model
  - No tokenizer to train or maintain

Config example (fits on T4 16GB):
    BLTConfig(
        patch_size=8,          # 8 bytes per patch
        local_dim=128,         # tiny local model
        local_layers=2,
        global_dim=1024,       # full Avus global model
        global_layers=12,
        global_heads=8,
        global_kv_heads=4,
        global_window=256,     # sliding window on patches
        global_n_experts=8,    # MoE
        global_n_experts_active=2,
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

# Import from avus — MoEFFN may not exist in older dataset snapshots
try:
    from avus import RMSNorm, SwiGLU, RoPE, AvusConfig, Avus, MoEFFN, AvusFFN
except ImportError:
    from avus import RMSNorm, SwiGLU, RoPE, AvusConfig, Avus, AvusFFN
    # MoEFFN not available — define a passthrough so BLT still works
    MoEFFN = AvusFFN

# ── BLT Config ────────────────────────────────────────────────────────────────

@dataclass
class BLTConfig:
    # ── Byte / patch settings ─────────────────────────────────────────────────
    patch_size:   int = 8      # bytes per patch (4-16 is typical)
    vocab_bytes:  int = 256    # always 256 — one per byte value

    # ── Local encoder/decoder (tiny, fast) ───────────────────────────────────
    local_dim:    int = 128
    local_layers: int = 2
    local_heads:  int = 4

    # ── Global model (full Avus) ──────────────────────────────────────────────
    global_dim:              int = 512
    global_layers:           int = 8
    global_heads:            int = 8
    global_kv_heads:         int = 4
    global_ffn_hidden:       Optional[int] = None
    global_window:           int = 256    # sliding window on patches
    global_n_experts:        int = 0      # 0 = dense, 8 = MoE
    global_n_experts_active: int = 2
    global_dropout:          float = 0.0
    eps:                     float = 1e-5
    max_patches:             int = 512    # max patches in one forward pass

    def to_avus_config(self) -> AvusConfig:
        """Build the AvusConfig for the global model — compatible with any avus.py version."""
        import inspect
        valid_fields = set(inspect.signature(AvusConfig.__init__).parameters.keys()) - {'self'}
        kwargs = dict(
            vocab_size        = self.global_dim,
            dim               = self.global_dim,
            n_layers          = self.global_layers,
            n_heads           = self.global_heads,
            n_kv_heads        = self.global_kv_heads,
            ffn_hidden        = self.global_ffn_hidden,
            max_seq_len       = self.max_patches,
            dropout           = self.global_dropout,
            eps               = self.eps,
            window_size       = self.global_window,
            n_experts         = self.global_n_experts,
            n_experts_active  = self.global_n_experts_active,
        )
        return AvusConfig(**{k: v for k, v in kwargs.items() if k in valid_fields})

    @classmethod
    def from_dict(cls, d: dict) -> "BLTConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ── Local Encoder ─────────────────────────────────────────────────────────────

class LocalByteEncoder(nn.Module):
    """
    Tiny transformer that reads patch_size raw bytes and produces one patch vector.

    Each patch is processed independently — no cross-patch attention here.
    The global model handles long-range dependencies.
    """
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.dim        = cfg.local_dim
        self.out_dim    = cfg.global_dim

        # Byte embedding: 256 possible byte values
        self.byte_emb = nn.Embedding(cfg.vocab_bytes, cfg.local_dim)

        # Positional embedding within a patch (learned, small)
        self.pos_emb  = nn.Embedding(cfg.patch_size, cfg.local_dim)

        # Small transformer layers
        self.layers = nn.ModuleList([
            _LocalBlock(cfg.local_dim, cfg.local_heads, cfg.eps)
            for _ in range(cfg.local_layers)
        ])
        self.norm = RMSNorm(cfg.local_dim, cfg.eps)

        # Project local dim → global dim
        self.proj = nn.Linear(cfg.local_dim, cfg.global_dim, bias=False)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: (B, N_patches, patch_size) — raw byte values 0-255
        Returns:
            patches: (B, N_patches, global_dim)
        """
        B, N, P = byte_ids.shape
        dev = byte_ids.device

        # Embed bytes + positions within patch
        pos = torch.arange(P, device=dev)
        x   = self.byte_emb(byte_ids) + self.pos_emb(pos)  # (B, N, P, local_dim)

        # Process each patch independently: reshape to (B*N, P, local_dim)
        x = x.view(B * N, P, self.dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Pool patch → single vector (mean pooling)
        x = x.mean(dim=1)           # (B*N, local_dim)
        x = x.view(B, N, self.dim)  # (B, N, local_dim)

        return self.proj(x)         # (B, N, global_dim)


class _LocalBlock(nn.Module):
    """Minimal transformer block for the local encoder/decoder."""
    def __init__(self, dim: int, n_heads: int, eps: float):
        super().__init__()
        self.ln1  = RMSNorm(dim, eps)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False)
        self.ln2  = RMSNorm(dim, eps)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


# ── Global Model ──────────────────────────────────────────────────────────────

class GlobalPatchModel(nn.Module):
    """
    Full Avus transformer operating on patch embeddings instead of token ids.

    Reuses all Avus components: GQA, RoPE, sliding window, MoE FFN.
    Input is patch vectors from LocalByteEncoder, not token embeddings.
    """
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        from avus import AvusBlock
        acfg = cfg.to_avus_config()
        self.config = acfg

        # No token embedding — input comes directly from local encoder
        self.dropout = nn.Dropout(acfg.dropout)
        self.blocks  = nn.ModuleList([AvusBlock(acfg) for _ in range(acfg.n_layers)])
        self.ln_f    = RMSNorm(acfg.dim, acfg.eps)

    def forward(
        self,
        patch_emb: torch.Tensor,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            patch_emb: (B, N_patches, global_dim)
        Returns:
            hidden: (B, N_patches, global_dim)
        """
        x = self.dropout(patch_emb)
        for block in self.blocks:
            x = block(x, use_cache=use_cache, cache_offset=cache_offset)
        return self.ln_f(x)

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    def moe_aux_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for block in self.blocks:
            if hasattr(block.ffn, "load_balancing_loss"):
                loss = loss + block.ffn.load_balancing_loss()
        return loss


# ── Local Decoder ─────────────────────────────────────────────────────────────

class LocalByteDecoder(nn.Module):
    """
    Takes a global patch hidden state and predicts the next patch_size bytes.

    Autoregressive within the patch: predicts byte[0], then byte[1] given byte[0], etc.
    """
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.dim        = cfg.local_dim
        self.vocab      = cfg.vocab_bytes

        # Project global hidden → local dim
        self.proj_in = nn.Linear(cfg.global_dim, cfg.local_dim, bias=False)

        # Byte embedding for teacher-forced decoding during training
        self.byte_emb = nn.Embedding(cfg.vocab_bytes, cfg.local_dim)
        self.pos_emb  = nn.Embedding(cfg.patch_size, cfg.local_dim)

        self.layers = nn.ModuleList([
            _LocalBlock(cfg.local_dim, cfg.local_heads if hasattr(cfg, 'local_heads') else 4, cfg.eps)
            for _ in range(cfg.local_layers)
        ])
        self.norm   = RMSNorm(cfg.local_dim, cfg.eps)
        self.head   = nn.Linear(cfg.local_dim, cfg.vocab_bytes, bias=False)

    def forward(
        self,
        global_hidden: torch.Tensor,
        target_bytes:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            global_hidden: (B, N_patches, global_dim)
            target_bytes:  (B, N_patches, patch_size) — for teacher forcing during training
        Returns:
            logits: (B, N_patches, patch_size, vocab_bytes)
            loss:   scalar cross-entropy loss (if target_bytes provided)
        """
        B, N, _ = global_hidden.shape
        dev = global_hidden.device

        # Project global hidden to local dim — one context vector per patch
        ctx = self.proj_in(global_hidden)  # (B, N, local_dim)

        pos = torch.arange(self.patch_size, device=dev)

        if target_bytes is not None:
            # Training: teacher-forced decoding
            # Shift target right: prepend zeros (start token) and drop last byte
            bos = torch.zeros(B, N, 1, dtype=torch.long, device=dev)
            inp = torch.cat([bos, target_bytes[:, :, :-1]], dim=2)  # (B, N, patch_size)

            x = self.byte_emb(inp) + self.pos_emb(pos)  # (B, N, P, local_dim)
            # Add global context to every byte position
            x = x + ctx.unsqueeze(2)

            x = x.view(B * N, self.patch_size, self.dim)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.head(x)                          # (B*N, P, vocab)
            logits = logits.view(B, N, self.patch_size, self.vocab)

            loss = F.cross_entropy(
                logits.view(-1, self.vocab),
                target_bytes.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            # Inference: autoregressive byte-by-byte within each patch
            generated = torch.zeros(B, N, self.patch_size, dtype=torch.long, device=dev)
            prev_byte = torch.zeros(B, N, dtype=torch.long, device=dev)  # BOS = 0

            for i in range(self.patch_size):
                x = self.byte_emb(prev_byte).unsqueeze(2)  # (B, N, 1, local_dim)
                x = x + self.pos_emb(torch.tensor([i], device=dev))
                x = x + ctx.unsqueeze(2)
                x = x.view(B * N, 1, self.dim)
                for layer in self.layers:
                    x = layer(x)
                x = self.norm(x)
                step_logits = self.head(x[:, 0, :])        # (B*N, vocab)
                next_byte   = step_logits.argmax(dim=-1).view(B, N)
                generated[:, :, i] = next_byte
                prev_byte = next_byte

            return generated, None


# ── ByteLatentAvus — Full Model ───────────────────────────────────────────────

class ByteLatentAvus(nn.Module):
    """
    Full Byte Latent Transformer.

    Processes raw bytes end-to-end:
      bytes → patches → global reasoning → byte predictions

    Drop-in replacement for Avus with a bytes-based interface.
    """

    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = LocalByteEncoder(cfg)
        self.global_ = GlobalPatchModel(cfg)
        self.decoder = LocalByteDecoder(cfg)

    def _bytes_to_patches(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat byte sequence into patches.
        Pads with zeros if not divisible by patch_size.

        Args:
            byte_ids: (B, T_bytes)
        Returns:
            (B, N_patches, patch_size)
        """
        B, T = byte_ids.shape
        P    = self.cfg.patch_size
        # Pad to multiple of patch_size
        pad  = (P - T % P) % P
        if pad > 0:
            byte_ids = F.pad(byte_ids, (0, pad), value=0)
        return byte_ids.view(B, -1, P)

    def forward(
        self,
        byte_ids: torch.Tensor,
        target_bytes: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            byte_ids:     (B, T_bytes) — raw byte values 0-255
            target_bytes: (B, T_bytes) — target bytes for loss computation
        Returns:
            logits: (B, N_patches, patch_size, 256)
            loss:   scalar (if target_bytes provided)
        """
        # Reshape bytes into patches
        patches_in = self._bytes_to_patches(byte_ids)  # (B, N, P)

        # Encode patches to global embeddings
        patch_emb = self.encoder(patches_in)            # (B, N, global_dim)

        # Global reasoning over patches
        hidden = self.global_(patch_emb, use_cache=use_cache,
                              cache_offset=cache_offset)  # (B, N, global_dim)

        # Decode back to bytes
        if target_bytes is not None:
            target_patches = self._bytes_to_patches(target_bytes)
            logits, loss = self.decoder(hidden, target_patches)
            # Add MoE load balancing loss
            moe_loss = self.global_.moe_aux_loss()
            if moe_loss.item() > 0:
                loss = loss + 0.01 * moe_loss.to(loss.device)
        else:
            logits, loss = self.decoder(hidden, None)

        return logits, loss

    def clear_cache(self):
        self.global_.clear_cache()

    @torch.no_grad()
    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Generate bytes autoregressively.

        Args:
            prompt_bytes: (B, T) raw byte tensor
            max_new_bytes: how many bytes to generate
        Returns:
            (B, T + max_new_bytes) byte tensor
        """
        self.eval()
        self.clear_cache()

        P   = self.cfg.patch_size
        out = prompt_bytes.clone()

        # Process prompt
        patches_in = self._bytes_to_patches(out)
        patch_emb  = self.encoder(patches_in)
        hidden     = self.global_(patch_emb, use_cache=True, cache_offset=0)

        n_generated = 0
        while n_generated < max_new_bytes:
            # Decode the last patch to get next bytes
            last_hidden = hidden[:, -1:, :]          # (B, 1, global_dim)
            gen_patch, _ = self.decoder(last_hidden)  # (B, 1, patch_size)
            new_bytes    = gen_patch[:, 0, :]         # (B, patch_size)

            # Apply temperature + top-k to the last byte of the patch
            # (for finer control; full patch is already generated greedily above)
            out = torch.cat([out, new_bytes], dim=1)
            n_generated += P

            # Encode the new patch and extend cache
            new_patch_emb = self.encoder(new_bytes.unsqueeze(1))  # (B, 1, global_dim)
            new_hidden    = self.global_(
                new_patch_emb, use_cache=True,
                cache_offset=hidden.shape[1]
            )
            hidden = torch.cat([hidden, new_hidden], dim=1)

        self.clear_cache()
        return out[:, :prompt_bytes.shape[1] + max_new_bytes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str):
        torch.save({
            "config": {
                "patch_size":              self.cfg.patch_size,
                "local_dim":               self.cfg.local_dim,
                "local_layers":            self.cfg.local_layers,
                "local_heads":             self.cfg.local_heads,
                "global_dim":              self.cfg.global_dim,
                "global_layers":           self.cfg.global_layers,
                "global_heads":            self.cfg.global_heads,
                "global_kv_heads":         self.cfg.global_kv_heads,
                "global_window":           self.cfg.global_window,
                "global_n_experts":        self.cfg.global_n_experts,
                "global_n_experts_active": self.cfg.global_n_experts_active,
                "max_patches":             self.cfg.max_patches,
            },
            "model_state_dict": self.state_dict(),
        }, path)
        print(f"ByteLatentAvus saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ByteLatentAvus":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = BLTConfig.from_dict(ckpt.get("config", {}))
        model = cls(cfg)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval()
        return model


# ── Text ↔ Bytes helpers ──────────────────────────────────────────────────────

def text_to_bytes(text: str) -> torch.Tensor:
    """Encode text to a byte tensor (UTF-8)."""
    b = text.encode("utf-8")
    return torch.tensor(list(b), dtype=torch.long).unsqueeze(0)  # (1, T)


def bytes_to_text(byte_tensor: torch.Tensor) -> str:
    """Decode a byte tensor back to text, ignoring invalid UTF-8."""
    b = byte_tensor.squeeze(0).tolist()
    return bytes(b).decode("utf-8", errors="replace")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Small config for smoke test
    cfg = BLTConfig(
        patch_size=8,
        local_dim=64,
        local_layers=2,
        local_heads=4,
        global_dim=256,
        global_layers=4,
        global_heads=4,
        global_kv_heads=2,
        global_window=64,
        global_n_experts=4,
        global_n_experts_active=2,
        max_patches=128,
    )

    model = ByteLatentAvus(cfg)
    n_params = model.count_parameters()
    print(f"ByteLatentAvus params: {n_params/1e6:.1f}M")

    # Encode a screen action example
    text = '<|startoftext|>Chrome is open. A \'Submit\' button is at (847,392). Click it. [ACT_START]{"type":"click","x":847,"y":392,"button":"left"}[ACT_END]<|endoftext|>'
    byte_ids = text_to_bytes(text)
    print(f"Input: {len(text)} chars → {byte_ids.shape[1]} bytes → "
          f"{byte_ids.shape[1] // cfg.patch_size} patches "
          f"(vs ~{len(text)//4} tokens with GPT-2)")

    # Forward pass
    logits, loss = model(byte_ids, target_bytes=byte_ids)
    print(f"Logits: {logits.shape}  Loss: {loss.item():.4f}")

    # Generation
    prompt = text_to_bytes("<|startoftext|>Chrome is open.")
    out    = model.generate(prompt, max_new_bytes=32)
    print(f"Generated: {bytes_to_text(out)!r}")
    print("ByteLatentAvus OK.")
