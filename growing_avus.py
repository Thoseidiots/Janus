"""
growing_avus.py — GrowingAvus: a transformer that grows its own architecture.

Starts small (seed_dim, seed_layers) and spawns new layers or expands width
when existing layers saturate during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

from avus import AvusConfig, AvusBlock, RMSNorm, _round256


# ── Health & Config dataclasses ───────────────────────────────────────────────

@dataclass
class LayerHealth:
    """Tracks the health/saturation state of a single transformer block."""
    grad_magnitude: float = 0.0       # running EMA of gradient norm
    activation_variance: float = 0.0  # variance of output activations
    loss_contribution: float = 0.0    # how much this layer reduces loss
    age_steps: int = 0                # training steps this layer has existed
    is_frozen: bool = False           # frozen layers live on CPU
    spawn_count: int = 0              # how many children this layer has spawned


@dataclass
class GrowthConfig:
    """Configuration for the growing behaviour of GrowingAvus."""
    # Seed architecture
    seed_dim: int = 256
    seed_layers: int = 2
    seed_heads: int = 4
    seed_kv_heads: int = 2

    # Hard ceilings
    max_dim: int = 8192
    max_layers: int = 80

    # Vocabulary / sequence
    vocab_size: int = 50304
    max_seq_len: int = 4096
    eps: float = 1e-5

    # Saturation thresholds
    grad_saturation_threshold: float = 0.001
    activation_collapse_threshold: float = 0.01

    # Age gates
    spawn_after_steps: int = 100
    freeze_after_steps: int = 500

    # VRAM budget
    vram_budget_gb: float = 14.0

    # Growth parameters
    dim_growth_factor: float = 1.5
    layer_growth_mode: str = "append"   # append | insert_after_weakest | both


# ── GrowingAvus ───────────────────────────────────────────────────────────────

class GrowingAvus(nn.Module):
    """
    A transformer that grows its own architecture during training.

    Starts with a small seed model and spawns new layers (or expands width)
    whenever existing layers become saturated.
    """

    def __init__(self, growth_config: GrowthConfig = None):
        super().__init__()
        self.growth_config = growth_config or GrowthConfig()
        gc = self.growth_config

        # Build initial AvusConfig from seed parameters
        self.avus_config = AvusConfig(
            vocab_size=gc.vocab_size,
            dim=gc.seed_dim,
            n_layers=gc.seed_layers,
            n_heads=gc.seed_heads,
            n_kv_heads=gc.seed_kv_heads,
            max_seq_len=gc.max_seq_len,
            eps=gc.eps,
        )

        # Core modules
        self.tok_emb = nn.Embedding(gc.vocab_size, gc.seed_dim)
        self.blocks  = nn.ModuleList(
            [AvusBlock(self.avus_config) for _ in range(gc.seed_layers)]
        )
        self.ln_f = RMSNorm(gc.seed_dim, gc.eps)
        self.head = nn.Linear(gc.seed_dim, gc.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Health tracking (one entry per block, kept in sync)
        self._health: List[LayerHealth] = [LayerHealth() for _ in self.blocks]

        # Growth history
        self._growth_log: List[dict] = []

        # Training step counter
        self._step: int = 0

        # Device tracking
        self._device: str = "cpu"

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_offset: int = 0,
    ):
        B, T = idx.shape
        x = self.tok_emb(idx)

        for i, block in enumerate(self.blocks):
            if self._health[i].is_frozen:
                # Frozen blocks live on CPU — run there, bring result back
                cpu_x = x.cpu()
                cpu_block = block  # already on CPU
                cpu_out = cpu_block(cpu_x, use_cache=use_cache, cache_offset=cache_offset)
                x = cpu_out.to(x.device)
            else:
                x = block(x, use_cache=use_cache, cache_offset=cache_offset)

            # Record activation variance for health tracking (no grad overhead)
            with torch.no_grad():
                self._health[i].activation_variance = float(x.var().item())

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Gradient recording ────────────────────────────────────────────────────

    def record_gradients(self):
        """
        Call after loss.backward() but before optimizer.step().
        Updates each layer's grad_magnitude EMA and increments age_steps.
        """
        alpha = 0.1
        for i, block in enumerate(self.blocks):
            if self._health[i].is_frozen:
                self._health[i].age_steps += 1
                continue

            total_norm = 0.0
            n_params = 0
            for p in block.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    n_params += 1

            grad_norm = math.sqrt(total_norm) if n_params > 0 else 0.0

            # Exponential moving average
            h = self._health[i]
            h.grad_magnitude = (1 - alpha) * h.grad_magnitude + alpha * grad_norm
            h.age_steps += 1

        self._step += 1

    # ── Growth logic ──────────────────────────────────────────────────────────

    def check_and_grow(self) -> bool:
        """
        Inspect layer health and trigger growth if needed.
        Returns True if any growth event occurred.
        """
        gc = self.growth_config
        grew = False

        saturated_indices = []
        for i, h in enumerate(self._health):
            if h.is_frozen:
                continue
            is_saturated = (
                h.grad_magnitude < gc.grad_saturation_threshold
                and h.age_steps > gc.spawn_after_steps
            )
            is_collapsed = h.activation_variance < gc.activation_collapse_threshold

            if is_saturated or is_collapsed:
                saturated_indices.append(i)

        if saturated_indices:
            # Decide where to spawn based on growth mode
            if gc.layer_growth_mode == "append":
                spawn_at = len(self.blocks) - 1
            elif gc.layer_growth_mode == "insert_after_weakest":
                spawn_at = saturated_indices[0]
            else:  # both — append first, then insert after weakest
                spawn_at = len(self.blocks) - 1

            if len(self.blocks) < gc.max_layers:
                self._spawn_layer(spawn_at)
                grew = True

            # If ALL non-frozen layers are saturated, also grow wider
            active = [i for i, h in enumerate(self._health) if not h.is_frozen]
            all_saturated = all(i in saturated_indices for i in active)
            if all_saturated and self.avus_config.dim < gc.max_dim:
                self._grow_wider()
                grew = True

        # Opportunistically freeze old saturated layers
        self._maybe_freeze_old_layers()

        return grew

    # ── Spawn a new layer ─────────────────────────────────────────────────────

    def _spawn_layer(self, after_idx: int):
        """Insert a new near-identity AvusBlock after position after_idx."""
        new_block = AvusBlock(self.avus_config)

        # Near-zero init so the new layer starts as approximately identity
        with torch.no_grad():
            for p in new_block.parameters():
                nn.init.normal_(p, mean=0.0, std=0.01)

        # Move to same device as existing blocks
        device = next(self.blocks[0].parameters()).device
        new_block = new_block.to(device)

        # Insert into ModuleList (rebuild to preserve nn.Module registration)
        blocks_list = list(self.blocks)
        insert_pos = after_idx + 1
        blocks_list.insert(insert_pos, new_block)
        self.blocks = nn.ModuleList(blocks_list)

        # Insert corresponding health entry
        self._health.insert(insert_pos, LayerHealth())

        # Mark parent as having spawned
        if after_idx < len(self._health):
            self._health[after_idx].spawn_count += 1

        event = {
            "event": "spawn_layer",
            "step": self._step,
            "after_idx": after_idx,
            "insert_pos": insert_pos,
            "total_layers": len(self.blocks),
            "dim": self.avus_config.dim,
        }
        self._growth_log.append(event)
        print(f"[GrowingAvus] Spawned layer {insert_pos} (total: {len(self.blocks)})")

    # ── Grow wider ────────────────────────────────────────────────────────────

    def _grow_wider(self):
        """
        Increase the model dimension.
        Existing weights are padded with near-zero values; new blocks are
        created with the new config and the old weights are copied in.
        """
        gc = self.growth_config
        old_dim = self.avus_config.dim
        raw_new = int(old_dim * gc.dim_growth_factor)
        # Round to nearest 64
        new_dim = min(((raw_new + 63) // 64) * 64, gc.max_dim)

        if new_dim <= old_dim:
            return

        # Scale heads proportionally (keep head_dim constant if possible)
        head_dim = old_dim // self.avus_config.n_heads
        new_heads = max(self.avus_config.n_heads, new_dim // head_dim)
        # Keep kv_heads ratio
        kv_ratio = self.avus_config.n_kv_heads / self.avus_config.n_heads
        new_kv_heads = max(1, round(new_heads * kv_ratio))

        new_config = AvusConfig(
            vocab_size=gc.vocab_size,
            dim=new_dim,
            n_layers=len(self.blocks),
            n_heads=new_heads,
            n_kv_heads=new_kv_heads,
            max_seq_len=gc.max_seq_len,
            eps=gc.eps,
        )

        device = next(self.tok_emb.parameters()).device

        # ── Expand tok_emb ────────────────────────────────────────────────────
        new_tok_emb = nn.Embedding(gc.vocab_size, new_dim)
        with torch.no_grad():
            nn.init.normal_(new_tok_emb.weight, mean=0.0, std=0.001)
            new_tok_emb.weight[:, :old_dim] = self.tok_emb.weight.data
        new_tok_emb = new_tok_emb.to(device)

        # ── Expand ln_f ───────────────────────────────────────────────────────
        new_ln_f = RMSNorm(new_dim, gc.eps).to(device)
        with torch.no_grad():
            new_ln_f.weight[:old_dim] = self.ln_f.weight.data
            new_ln_f.weight[old_dim:] = 1.0  # ones for new dims

        # ── Expand each block ─────────────────────────────────────────────────
        new_blocks = nn.ModuleList()
        for i, old_block in enumerate(self.blocks):
            nb = AvusBlock(new_config)
            with torch.no_grad():
                self._expand_block_weights(old_block, nb, old_dim, new_dim, new_config)
            if self._health[i].is_frozen:
                nb = nb.cpu()
            else:
                nb = nb.to(device)
            new_blocks.append(nb)

        # ── Swap everything in ────────────────────────────────────────────────
        self.tok_emb = new_tok_emb
        self.blocks  = new_blocks
        self.ln_f    = new_ln_f

        # New head (no bias) — re-tie weights
        new_head = nn.Linear(new_dim, gc.vocab_size, bias=False).to(device)
        with torch.no_grad():
            nn.init.normal_(new_head.weight, mean=0.0, std=0.001)
            new_head.weight[:, :old_dim] = self.head.weight.data
        self.head = new_head
        self.head.weight = self.tok_emb.weight  # re-tie

        self.avus_config = new_config

        event = {
            "event": "grow_wider",
            "step": self._step,
            "old_dim": old_dim,
            "new_dim": new_dim,
            "total_layers": len(self.blocks),
        }
        self._growth_log.append(event)
        print(f"[GrowingAvus] Grew wider: dim {old_dim} -> {new_dim}")

    def _expand_block_weights(
        self,
        old_block: AvusBlock,
        new_block: AvusBlock,
        old_dim: int,
        new_dim: int,
        new_config: AvusConfig,
    ):
        """Copy-and-pad weights from old_block into new_block."""
        # RMSNorm weights
        new_block.ln1.weight[:old_dim] = old_block.ln1.weight.data
        new_block.ln1.weight[old_dim:] = 1.0
        new_block.ln2.weight[:old_dim] = old_block.ln2.weight.data
        new_block.ln2.weight[old_dim:] = 1.0

        # Attention projections
        attn_old = old_block.attn
        attn_new = new_block.attn

        # q_proj: (n_heads*head_dim, dim) -> (new_heads*head_dim, new_dim)
        self._pad_linear(attn_old.q_proj, attn_new.q_proj)
        self._pad_linear(attn_old.k_proj, attn_new.k_proj)
        self._pad_linear(attn_old.v_proj, attn_new.v_proj)
        self._pad_linear(attn_old.out,    attn_new.out)

        # FFN
        ffn_old = old_block.ffn
        ffn_new = new_block.ffn
        self._pad_linear(ffn_old.gate_up, ffn_new.gate_up)
        self._pad_linear(ffn_old.down,    ffn_new.down)

    def _pad_linear(self, old_lin: nn.Linear, new_lin: nn.Linear):
        """
        Copy old weight into new_lin, padding extra rows/cols with near-zero noise.
        Works for any (out_features, in_features) shape change.
        """
        old_w = old_lin.weight.data
        new_w = new_lin.weight.data
        old_out, old_in = old_w.shape
        new_out, new_in = new_w.shape

        # Fill new weight with near-zero noise, then overwrite the old region
        nn.init.normal_(new_w, mean=0.0, std=0.001)
        copy_out = min(old_out, new_out)
        copy_in  = min(old_in,  new_in)
        new_w[:copy_out, :copy_in] = old_w[:copy_out, :copy_in]

    # ── Freeze old saturated layers ───────────────────────────────────────────

    def _maybe_freeze_old_layers(self):
        """Freeze layers that are old and saturated to free VRAM."""
        gc = self.growth_config
        for i, h in enumerate(self._health):
            if h.is_frozen:
                continue
            if (h.age_steps > gc.freeze_after_steps
                    and h.grad_magnitude < gc.grad_saturation_threshold):
                # Freeze and move to CPU
                for p in self.blocks[i].parameters():
                    p.requires_grad_(False)
                self.blocks[i] = self.blocks[i].cpu()
                h.is_frozen = True
                print(f"[GrowingAvus] Froze layer {i} (age={h.age_steps})")

    # ── Utility ───────────────────────────────────────────────────────────────

    def _estimate_vram_gb(self, fp16: bool = False) -> float:
        """Rough VRAM estimate based on parameter count."""
        bytes_per_param = 2 if fp16 else 4
        return self.count_parameters() * bytes_per_param / 1e9

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def growth_summary(self) -> str:
        """Human-readable summary of the model's growth history."""
        gc = self.growth_config
        n_frozen = sum(1 for h in self._health if h.is_frozen)
        n_active = len(self.blocks) - n_frozen
        frozen_ids = [i for i, h in enumerate(self._health) if h.is_frozen]

        lines = [
            "=" * 55,
            "  GrowingAvus — Growth Summary",
            "=" * 55,
            f"  Depth  : {len(self.blocks)} layers  ({n_active} active, {n_frozen} frozen)",
            f"  Width  : dim={self.avus_config.dim}  heads={self.avus_config.n_heads}",
            f"  Params : {self.count_parameters():,}",
            f"  Steps  : {self._step}",
            f"  Events : {len(self._growth_log)} growth events",
        ]
        if frozen_ids:
            lines.append(f"  Frozen : layers {frozen_ids}")

        if self._growth_log:
            lines.append("")
            lines.append("  Growth log:")
            for ev in self._growth_log:
                if ev["event"] == "spawn_layer":
                    lines.append(
                        f"    step {ev['step']:>5}: spawn layer {ev['insert_pos']} "
                        f"(total={ev['total_layers']}, dim={ev['dim']})"
                    )
                elif ev["event"] == "grow_wider":
                    lines.append(
                        f"    step {ev['step']:>5}: grow wider "
                        f"{ev['old_dim']} -> {ev['new_dim']} "
                        f"(layers={ev['total_layers']})"
                    )

        lines.append("=" * 55)
        return "\n".join(lines)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save model state, growth config, growth log, and health data."""
        gc = self.growth_config
        torch.save(
            {
                "growth_config": {
                    k: v for k, v in gc.__dict__.items()
                },
                "avus_config": {
                    "vocab_size":  self.avus_config.vocab_size,
                    "dim":         self.avus_config.dim,
                    "n_layers":    self.avus_config.n_layers,
                    "n_heads":     self.avus_config.n_heads,
                    "n_kv_heads":  self.avus_config.n_kv_heads,
                    "ffn_hidden":  self.avus_config.ffn_hidden,
                    "max_seq_len": self.avus_config.max_seq_len,
                    "eps":         self.avus_config.eps,
                },
                "model_state_dict": self.state_dict(),
                "growth_log": self._growth_log,
                "health": [h.__dict__ for h in self._health],
                "step": self._step,
            },
            path,
        )
        print(f"[GrowingAvus] Checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "GrowingAvus":
        """Load GrowingAvus from a checkpoint."""
        ckpt = torch.load(path, map_location=device)

        gc_dict = ckpt.get("growth_config", {})
        gc = GrowthConfig(**{k: v for k, v in gc_dict.items()
                             if k in GrowthConfig.__dataclass_fields__})

        model = cls(gc)

        # Restore avus_config (dim may have grown)
        ac_dict = ckpt.get("avus_config", {})
        model.avus_config = AvusConfig(**{k: v for k, v in ac_dict.items()
                                          if k in AvusConfig.__dataclass_fields__})

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model._growth_log = ckpt.get("growth_log", [])
        model._step = ckpt.get("step", 0)

        health_data = ckpt.get("health", [])
        model._health = [LayerHealth(**h) for h in health_data]

        model.to(device)
        model.eval()
        return model

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation — same interface as Avus.generate()."""
        self.eval()
        for block in self.blocks:
            block.clear_cache()

        # Process prompt with KV cache
        _, _ = self.forward(idx, use_cache=True, cache_offset=0)
        prompt_len = idx.shape[1]

        for step in range(max_new_tokens):
            last = idx[:, -1:]
            logits, _ = self.forward(
                last,
                use_cache=True,
                cache_offset=prompt_len + step,
            )
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

        for block in self.blocks:
            block.clear_cache()
        return idx


# ── Main — quick smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building tiny GrowingAvus (dim=64, layers=2) for quick test...\n")

    gc = GrowthConfig(
        seed_dim=64,
        seed_layers=2,
        seed_heads=4,
        seed_kv_heads=2,
        vocab_size=256,       # tiny vocab for speed
        max_seq_len=64,
        spawn_after_steps=5,  # low threshold so growth triggers quickly
        freeze_after_steps=30,
        grad_saturation_threshold=0.01,
        activation_collapse_threshold=0.001,
    )

    model = GrowingAvus(gc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    param_history = []

    print(f"{'Step':>5}  {'Params':>10}  {'Layers':>7}  {'Dim':>6}  {'Grew?':>6}")
    print("-" * 45)

    for step in range(50):
        # Fake batch: random token ids
        x = torch.randint(0, gc.vocab_size, (2, 16))
        y = torch.randint(0, gc.vocab_size, (2, 16))

        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()

        model.record_gradients()
        grew = model.check_and_grow()

        # Rebuild optimizer if architecture changed (new params added)
        if grew:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        optimizer.step()

        n_params = model.count_parameters()
        param_history.append(n_params)

        if step % 5 == 0 or grew:
            print(
                f"{step:>5}  {n_params:>10,}  {len(model.blocks):>7}  "
                f"{model.avus_config.dim:>6}  {'YES' if grew else '':>6}"
            )

    print()
    print(model.growth_summary())

    print(f"\nParameter count over time (every 5 steps):")
    for i, p in enumerate(param_history[::5]):
        print(f"  step {i*5:>3}: {p:,}")
