"""
collaborative_trainer.py
========================
Mutual distillation training for Avus (token-based) and ByteLatentAvus (byte-based).

Each model teaches the other during training via hidden-state distillation.
Neither model's intelligence is reduced — they pull each other toward better
representations than either would find alone.

How it works per step:
  1. Both models forward pass on the same data (Avus sees tokens, BLT sees bytes)
  2. Extract hidden states from the final layer of each model
  3. Project both to a shared distillation space (shared_dim)
  4. KL divergence loss between the two projected distributions
  5. Each model's total loss = task_loss + distill_weight * kl_loss

The shared projection is learned — it finds the alignment between token-space
and byte-space representations automatically.

Usage in train_avus_kaggle.py:
    from collaborative_trainer import CollaborativeTrainer
    trainer = CollaborativeTrainer(avus_model, blt_model, device)
    loss_avus, loss_blt = trainer.step(token_x, token_y, byte_x, byte_y,
                                        avus_optimizer, blt_optimizer,
                                        avus_scaler, blt_scaler, step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SharedProjection(nn.Module):
    """
    Projects hidden states from both models into a shared distillation space.

    Avus hidden: (B, T_tokens, avus_dim)  → (B, T_shared, shared_dim)
    BLT hidden:  (B, N_patches, blt_dim)  → (B, T_shared, shared_dim)

    Both are mean-pooled to (B, shared_dim) for comparison.
    """
    def __init__(self, avus_dim: int, blt_dim: int, shared_dim: int = 256):
        super().__init__()
        self.avus_proj = nn.Sequential(
            nn.Linear(avus_dim, shared_dim, bias=False),
            nn.LayerNorm(shared_dim),
        )
        self.blt_proj = nn.Sequential(
            nn.Linear(blt_dim, shared_dim, bias=False),
            nn.LayerNorm(shared_dim),
        )

    def project_avus(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T, avus_dim) → (B, shared_dim) via mean pool + projection."""
        return self.avus_proj(hidden.mean(dim=1))

    def project_blt(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, N, blt_dim) → (B, shared_dim) via mean pool + projection."""
        return self.blt_proj(hidden.mean(dim=1))


def _kl_divergence(p: torch.Tensor, q: torch.Tensor,
                   temperature: float = 2.0) -> torch.Tensor:
    """
    Symmetric KL divergence between two distributions in shared space.
    Temperature > 1 softens the distributions — standard knowledge distillation trick.
    Higher temperature = more information transferred from soft labels.
    """
    p_soft = F.softmax(p / temperature, dim=-1)
    q_soft = F.softmax(q / temperature, dim=-1)
    # KL(p||q) + KL(q||p) — symmetric, so neither model dominates
    kl_pq = F.kl_div(q_soft.log(), p_soft, reduction="batchmean")
    kl_qp = F.kl_div(p_soft.log(), q_soft, reduction="batchmean")
    return (kl_pq + kl_qp) * 0.5 * (temperature ** 2)


class CollaborativeTrainer:
    """
    Manages joint training of Avus and ByteLatentAvus with mutual distillation.

    Both models train on the same semantic content but in their native formats:
    - Avus receives tokenized sequences
    - BLT receives raw byte sequences

    After each forward pass, hidden states are projected to a shared space
    and a distillation loss encourages alignment between the two representations.
    """

    def __init__(
        self,
        avus_model,
        blt_model,
        device,
        shared_dim:      int   = 256,
        distill_weight:  float = 0.1,   # weight of distillation loss
        distill_temp:    float = 2.0,   # softmax temperature for distillation
        distill_every:   int   = 4,     # only distill every N steps (saves memory)
    ):
        self.avus        = avus_model
        self.blt         = blt_model
        self.device      = device
        self.distill_w   = distill_weight
        self.distill_t   = distill_temp
        self.distill_every = distill_every

        # Shared projection — learns the alignment between token and byte spaces
        avus_dim = avus_model.config.dim
        blt_dim  = blt_model.cfg.global_dim
        self.projector = SharedProjection(avus_dim, blt_dim, shared_dim).to(device)

        # Track distillation stats
        self._distill_losses = []
        self._step = 0

    def parameters(self):
        """All parameters including the shared projector."""
        return list(self.projector.parameters())

    def _extract_avus_hidden(self, avus_model, idx: torch.Tensor) -> torch.Tensor:
        """
        Run Avus forward and extract final layer hidden states before the LM head.
        Returns (B, T, dim).
        """
        B, T = idx.shape
        x = avus_model.dropout(avus_model.tok_emb(idx))
        for block in avus_model.blocks:
            x = block(x)
        return avus_model.ln_f(x)  # (B, T, dim)

    def _extract_blt_hidden(self, blt_model, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Run BLT encoder + global model and extract patch hidden states.
        Returns (B, N_patches, global_dim).
        """
        patches_in = blt_model._bytes_to_patches(byte_ids)
        patch_emb  = blt_model.encoder(patches_in)
        hidden     = blt_model.global_(patch_emb)
        return hidden  # (B, N_patches, global_dim)

    def step(
        self,
        # Avus inputs (tokenized)
        token_x:      torch.Tensor,
        token_y:      torch.Tensor,
        avus_optimizer,
        avus_scaler,
        # BLT inputs (raw bytes)
        byte_x:       torch.Tensor,
        byte_y:       torch.Tensor,
        blt_optimizer,
        blt_scaler,
        # Shared projector optimizer
        proj_optimizer,
        grad_accum_steps: int = 1,
        kaggle_mode:      bool = False,
    ) -> Tuple[float, float, float]:
        """
        One collaborative training step.

        Returns:
            (avus_loss, blt_loss, distill_loss) as floats
        """
        self._step += 1
        use_amp = (self.device.type == "cuda" and not kaggle_mode)
        do_distill = (self._step % self.distill_every == 0)

        # ── Avus forward ──────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=use_amp):
            avus_logits, avus_task_loss = self.avus(token_x, targets=token_y)
            if avus_task_loss is not None and avus_task_loss.dim() > 0:
                avus_task_loss = avus_task_loss.mean()

        # ── BLT forward ───────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, blt_task_loss = self.blt(byte_x, target_bytes=byte_y)
            if blt_task_loss is not None and blt_task_loss.dim() > 0:
                blt_task_loss = blt_task_loss.mean()

        # ── Mutual distillation ───────────────────────────────────────────────
        distill_loss_val = 0.0
        if do_distill and avus_task_loss is not None and blt_task_loss is not None:
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Extract hidden states
                raw_avus = self.avus.module if hasattr(self.avus, "module") else self.avus
                raw_blt  = self.blt.module  if hasattr(self.blt,  "module") else self.blt

                avus_hidden = self._extract_avus_hidden(raw_avus, token_x)
                blt_hidden  = self._extract_blt_hidden(raw_blt,  byte_x)

                # Project to shared space
                avus_proj = self.projector.project_avus(avus_hidden)  # (B, shared_dim)
                blt_proj  = self.projector.project_blt(blt_hidden)    # (B, shared_dim)

                # Symmetric KL divergence
                distill_loss = _kl_divergence(avus_proj, blt_proj, self.distill_t)
                distill_loss_val = distill_loss.item()
                self._distill_losses.append(distill_loss_val)

                # Add distillation loss to both models
                avus_total = avus_task_loss + self.distill_w * distill_loss
                blt_total  = blt_task_loss  + self.distill_w * distill_loss
        else:
            avus_total = avus_task_loss if avus_task_loss is not None else torch.tensor(0.0)
            blt_total  = blt_task_loss  if blt_task_loss  is not None else torch.tensor(0.0)

        # ── Backward: Avus ────────────────────────────────────────────────────
        avus_scaler.scale(avus_total / grad_accum_steps).backward(retain_graph=do_distill)

        # ── Backward: BLT ─────────────────────────────────────────────────────
        blt_scaler.scale(blt_total / grad_accum_steps).backward()

        # ── Optimizer steps ───────────────────────────────────────────────────
        if self._step % grad_accum_steps == 0:
            # Avus
            avus_scaler.unscale_(avus_optimizer)
            torch.nn.utils.clip_grad_norm_(self.avus.parameters(), 1.0)
            avus_scaler.step(avus_optimizer)
            avus_scaler.update()
            avus_optimizer.zero_grad(set_to_none=True)

            # BLT
            blt_scaler.unscale_(blt_optimizer)
            torch.nn.utils.clip_grad_norm_(self.blt.parameters(), 1.0)
            blt_scaler.step(blt_optimizer)
            blt_scaler.update()
            blt_optimizer.zero_grad(set_to_none=True)

            # Shared projector
            if do_distill:
                proj_optimizer.step()
                proj_optimizer.zero_grad(set_to_none=True)

        return (
            avus_task_loss.item() if avus_task_loss is not None else 0.0,
            blt_task_loss.item()  if blt_task_loss  is not None else 0.0,
            distill_loss_val,
        )

    def distill_summary(self) -> str:
        """Print a summary of distillation alignment over recent steps."""
        if not self._distill_losses:
            return "[collab] No distillation steps yet"
        recent = self._distill_losses[-50:]
        avg    = sum(recent) / len(recent)
        trend  = "↓ converging" if len(recent) > 10 and recent[-1] < recent[0] else "→ stable"
        return (f"[collab] Distillation loss (last {len(recent)} steps): "
                f"avg={avg:.4f}  {trend}  "
                f"(lower = models more aligned)")
