"""
train.py
========
Unified training entry point for all Janus AI models.

Trains:
  - Avus transformer (any size: 1b, 3b, 7b, 13b, 34b, 70b)
  - HolographicBrainMemory (complex + real-valued)
  - SpawningBrain

Also supports weight combining:
  - SLERP  : smooth interpolation between two checkpoints (same architecture)
  - DARE   : task arithmetic — merge specialist models into one
  - SOUP   : simple weight averaging across N checkpoints

No API keys. Runs locally on CPU or any CUDA GPU.

Usage:
    # Train Avus 1B from scratch
    python train.py --model avus --size 1b --epochs 10

    # Resume training
    python train.py --model avus --size 7b --resume --epochs 5

    # Train HBM
    python train.py --model hbm --epochs 20

    # Combine two Avus checkpoints (SLERP)
    python train.py --merge slerp --a weights_a.pt --b weights_b.pt --out merged.pt

    # Merge N specialist checkpoints (DARE/task arithmetic)
    python train.py --merge dare --inputs a.pt b.pt c.pt --out merged.pt

    # Average N checkpoints (model soup)
    python train.py --merge soup --inputs a.pt b.pt c.pt --out merged.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Repo root on path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """
    Flat token dataset. Loads a .pt tensor or generates synthetic data.
    Each sample is a (seq_len,) slice; target is the same slice shifted by 1.
    """
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n       = max(0, len(tokens) - seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def _synthetic_tokens(vocab_size: int, n_tokens: int = 500_000) -> torch.Tensor:
    """Generate random token stream for smoke-testing."""
    return torch.randint(0, vocab_size, (n_tokens,), dtype=torch.long)


def _load_or_generate_tokens(data_path: Optional[str], vocab_size: int,
                              n_tokens: int = 500_000) -> torch.Tensor:
    if data_path and Path(data_path).exists():
        print(f"[data] Loading tokens from {data_path}")
        obj = torch.load(data_path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.long()
        if isinstance(obj, dict) and "tokens" in obj:
            return obj["tokens"].long()
    print(f"[data] No data file found — generating {n_tokens:,} synthetic tokens")
    return _synthetic_tokens(vocab_size, n_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_lr(step: int, warmup: int, total: int, max_lr: float,
               min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Avus trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_avus(args):
    from avus import Avus, AvusConfig

    # Route to GrowingAvus if requested
    if getattr(args, 'growing', False):
        return train_growing(args)

    device = torch.device(args.device)
    print(f"\n[avus] Training Avus-{args.size.upper()} on {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    config_file = ROOT / f"config_avus_{args.size}.json"
    if config_file.exists():
        cfg = AvusConfig.from_file(str(config_file))
        print(f"[avus] Config from {config_file}")
    else:
        cfg = AvusConfig()
        print(f"[avus] Using default AvusConfig (1B)")

    print(f"[avus] dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"kv={cfg.n_kv_heads} ffn={cfg.ffn_hidden} seq={cfg.max_seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Avus(cfg).to(device)
    total_params = model.count_parameters()
    print(f"[avus] Parameters: {total_params/1e9:.2f}B ({total_params:,})")

    # ── Resume ────────────────────────────────────────────────────────────────
    weights_path = ROOT / f"avus_{args.size}_weights.pt"
    start_epoch  = 0
    if args.resume and weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        sd   = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get("epoch", 0)
        print(f"[avus] Resumed from epoch {start_epoch}")

    # ── Skill curriculum ──────────────────────────────────────────────────────
    from skill_curriculum import SkillTree, CurriculumSampler
    skill_tree = SkillTree()
    skill_path = ROOT / "skill_state.json"
    if skill_path.exists():
        skill_tree.load(str(skill_path))
        print(f"[avus] Skill state loaded from {skill_path}")
    sampler = CurriculumSampler(skill_tree)
    print(f"[avus] Skill curriculum active — training: {sampler.next_domain()}")

    # ── Data ──────────────────────────────────────────────────────────────────
    tokens  = _load_or_generate_tokens(args.data, cfg.vocab_size)
    seq_len = min(cfg.max_seq_len, 512)   # cap for memory safety
    dataset = TokenDataset(tokens, seq_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[avus] Dataset: {len(dataset):,} samples | batch={args.batch_size} | seq={seq_len}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    total_steps  = len(loader) * args.epochs
    warmup_steps = min(2000, total_steps // 10)
    scaler       = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            lr = _cosine_lr(global_step, warmup_steps, total_steps, args.lr, args.lr / 10)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss  += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                # Update skill confidence based on inverse loss (lower loss = higher confidence)
                current_domain = sampler.next_domain()
                confidence = max(0.0, min(1.0, 1.0 - (avg / 10.0)))
                skill_tree.update(current_domain, confidence)
                print(f"  epoch {epoch+1}/{start_epoch+args.epochs} "
                      f"step {step}/{len(loader)} "
                      f"loss={avg:.4f} lr={lr:.2e} t={elapsed:.0f}s "
                      f"| skill={current_domain} conf={confidence:.2f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[avus] Epoch {epoch+1} done — avg loss={avg_loss:.4f}")

        # Save skill state and render chart
        skill_tree.save(str(skill_path))
        skill_tree.plot(str(ROOT / f"skill_chart_epoch{epoch+1}.png"))
        print(f"[avus] Skill state: {skill_tree.best_skill_to_train()} is next priority")

        # Save checkpoint
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": cfg.vocab_size, "dim": cfg.dim,
                "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads, "ffn_hidden": cfg.ffn_hidden,
                "max_seq_len": cfg.max_seq_len,
            },
            "loss": avg_loss,
        }, weights_path)
        print(f"[avus] Saved → {weights_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[avus] Training complete. Weights at {weights_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GrowingAvus trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_growing(args):
    """Train GrowingAvus — no fixed size, grows as needed."""
    from growing_avus import GrowingAvus, GrowthConfig

    device = torch.device(args.device)
    print(f"\n[growing] Training GrowingAvus on {device}")

    gc = GrowthConfig(
        seed_dim=256, seed_layers=2, seed_heads=4, seed_kv_heads=2,
        vocab_size=50304, max_seq_len=512,
        spawn_after_steps=100,
        freeze_after_steps=500,
        grad_saturation_threshold=0.001,
        activation_collapse_threshold=0.01,
        vram_budget_gb=14.0,
    )

    weights_path = ROOT / "avus_growing_weights.pt"
    model = GrowingAvus(gc).to(device)

    if args.resume and weights_path.exists():
        model = GrowingAvus.from_checkpoint(str(weights_path), device=str(device))
        print(f"[growing] Resumed — {len(model.blocks)} layers, "
              f"dim={model.avus_config.dim}, step={model._step}")
    else:
        print(f"[growing] Starting fresh — seed: {gc.seed_layers} layers, dim={gc.seed_dim}")

    # Skill curriculum
    from skill_curriculum import SkillTree, CurriculumSampler
    skill_tree = SkillTree()
    skill_path = ROOT / "skill_state.json"
    if skill_path.exists():
        skill_tree.load(str(skill_path))
    sampler = CurriculumSampler(skill_tree)

    # Data
    tokens  = _load_or_generate_tokens(args.data, gc.vocab_size)
    seq_len = min(gc.max_seq_len, 512)
    dataset = TokenDataset(tokens, seq_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0,
                         pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, loss = model(x, targets=y)
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()

            # Record gradients for growth decisions
            model.record_gradients()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Check if model should grow
            grew = model.check_and_grow()
            if grew:
                # Rebuild optimizer with new parameters
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr,
                    betas=(0.9, 0.95), weight_decay=0.1
                )

            epoch_loss += loss.item()

            if step % args.log_every == 0:
                avg     = epoch_loss / (step + 1)
                elapsed = time.time() - t0
                domain  = sampler.next_domain()
                conf    = max(0.0, min(1.0, 1.0 - avg / 10.0))
                skill_tree.update(domain, conf)
                print(f"  epoch {epoch+1} step {step}/{len(loader)} "
                      f"loss={avg:.4f} t={elapsed:.0f}s "
                      f"layers={len(model.blocks)} dim={model.avus_config.dim} "
                      f"params={model.count_parameters()/1e6:.1f}M")

        avg_loss = epoch_loss / len(loader)
        print(f"\n[growing] Epoch {epoch+1} — loss={avg_loss:.4f}")
        print(model.growth_summary())

        model.save_checkpoint(str(weights_path))
        skill_tree.save(str(skill_path))
        gc_module = __import__("gc")
        gc_module.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[growing] Training complete.")
    print(model.growth_summary())


# ─────────────────────────────────────────────────────────────────────────────
# Train all configs sequentially
# ─────────────────────────────────────────────────────────────────────────────

def train_all_configs(args):
    """
    Train every Avus config that fits in available VRAM, smallest first.
    Uses the same dataset and skill state across all runs.
    Stops when a config would OOM.
    """
    import torch

    sizes = ["1b", "3b", "7b", "13b", "34b", "70b"]
    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_gb *= torch.cuda.device_count()
        print(f"[all] Total VRAM: {vram_gb:.1f} GB across {torch.cuda.device_count()} GPUs")

    # Rough fp16 VRAM requirements per size
    vram_needed = {
        "1b": 2.5, "3b": 6.0, "7b": 14.0,
        "13b": 26.0, "34b": 68.0, "70b": 140.0,
    }

    trained = []
    for size in sizes:
        needed = vram_needed.get(size, 999)
        if vram_gb > 0 and needed > vram_gb * 0.85:
            print(f"[all] Skipping avus-{size} — needs {needed:.0f} GB, "
                  f"have {vram_gb:.1f} GB")
            continue

        config_file = ROOT / f"config_avus_{size}.json"
        if not config_file.exists():
            print(f"[all] Skipping avus-{size} — config not found")
            continue

        print(f"\n[all] ── Training avus-{size} ──────────────────────────────")
        args.size = size
        try:
            train_avus(args)
            trained.append(size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[all] avus-{size} OOM — stopping here")
                break
            raise

    print(f"\n[all] Trained: {trained}")
    return trained


# ─────────────────────────────────────────────────────────────────────────────
# HBM trainer
# ─────────────────────────────────────────────────────────────────────────────

class HBMDataset(Dataset):
    """
    Synthetic dataset for HBM: random key-value pairs to memorize and retrieve.
    """
    def __init__(self, dim: int, n_samples: int = 10_000):
        self.dim      = dim
        self.n        = n_samples
        self.keys     = torch.randn(n_samples, dim)
        self.values   = torch.randn(n_samples, dim)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.keys[idx], self.values[idx]


def train_hbm(args):
    from holographic_brain_memory.core import HolographicBrainMemory, PhaseBrainLayer
    from holographic_brain_memory.real_valued import RealHolographicMemory, RealPhaseBrainLayer
    from holographic_brain_memory.spawning import SpawningBrain

    device = torch.device(args.device)
    dim    = args.hbm_dim
    in_dim = args.hbm_in_dim

    print(f"\n[hbm] Training HBM models on {device}")
    print(f"[hbm] dim={dim} in_dim={in_dim}")

    results = {}

    for variant, MemCls, LayerCls in [
        ("complex", HolographicBrainMemory, PhaseBrainLayer),
        ("real",    RealHolographicMemory,  RealPhaseBrainLayer),
    ]:
        print(f"\n[hbm] Variant: {variant}")
        memory = MemCls(dim=dim).to(device)
        layer  = LayerCls(in_dim=in_dim, memory=memory).to(device)

        dataset = HBMDataset(in_dim, n_samples=5_000)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(args.epochs):
            total_loss = 0.0
            for keys, _ in loader:
                keys = keys.to(device)
                out  = layer(keys)
                # Self-supervised: output should reconstruct input projection
                target = keys[:1].expand(out.shape[0], -1)[:, :out.shape[-1]]
                if out.shape != target.shape:
                    target = torch.zeros_like(out)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"  [{variant}] epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

        path = ROOT / f"hbm_{variant}_weights.pt"
        torch.save({
            "memory_state": memory.state_dict(),
            "layer_state":  layer.state_dict(),
            "dim":          dim,
            "in_dim":       in_dim,
            "variant":      variant,
        }, path)
        print(f"[hbm] Saved {variant} → {path}")
        results[variant] = str(path)

    # SpawningBrain
    print(f"\n[hbm] Training SpawningBrain")
    brain = SpawningBrain(in_dim=in_dim, memory_dim=dim, initial_layers=1).to(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for _ in range(100):
            x    = torch.randn(1, in_dim, device=device)
            out  = brain(x)
            loss = F.mse_loss(out, torch.zeros_like(out))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            brain.check_and_spawn(x, threshold=0.1)

        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  [spawning] epoch {epoch+1}/{args.epochs} "
                  f"loss={total_loss/100:.4f} neurons={brain.get_logical_capacity()}")

    path = ROOT / "hbm_spawning_weights.pt"
    torch.save({"state": brain.state_dict(), "spawn_history": brain.spawn_history}, path)
    print(f"[hbm] Saved spawning → {path}")

    print("\n[hbm] All HBM models trained.")


# ─────────────────────────────────────────────────────────────────────────────
# Weight combining
# ─────────────────────────────────────────────────────────────────────────────

def _load_sd(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a state dict from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _save_merged(sd: Dict[str, torch.Tensor], path: str, meta: dict):
    """Save a merged state dict with metadata."""
    torch.save({"model_state_dict": sd, "merge_meta": meta}, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"[merge] Saved → {path} ({size_mb:.1f} MB)")


def merge_slerp(args):
    """
    SLERP: Spherical Linear Interpolation between two checkpoints.
    Produces a smooth blend — good for merging the same model at different
    training stages or with different data mixes.

    t=0.0 → pure model A
    t=1.0 → pure model B
    t=0.5 → midpoint (default)
    """
    t  = args.slerp_t
    sd_a = _load_sd(args.a)
    sd_b = _load_sd(args.b)

    print(f"[slerp] Merging {args.a} + {args.b} at t={t}")

    merged = {}
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    common = keys_a & keys_b

    for key in common:
        a = sd_a[key].float()
        b = sd_b[key].float()

        if a.shape != b.shape:
            print(f"  [slerp] shape mismatch on {key} — using A")
            merged[key] = a
            continue

        # Non-float tensors (e.g. masks, int buffers) — just take A
        if not a.is_floating_point():
            merged[key] = a
            continue

        # Flatten for SLERP
        flat_a = a.reshape(-1)
        flat_b = b.reshape(-1)

        norm_a = flat_a.norm()
        norm_b = flat_b.norm()

        if norm_a < 1e-8 or norm_b < 1e-8:
            merged[key] = (a * (1 - t) + b * t).to(sd_a[key].dtype)
            continue

        unit_a = flat_a / norm_a
        unit_b = flat_b / norm_b

        dot = torch.clamp((unit_a * unit_b).sum(), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta.abs() < 1e-6:
            # Nearly identical — linear blend
            result = flat_a * (1 - t) + flat_b * t
        else:
            sin_theta = torch.sin(theta)
            result = (torch.sin((1 - t) * theta) / sin_theta) * flat_a + \
                     (torch.sin(t * theta)       / sin_theta) * flat_b

        merged[key] = result.reshape(a.shape).to(sd_a[key].dtype)

    # Keys only in A or B
    for key in keys_a - keys_b:
        merged[key] = sd_a[key]
    for key in keys_b - keys_a:
        merged[key] = sd_b[key]

    _save_merged(merged, args.out, {"method": "slerp", "t": t, "a": args.a, "b": args.b})
    print(f"[slerp] Done. {len(merged)} tensors merged.")


def merge_dare(args):
    """
    DARE / Task Arithmetic: merge N specialist models into one base model.

    Algorithm:
      1. Compute task vectors: delta_i = model_i - base
      2. Sparsify each delta (drop weights below threshold)
      3. Sum scaled task vectors onto base: merged = base + scale * sum(deltas)

    If no base is provided, uses the first input as base.
    """
    scale     = args.dare_scale
    density   = args.dare_density   # fraction of weights to keep per delta
    inputs    = args.inputs

    print(f"[dare] Merging {len(inputs)} models | scale={scale} density={density}")

    base_sd = _load_sd(inputs[0])
    print(f"[dare] Base: {inputs[0]}")

    # Accumulate task vectors
    delta_sum: Dict[str, torch.Tensor] = {}

    for path in inputs[1:]:
        print(f"[dare] Adding task vector from {path}")
        sd = _load_sd(path)

        for key in base_sd:
            if key not in sd:
                continue
            base = base_sd[key].float()
            expert = sd[key].float()

            if base.shape != expert.shape or not base.is_floating_point():
                continue

            delta = expert - base

            # DARE sparsification: randomly drop (1-density) fraction of weights
            if density < 1.0:
                mask  = torch.bernoulli(torch.full_like(delta, density))
                delta = delta * mask / max(density, 1e-8)

            if key not in delta_sum:
                delta_sum[key] = torch.zeros_like(delta)
            delta_sum[key] += delta

    # Apply to base
    merged = {}
    for key, base_val in base_sd.items():
        if key in delta_sum:
            result = base_val.float() + scale * delta_sum[key]
            merged[key] = result.to(base_val.dtype)
        else:
            merged[key] = base_val

    _save_merged(merged, args.out, {
        "method": "dare", "scale": scale, "density": density, "inputs": inputs
    })
    print(f"[dare] Done. {len(merged)} tensors merged.")


def merge_soup(args):
    """
    Model Soup: simple uniform weight averaging across N checkpoints.
    Best when all checkpoints are fine-tunes of the same base model.
    """
    inputs = args.inputs
    print(f"[soup] Averaging {len(inputs)} checkpoints")

    sds = [_load_sd(p) for p in inputs]
    all_keys = set(sds[0].keys())

    merged = {}
    for key in all_keys:
        tensors = [sd[key].float() for sd in sds if key in sd and sd[key].is_floating_point()]
        if tensors:
            merged[key] = torch.stack(tensors).mean(0).to(sds[0][key].dtype)
        else:
            merged[key] = sds[0][key]

    _save_merged(merged, args.out, {"method": "soup", "inputs": inputs})
    print(f"[soup] Done. {len(merged)} tensors averaged.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Janus unified training & weight merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model",  choices=["avus", "hbm", "all"],
                      help="Train a model")
    mode.add_argument("--merge",  choices=["slerp", "dare", "soup"],
                      help="Merge/combine weights")

    # ── Training args ─────────────────────────────────────────────────────────
    parser.add_argument("--size",       default="1b",
                        choices=["1b","3b","7b","13b","34b","70b"],
                        help="Avus model size (default: 1b)")
    parser.add_argument("--growing",    action="store_true",
                        help="Use GrowingAvus (no fixed size, grows during training)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=4,   dest="batch_size")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--data",       type=str,   default=None,
                        help="Path to token .pt file (optional)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every",  type=int,   default=50,  dest="log_every")

    # ── HBM args ──────────────────────────────────────────────────────────────
    parser.add_argument("--hbm-dim",    type=int,   default=1024, dest="hbm_dim")
    parser.add_argument("--hbm-in-dim", type=int,   default=64,   dest="hbm_in_dim")

    # ── Merge args ────────────────────────────────────────────────────────────
    parser.add_argument("--a",          type=str,   help="SLERP: first checkpoint")
    parser.add_argument("--b",          type=str,   help="SLERP: second checkpoint")
    parser.add_argument("--inputs",     nargs="+",  help="DARE/SOUP: list of checkpoints")
    parser.add_argument("--out",        type=str,   default="merged.pt",
                        help="Output path for merged weights")
    parser.add_argument("--slerp-t",    type=float, default=0.5,  dest="slerp_t",
                        help="SLERP interpolation factor (0=A, 1=B, default 0.5)")
    parser.add_argument("--dare-scale", type=float, default=0.5,  dest="dare_scale",
                        help="DARE task vector scale (default 0.5)")
    parser.add_argument("--dare-density", type=float, default=0.5, dest="dare_density",
                        help="DARE sparsification density (default 0.5)")

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.model:
        if args.model in ("avus", "all"):
            if getattr(args, 'growing', False):
                train_growing(args)
            elif getattr(args, 'all_configs', False):
                train_all_configs(args)
            else:
                train_avus(args)
        if args.model in ("hbm", "all"):
            train_hbm(args)

    elif args.merge == "slerp":
        if not args.a or not args.b:
            parser.error("--merge slerp requires --a and --b")
        merge_slerp(args)

    elif args.merge == "dare":
        if not args.inputs or len(args.inputs) < 2:
            parser.error("--merge dare requires at least 2 --inputs")
        merge_dare(args)

    elif args.merge == "soup":
        if not args.inputs or len(args.inputs) < 2:
            parser.error("--merge soup requires at least 2 --inputs")
        merge_soup(args)


if __name__ == "__main__":
    main()
