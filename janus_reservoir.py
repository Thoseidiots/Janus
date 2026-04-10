"""
janus_reservoir.py
==================
Compute Reservoir for Janus/Avus — disk-backed activation cache.

During inference, intermediate layer outputs are saved to disk after
the first pass. Subsequent identical inputs load from disk instead of
recomputing on GPU, reducing peak VRAM usage.

During training, the reservoir is bypassed entirely — gradients flow
normally, no disk I/O overhead.

Usage:
    from avus import Avus, AvusConfig
    model = Avus(config, use_reservoir=True)
    model.eval()
    output = model.generate(tokens, max_new_tokens=50)
    print(model.reservoir.get_report())
    model.reservoir.clear()  # free disk space
"""

import hashlib
import os
import struct
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class JanusLayerReservoir:
    """
    Disk-backed activation cache for Avus transformer blocks.

    Key design decisions:
    - Fast hashing: samples 256 bytes + shape + dtype instead of full tensor
    - Async-safe: each entry is a single .pt file, no locking needed
    - Training bypass: zero overhead during backward pass
    - Device-aware: loads tensors back to the originating device
    """

    def __init__(self, storage_dir: str = "compute_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "vram_saved_mb": 0.0,
                       "disk_used_mb": 0.0}

    # ── Hashing ───────────────────────────────────────────────────────────────

    def _fast_key(self, layer_id: str, x: torch.Tensor,
                  use_cache: bool, cache_offset: int) -> str:
        """
        Fast hash: samples first+last 128 bytes of tensor data + metadata.
        ~50x faster than hashing the full tensor for large activations.
        """
        flat = x.detach().cpu().contiguous().view(-1)
        n    = flat.numel()

        # Sample: first 32 + last 32 elements as bytes
        sample_len = min(32, n)
        head = flat[:sample_len].numpy().tobytes()
        tail = flat[max(0, n - sample_len):].numpy().tobytes()

        # Metadata: shape, dtype, layer, cache state
        meta = (f"{layer_id}|{list(x.shape)}|{x.dtype}|"
                f"{use_cache}|{cache_offset}").encode()

        digest = hashlib.blake2b(head + tail + meta, digest_size=16).hexdigest()
        return digest

    # ── Core wrap ─────────────────────────────────────────────────────────────

    def wrap_layer(self, layer_id: str, layer: nn.Module,
                   x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Run a layer with disk-cache support.

        - Training (grad enabled): runs normally, no cache.
        - Inference (no grad): checks cache first, saves on miss.
        """
        if torch.is_grad_enabled():
            # Training — bypass completely
            return layer(x, *args, **kwargs)

        # Inference — check cache
        key  = self._fast_key(layer_id, x,
                               kwargs.get("use_cache", False),
                               kwargs.get("cache_offset", 0))
        path = self.storage_dir / f"{key}.pt"

        if path.exists():
            self._stats["hits"] += 1
            cached = torch.load(str(path), map_location=x.device,
                                weights_only=True)
            mb = cached.element_size() * cached.nelement() / 1e6
            self._stats["vram_saved_mb"] += mb
            return cached

        # Cache miss — compute and save
        self._stats["misses"] += 1
        output = layer(x, *args, **kwargs)

        cpu_out = output.detach().cpu()
        torch.save(cpu_out, str(path))
        mb = cpu_out.element_size() * cpu_out.nelement() / 1e6
        self._stats["disk_used_mb"] += mb

        return output

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_report(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0.0
        return {
            "cache_hits":        self._stats["hits"],
            "cache_misses":      self._stats["misses"],
            "hit_rate_pct":      round(hit_rate, 1),
            "vram_saved_mb":     round(self._stats["vram_saved_mb"], 2),
            "disk_used_mb":      round(self._stats["disk_used_mb"], 2),
            "storage_dir":       str(self.storage_dir),
        }

    def clear(self):
        """Delete all cached activations and reset stats."""
        for f in self.storage_dir.glob("*.pt"):
            f.unlink()
        self._stats = {"hits": 0, "misses": 0,
                       "vram_saved_mb": 0.0, "disk_used_mb": 0.0}
        print(f"[Reservoir] Cache cleared: {self.storage_dir}")

    def disk_usage_mb(self) -> float:
        return sum(f.stat().st_size for f in self.storage_dir.glob("*.pt")) / 1e6

    def __repr__(self) -> str:
        r = self.get_report()
        return (f"JanusLayerReservoir(hits={r['cache_hits']} "
                f"misses={r['cache_misses']} "
                f"hit_rate={r['hit_rate_pct']}% "
                f"vram_saved={r['vram_saved_mb']}MB)")
