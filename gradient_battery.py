"""
gradient_battery.py
===================
Persistent gradient accumulation buffer across training sessions.

Concept:
  Like a battery — charge it slowly (accumulate gradients from many
  small runs), discharge it in one shot (apply a large optimizer step).

  Session 1: compute gradients for 500 steps, save to disk. Don't step.
  Session 2: load saved gradients, add 500 more, save. Don't step.
  Session N: load, add, discharge — apply one massive optimizer step
             equivalent to N*500 steps of gradient signal.

Why this helps:
  - Each session contributes gradient signal even if it can't finish an epoch
  - The optimizer step happens when YOU decide, not when the session ends
  - Gradients from different data batches accumulate into a stronger signal
  - Works on CPU — you can charge the battery locally without a GPU

Usage:
    from gradient_battery import GradientBattery

    battery = GradientBattery("gradient_battery.pt")

    # Charging (accumulate gradients, don't step optimizer)
    loss.backward()
    battery.charge(model)          # saves gradients, clears from model
    optimizer.zero_grad()

    # Check charge level
    print(battery.status())

    # Discharging (apply accumulated gradients)
    if battery.is_ready(min_steps=1000):
        battery.discharge(model)   # loads gradients back into model.grad
        optimizer.step()
        optimizer.zero_grad()
        battery.reset()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class GradientBattery:
    """
    Persistent cross-session gradient accumulation buffer.

    Stores accumulated gradients on disk between sessions.
    Charge by calling charge(model) after loss.backward().
    Discharge by calling discharge(model) before optimizer.step().
    """

    def __init__(self, path: str = "gradient_battery.pt",
                 device: str = "cpu"):
        """
        Args:
            path:   Where to store the gradient buffer on disk
            device: Device to load gradients onto during discharge
                    ('cpu' for local, 'cuda' for Kaggle)
        """
        self.path   = Path(path)
        self.device = device
        self._buf:  Dict[str, torch.Tensor] = {}  # param_name -> grad tensor
        self._meta: Dict = {
            "steps_charged":   0,
            "total_samples":   0,
            "created_at":      time.time(),
            "last_charged_at": None,
            "last_discharged_at": None,
            "charge_sessions": 0,
        }
        self._load()

    # ── Charging ──────────────────────────────────────────────────────────────

    def charge(self, model: nn.Module, n_samples: int = 1,
               scale: float = 1.0):
        """
        Accumulate current model gradients into the battery.

        Call this after loss.backward() and before optimizer.zero_grad().
        Gradients are moved to CPU and added to the persistent buffer.

        Args:
            model:     The model whose .grad tensors to accumulate
            n_samples: Number of samples this backward pass covered
                       (used for tracking, not scaling)
            scale:     Optional scaling factor for the gradients
                       (e.g. 1/grad_accum_steps)
        """
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().cpu()
            if scale != 1.0:
                grad = grad * scale
            if name in self._buf:
                self._buf[name] += grad
            else:
                self._buf[name] = grad.clone()

        self._meta["steps_charged"]   += 1
        self._meta["total_samples"]   += n_samples
        self._meta["last_charged_at"]  = time.time()
        self._meta["charge_sessions"]  = self._meta.get("charge_sessions", 0) + 1

    def save(self):
        """Persist the battery to disk."""
        torch.save({
            "gradients": self._buf,
            "meta":      self._meta,
        }, str(self.path))

    # ── Discharging ───────────────────────────────────────────────────────────

    def discharge(self, model: nn.Module, normalize: bool = True):
        """
        Load accumulated gradients back into model.grad tensors.

        Call this before optimizer.step(). After calling this,
        the optimizer will apply the full accumulated gradient signal.

        Args:
            model:     The model to load gradients into
            normalize: If True, divide by steps_charged so the effective
                       gradient magnitude is per-step (recommended)
        """
        if not self._buf:
            print("[GradientBattery] Battery is empty — nothing to discharge")
            return

        steps = max(1, self._meta["steps_charged"])
        scale = 1.0 / steps if normalize else 1.0

        loaded = 0
        for name, param in model.named_parameters():
            if name not in self._buf:
                continue
            grad = self._buf[name].to(self.device) * scale
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad
            loaded += 1

        self._meta["last_discharged_at"] = time.time()
        print(f"[GradientBattery] Discharged {loaded} param gradients "
              f"({steps} steps accumulated, normalize={normalize})")

    # ── Status ────────────────────────────────────────────────────────────────

    def is_ready(self, min_steps: int = 500) -> bool:
        """Returns True if enough gradient steps have been accumulated."""
        return self._meta["steps_charged"] >= min_steps

    def charge_level(self) -> float:
        """
        Returns charge level as a fraction of a target (1000 steps = full).
        0.0 = empty, 1.0 = full, >1.0 = overcharged (still valid).
        """
        return self._meta["steps_charged"] / 1000.0

    def status(self) -> str:
        """Human-readable battery status."""
        steps    = self._meta["steps_charged"]
        samples  = self._meta["total_samples"]
        sessions = self._meta.get("charge_sessions", 0)
        n_params = len(self._buf)
        level    = self.charge_level()
        bar_len  = min(20, int(level * 20))
        bar      = "█" * bar_len + "░" * (20 - bar_len)

        last_charged = self._meta.get("last_charged_at")
        last_str = (f"{(time.time() - last_charged)/3600:.1f}h ago"
                    if last_charged else "never")

        return (
            f"GradientBattery [{bar}] {level*100:.0f}%\n"
            f"  Steps charged:   {steps:,}\n"
            f"  Total samples:   {samples:,}\n"
            f"  Charge sessions: {sessions}\n"
            f"  Param buffers:   {n_params}\n"
            f"  Last charged:    {last_str}\n"
            f"  Storage:         {self.path} "
            f"({self._disk_size_mb():.1f} MB)"
        )

    def reset(self):
        """Clear the battery after discharging."""
        self._buf  = {}
        self._meta["steps_charged"]      = 0
        self._meta["total_samples"]      = 0
        self._meta["last_discharged_at"] = time.time()
        if self.path.exists():
            self.path.unlink()
        print("[GradientBattery] Battery reset")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self):
        """Load existing battery from disk if it exists."""
        if not self.path.exists():
            return
        try:
            data = torch.load(str(self.path), map_location="cpu",
                              weights_only=False)
            self._buf  = data.get("gradients", {})
            self._meta.update(data.get("meta", {}))
            steps = self._meta["steps_charged"]
            print(f"[GradientBattery] Loaded from {self.path} "
                  f"({steps} steps charged, "
                  f"{len(self._buf)} param buffers)")
        except Exception as e:
            print(f"[GradientBattery] Load failed ({e}) — starting fresh")
            self._buf  = {}

    def _disk_size_mb(self) -> float:
        if not self.path.exists():
            return 0.0
        return self.path.stat().st_size / 1e6

    def __repr__(self) -> str:
        return (f"GradientBattery(steps={self._meta['steps_charged']} "
                f"params={len(self._buf)} "
                f"path={self.path})")


# ── Convenience: integrate with training loop ─────────────────────────────────

def charge_from_dataloader(
    model: nn.Module,
    loader,
    loss_fn,
    battery: GradientBattery,
    device: str = "cpu",
    max_steps: int = 500,
    grad_accum: int = 1,
):
    """
    Convenience function: run a charging loop without an optimizer.
    Accumulates gradients into the battery without updating weights.

    Useful for local CPU charging between Kaggle sessions.

    Args:
        model:      The model to compute gradients for
        loader:     DataLoader
        loss_fn:    Callable(logits, targets) -> loss
        battery:    GradientBattery to charge
        device:     Device to run on
        max_steps:  Stop after this many steps
        grad_accum: Gradient accumulation steps
    """
    model.train()
    model.to(device)
    step = 0

    print(f"[GradientBattery] Charging for up to {max_steps} steps on {device}...")

    for x, y in loader:
        if step >= max_steps:
            break

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        if loss is None:
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            battery.charge(model, n_samples=x.shape[0],
                           scale=1.0 / grad_accum)
            # Zero grads but DON'T step optimizer
            for p in model.parameters():
                p.grad = None

        step += 1
        if step % 50 == 0:
            print(f"  step {step}/{max_steps} loss={loss.item():.4f}")

    battery.save()
    print(f"[GradientBattery] Charging complete. {battery.status()}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from avus import Avus, AvusConfig

    print("Testing GradientBattery...\n")

    cfg   = AvusConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       vocab_size=256, max_seq_len=16)
    model = Avus(cfg)
    battery = GradientBattery("test_battery.pt")

    # Simulate 3 charging sessions
    for session in range(3):
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        _, loss = model(x, targets=y)
        loss.backward()
        battery.charge(model, n_samples=2)
        for p in model.parameters():
            p.grad = None
        battery.save()
        print(f"Session {session+1}: charged")

    print("\n" + battery.status())

    # Discharge
    print("\nDischarging...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    battery.discharge(model)
    opt.step()
    opt.zero_grad()

    print("Optimizer step applied.")
    battery.reset()

    # Cleanup
    Path("test_battery.pt").unlink(missing_ok=True)
    print("Done.")
