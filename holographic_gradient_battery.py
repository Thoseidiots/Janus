import os
import hashlib
import torch
from pathlib import Path

class HolographicGradientBattery:
    """
    Holographic Brain Memory (HBM) applied to Parameter Gradient storage.
    
    Compresses Gigabytes of raw neural network gradients into an ultra-dense, 
    fixed-size "pocket dimension" phase-space vector using Count-Sketch 
    Random Projection. This bypasses disk space explosions during large-scale 
    Kaggle runs, turning a ~1GB serialization file into ~4MB (or 16KB).
    """

    def __init__(self, path: str, device: str = "cpu", capacity_dim: int = 2**20):
        # Default 2**20 (1 Megabyte floats) provides ~4MB disk footprint.
        self.path = Path(path)
        self.device = torch.device(device)
        self.capacity_dim = capacity_dim
        
        self._meta = {
            "steps_charged": 0, 
            "samples_seen": 0, 
            "capacity_dim": capacity_dim,
            "version": "hbm-countsketch-v1"
        }
        
        # The ultimate "pocket dimension" superposition array.
        self._memory = torch.zeros(self.capacity_dim, dtype=torch.float32, device=self.device)

    def is_ready(self, discharge_threshold: int) -> bool:
        """Checks if enough steps have aggregated into the pocket dimension."""
        return self._meta["steps_charged"] >= discharge_threshold

    def status(self) -> str:
        """Returns the compression metrics."""
        return (f"[HGB] Charged Steps: {self._meta['steps_charged']} | "
                f"Samples: {self._meta['samples_seen']} | "
                f"Pocket Dimension Size: {self.capacity_dim} (~{self.capacity_dim * 4 / 1024 / 1024:.2f} MB)")

    def charge(self, model: torch.nn.Module, n_samples: int = 1, scale: float = 1.0):
        """
        Compresses the incoming raw gradients using fractional binding 
        and adds them perfectly into the global memory superposition.
        """
        self._meta["steps_charged"] += 1
        self._meta["samples_seen"] += n_samples
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                flat_grad = (p.grad.detach().flatten() * scale).to(self.device, dtype=torch.float32)
                M = flat_grad.shape[0]
                
                # Deterministic PRNG Seed based on layer name ensures exactly reversible keys
                # WITHOUT needing 1 Billion parameters to be stored manually in VRAM natively.
                seed = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % (2**32)
                rng = torch.Generator(device=self.device)
                rng.manual_seed(seed)
                
                # 1. Generate signature phase (+1, -1)
                signs = torch.randint(0, 2, (M,), generator=rng, device=self.device, dtype=torch.float32) * 2 - 1
                
                # 2. Map coordinates (Random Indexing Mapping)
                indices = torch.randint(0, self.capacity_dim, (M,), generator=rng, device=self.device)
                
                # 3. Superposition Fold (Fractional Binding accumulation natively onto the fixed memory vector)
                self._memory.index_add_(0, indices, flat_grad * signs)

    def discharge(self, model: torch.nn.Module):
        """
        Unbinds and decompresses the global pocket dimension back into 
        layer-specific Pytorch gradient approximations.
        """
        if self._meta["steps_charged"] == 0:
            return

        for name, p in model.named_parameters():
            if p.requires_grad:
                M = p.numel()
                seed = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % (2**32)
                rng = torch.Generator(device=self.device)
                rng.manual_seed(seed)
                
                signs = torch.randint(0, 2, (M,), generator=rng, device=self.device, dtype=torch.float32) * 2 - 1
                indices = torch.randint(0, self.capacity_dim, (M,), generator=rng, device=self.device)
                
                # Decompress via reverse interference extraction
                approx_grad = (self._memory[indices] * signs).view(p.shape).to(p.device, dtype=p.dtype)
                
                if p.grad is None:
                    p.grad = approx_grad
                else:
                    p.grad += approx_grad

    def save(self):
        """Persists the 4MB representation cleanly onto Disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "memory": self._memory,
            "meta": self._meta,
        }, str(self.path))
        print(f"[HGB] Saved Holographic battery -> {self.path.name}")

    def load(self):
        """Restore the Holographic pocket dimension natively."""
        if self.path.exists():
            checkpoint = torch.load(str(self.path), map_location=self.device)
            self._memory = checkpoint["memory"]
            self._meta = checkpoint["meta"]
            self.capacity_dim = self._meta["capacity_dim"]

    def reset(self):
        """Annihilates the current superposition buffer out natively."""
        self._memory.zero_()
        self._meta["steps_charged"] = 0
        self._meta["samples_seen"] = 0
