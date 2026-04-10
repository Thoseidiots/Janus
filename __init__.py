import torch
import torch.nn as nn
from typing import List, Optional
from .core import HolographicBrainMemory, PhaseBrainLayer

class SpawningBrain(nn.Module):
    """
    A collection of PhaseBrainLayers sharing a single HolographicBrainMemory.
    
    This class handles 'singularity spawning' where new layers are added 
    dynamically when performance or output magnitude drops.
    """
    def __init__(self, in_dim: int, memory_dim: int = 1024, initial_layers: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.memory = HolographicBrainMemory(dim=memory_dim)
        # Store layers in a ModuleList for proper parameter registration
        self.layers = nn.ModuleList([
            PhaseBrainLayer(in_dim, self.memory) for _ in range(initial_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through all current layers and aggregate their results.
        """
        outputs = []
        for layer in self.layers:
            out = layer(x)
            outputs.append(out)
            
        # Aggregate all 'neuron' outputs (e.g., mean)
        return torch.stack(outputs).mean(dim=0)

    def check_and_spawn(self, x: torch.Tensor, threshold: float = 0.05) -> bool:
        """
        Monitor layer outputs and spawn new 'neurons' if magnitude is too low.
        
        Args:
            x: Current input tensor.
            threshold: Minimum average magnitude before spawning.
            
        Returns:
            True if a new neuron was spawned.
        """
        spawned = False
        with torch.no_grad():
            # Check the last layer's output (or aggregate)
            last_out = self.layers[-1](x)
            avg_mag = torch.abs(last_out).mean().item()
            
            if avg_mag < threshold:
                # Singularity detected! Spawn new neuron
                new_layer = PhaseBrainLayer(self.in_dim, self.memory)
                
                # Perturb the new phase key based on the previous one
                old_theta = self.layers[-1].theta
                new_layer.theta.data = old_theta + torch.randn_like(old_theta) * 0.2
                
                # Inject a seed signal to 'boot' the new neuron
                seed_signal = torch.randn(self.memory.dim) * 0.01
                self.memory.write(seed_signal, new_layer.theta)
                
                # Add to the brain
                self.layers.append(new_layer)
                spawned = True
                print(f"🧠 Singularity detected (mag={avg_mag:.4f}) → new neuron spawned. Total: {len(self.layers)}")
                
        return spawned

    def get_memory_footprint(self) -> int:
        """
        Calculate the byte footprint of the holographic memory.
        """
        # Complex float32: 8 bytes per element (4 real, 4 imag)
        return self.memory.memory.numel() * 8
