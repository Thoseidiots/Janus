"""
SpawningBrain: Dynamic layer creation and singularity spawning
"""

import torch
import torch.nn as nn
from typing import List
from .core import HolographicBrainMemory, PhaseBrainLayer


class SpawningBrain(nn.Module):
    """
    A brain that dynamically spawns new logical neurons into the fixed memory buffer.
    
    Demonstrates how logical capacity can grow without increasing physical memory footprint.
    """
    
    def __init__(self, in_dim: int, memory_dim: int = 1024, initial_layers: int = 1):
        """
        Initialize SpawningBrain.
        
        Args:
            in_dim: Input dimension
            memory_dim: Dimension of shared holographic memory
            initial_layers: Number of initial PhaseBrainLayers
        """
        super().__init__()
        self.in_dim = in_dim
        self.memory_dim = memory_dim
        
        # Shared holographic memory
        self.memory = HolographicBrainMemory(dim=memory_dim)
        
        # Initialize layers
        self.layers = nn.ModuleList([
            PhaseBrainLayer(in_dim=in_dim, memory=self.memory)
            for _ in range(initial_layers)
        ])
        
        self.spawn_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            
        Returns:
            Output from the last layer
        """
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def check_and_spawn(self, x: torch.Tensor, threshold: float = 0.05) -> bool:
        """
        Check if a new neuron should be spawned based on output magnitude.
        
        Args:
            x: Input tensor
            threshold: Magnitude threshold for spawning
            
        Returns:
            True if a new neuron was spawned, False otherwise
        """
        # Get output from last layer
        with torch.no_grad():
            output = self.forward(x)
            magnitude = torch.abs(output).mean().item()
        
        # Spawn if magnitude is below threshold
        if magnitude < threshold:
            new_layer = PhaseBrainLayer(in_dim=self.in_dim, memory=self.memory)
            self.layers.append(new_layer)
            self.spawn_history.append({
                'step': len(self.spawn_history),
                'magnitude': magnitude,
                'total_neurons': len(self.layers)
            })
            return True
        
        return False
    
    def get_memory_footprint(self) -> int:
        """
        Get the physical memory footprint in bytes.
        
        Returns:
            Memory footprint in bytes (fixed, regardless of logical capacity)
        """
        # Complex64 = 8 bytes per element
        return self.memory_dim * 8
    
    def get_logical_capacity(self) -> int:
        """
        Get the logical capacity (number of neurons).
        
        Returns:
            Number of logical neurons
        """
        return len(self.layers)
    
    def get_capacity_ratio(self) -> float:
        """
        Get the ratio of logical capacity to physical memory.
        
        Returns:
            Logical capacity / physical memory footprint
        """
        return self.get_logical_capacity() / self.get_memory_footprint()
