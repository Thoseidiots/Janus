"""
Real-valued HRR implementation: RealHolographicMemory and RealPhaseBrainLayer
"""

import torch
import torch.nn as nn
from typing import Optional


class RealHolographicMemory(nn.Module):
    """
    Real-valued Holographic Brain Memory using real-valued Holographic Reduced Representations.
    
    Provides an alternative to complex-valued HRR for enhanced hardware compatibility.
    """
    
    def __init__(self, dim: int = 512):
        """
        Initialize RealHolographicMemory.
        
        Args:
            dim: Dimension of the memory vector
        """
        super().__init__()
        self.dim = dim
        # Initialize memory as real tensor
        self.register_buffer(
            'memory',
            torch.zeros(dim, dtype=torch.float32)
        )
        self.access_count = 0
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Bind two real vectors using circular convolution.
        
        Args:
            a: First vector (real)
            b: Second vector (real)
            
        Returns:
            Bound vector (real)
        """
        # Real circular convolution via FFT
        a_fft = torch.fft.rfft(a, dim=-1)
        b_fft = torch.fft.rfft(b, dim=-1)
        result_fft = a_fft * b_fft
        result = torch.fft.irfft(result_fft, n=self.dim, dim=-1)
        return result
    
    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Unbind two real vectors (approximate inverse of bind).
        
        Args:
            a: First vector (real)
            b: Second vector (real)
            
        Returns:
            Unbound vector (real)
        """
        # For real-valued, unbind is bind with reversed b
        b_reversed = torch.flip(b, dims=[-1])
        return self.bind(a, b_reversed)
    
    def write(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0):
        """
        Write a key-value pair to memory.
        
        Args:
            key: Key vector (real)
            value: Value vector (real)
            strength: Strength of the write operation
        """
        bound = self.bind(key, value)
        self.memory = self.memory + strength * bound
        self.access_count += 1
    
    def read(self, key: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using a key.
        
        Args:
            key: Key vector (real)
            
        Returns:
            Retrieved value (real)
        """
        retrieved = self.unbind(self.memory, key)
        self.access_count += 1
        return retrieved
    
    def get_magnitude(self) -> float:
        """Get the magnitude of the memory vector."""
        return torch.abs(self.memory).mean().item()
    
    def reset(self):
        """Reset memory to zero."""
        self.memory.zero_()
        self.access_count = 0


class RealPhaseBrainLayer(nn.Module):
    """
    Real-valued computational layer that interacts with RealHolographicMemory.
    """
    
    def __init__(self, in_dim: int, memory: RealHolographicMemory, out_dim: Optional[int] = None):
        """
        Initialize RealPhaseBrainLayer.
        
        Args:
            in_dim: Input dimension
            memory: RealHolographicMemory instance to interact with
            out_dim: Output dimension (defaults to memory dimension)
        """
        super().__init__()
        self.in_dim = in_dim
        self.memory = memory
        self.out_dim = out_dim or memory.dim
        
        # Real-valued encoding weights
        self.register_buffer(
            'encoding_weights',
            torch.randn(in_dim, memory.dim) / (in_dim ** 0.5)
        )
        
        # Output projection
        self.output_proj = nn.Linear(memory.dim, self.out_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using real-valued weights.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            
        Returns:
            Encoded vector (batch_size, memory.dim)
        """
        encoded = torch.matmul(x, self.encoding_weights)
        return encoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input, write to memory, read back, and project.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            
        Returns:
            Output tensor (batch_size, out_dim)
        """
        # Encode input
        encoded = self.encode(x)
        
        # Write to memory
        for i in range(encoded.shape[0]):
            self.memory.write(encoded[i], encoded[i], strength=0.1)
        
        # Read from memory
        retrieved = self.memory.read(encoded[0])
        
        # Project to output dimension
        output = self.output_proj(retrieved.unsqueeze(0))
        
        return output
