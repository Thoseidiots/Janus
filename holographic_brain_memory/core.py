"""
Core HBM components: HolographicBrainMemory and PhaseBrainLayer (complex-valued HRR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HolographicBrainMemory(nn.Module):
    """
    Holographic Brain Memory using complex-valued Holographic Reduced Representations.
    
    Stores logical items in a fixed-size complex vector using holographic superposition.
    """
    
    def __init__(self, dim: int = 1024):
        """
        Initialize HolographicBrainMemory.
        
        Args:
            dim: Dimension of the memory vector
        """
        super().__init__()
        self.dim = dim
        # Initialize memory as complex tensor
        self.register_buffer(
            'memory',
            torch.zeros(dim, dtype=torch.complex64)
        )
        self.access_count = 0
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Bind two vectors using circular convolution (element-wise multiplication in frequency domain).
        
        Args:
            a: First vector (complex)
            b: Second vector (complex)
            
        Returns:
            Bound vector (complex)
        """
        # FFT-based circular convolution
        a_fft = torch.fft.fft(a, dim=-1)
        b_fft = torch.fft.fft(b, dim=-1)
        result_fft = a_fft * b_fft
        result = torch.fft.ifft(result_fft, dim=-1)
        return result
    
    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Unbind two vectors (inverse of bind operation).
        
        Args:
            a: First vector (complex)
            b: Second vector (complex)
            
        Returns:
            Unbound vector (complex)
        """
        # Unbind is bind with conjugate of b
        b_conj = torch.conj(b)
        return self.bind(a, b_conj)
    
    def write(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0):
        """
        Write a key-value pair to memory via holographic superposition.
        
        Args:
            key: Key vector (complex)
            value: Value vector (complex)
            strength: Strength of the write operation
        """
        bound = self.bind(key, value)
        self.memory = self.memory + strength * bound
        self.access_count += 1
    
    def read(self, key: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using a key.
        
        Args:
            key: Key vector (complex)
            
        Returns:
            Retrieved value (complex)
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


class PhaseBrainLayer(nn.Module):
    """
    A computational layer that interacts with HolographicBrainMemory.
    Uses phase-only weights for efficient binding operations.
    """
    
    def __init__(self, in_dim: int, memory: HolographicBrainMemory, out_dim: Optional[int] = None):
        """
        Initialize PhaseBrainLayer.
        
        Args:
            in_dim: Input dimension
            memory: HolographicBrainMemory instance to interact with
            out_dim: Output dimension (defaults to memory dimension)
        """
        super().__init__()
        self.in_dim = in_dim
        self.memory = memory
        self.out_dim = out_dim or memory.dim
        
        # Phase-only encoding weights
        self.register_buffer(
            'phase_weights',
            torch.exp(1j * torch.randn(in_dim, memory.dim))
        )
        
        # Output projection
        self.output_proj = nn.Linear(memory.dim, self.out_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using phase-only weights.

        Args:
            x: Input tensor (batch_size, in_dim) — float or complex

        Returns:
            Encoded complex vector (batch_size, memory.dim)
        """
        # Cast x to complex so matmul with complex phase_weights works
        if not x.is_complex():
            x = x.to(torch.complex64)
        # x: (batch, in_dim) -> (batch, memory.dim)
        encoded = torch.matmul(x.unsqueeze(1), self.phase_weights.unsqueeze(0)).squeeze(1)
        return encoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input, write to memory, read back, and project.

        Args:
            x: Input tensor (batch_size, in_dim)

        Returns:
            Output tensor (batch_size, out_dim)
        """
        # Encode input (returns complex)
        encoded = self.encode(x)

        # Write each item to memory
        for i in range(encoded.shape[0]):
            self.memory.write(encoded[i], encoded[i], strength=0.1)

        # Read back for every item in the batch
        retrieved_list = []
        for i in range(encoded.shape[0]):
            retrieved_list.append(torch.real(self.memory.read(encoded[i])))
        retrieved_real = torch.stack(retrieved_list, dim=0)  # (batch, memory.dim)

        # Project to output dimension
        output = self.output_proj(retrieved_real)
        return output
