import torch
import torch.nn as nn
import torch.fft as fft
from typing import Optional

class RealHolographicMemory(nn.Module):
    """
    Real-valued Holographic Reduced Representations (HRR) Memory.
    
    This version uses real-valued circular convolution, which is more 
    hardware-friendly for certain architectures and avoids complex numbers.
    """
    def __init__(self, dim: int = 1024, decay: float = 0.98):
        super().__init__()
        self.dim = dim
        self.decay = decay
        # Memory is a real-valued vector
        self.memory = nn.Parameter(torch.randn(dim) * 0.01)

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Bind two real-valued vectors using circular convolution.
        
        Args:
            a, b: Real-valued tensors.
            
        Returns:
            Bound real-valued tensor.
        """
        a_f = fft.rfft(a, dim=-1)
        b_f = fft.rfft(b, dim=-1)
        # Pointwise multiplication in frequency domain = circular convolution in time domain
        return fft.irfft(a_f * b_f, n=self.dim, dim=-1)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unbind a vector from a bound representation using the approximate inverse.
        
        Args:
            bound: The bound representation.
            key: The key used during binding.
            
        Returns:
            Unbound real-valued tensor.
        """
        # Approximate inverse in circular convolution is the involution (reverse order)
        # or conjugate in the frequency domain.
        bound_f = fft.rfft(bound, dim=-1)
        key_f = fft.rfft(key, dim=-1)
        # Unbinding is pointwise multiplication with the conjugate
        return fft.irfft(bound_f * torch.conj(key_f), n=self.dim, dim=-1)

    def write(self, signal: torch.Tensor, key: torch.Tensor):
        """
        Superpose a new bound signal into the real-valued memory.
        """
        bound = self.bind(signal, key)
        update_val = bound.mean(dim=0) if bound.dim() > 1 else bound
        
        with torch.no_grad():
            self.memory.data = self.memory.data * self.decay + update_val
            # Normalize for stability
            self.memory.data = self.memory.data / (self.memory.abs().mean() + 1e-8)

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """
        Retrieve a signal from memory using its key.
        """
        return self.unbind(self.memory, key)

class RealPhaseBrainLayer(nn.Module):
    """
    Real-valued 'neuron' or layer that interacts with a RealHolographicMemory.
    """
    def __init__(self, in_dim: int, memory: RealHolographicMemory):
        super().__init__()
        self.in_dim = in_dim
        self.memory = memory
        self.encoder = nn.Linear(in_dim, memory.dim, bias=False)
        # In real HRR, keys are typically random vectors with unit-like norm
        self.key = nn.Parameter(torch.randn(memory.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        self.memory.write(encoded, self.key)
        return self.memory.read(self.key)
