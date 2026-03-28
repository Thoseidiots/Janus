import torch
import torch.nn as nn
import torch.fft as fft
from typing import Optional, Union

class HolographicBrainMemory(nn.Module):
    """
    Holographic Reduced Representations (HRR) Memory Core.
    
    This module implements a fixed-size holographic memory buffer that can store
    multiple overlapping logical items using circular convolution and superposition.
    """
    def __init__(self, dim: int = 1024, decay: float = 0.98, use_proj: bool = False):
        super().__init__()
        self.dim = dim
        self.decay = decay
        # Memory is stored as a complex vector to leverage phase-based binding
        self.memory = nn.Parameter(torch.zeros(dim, dtype=torch.cfloat))
        
        # Optional: learned cleanup / projection for better capacity
        if use_proj:
            self.proj = nn.Linear(dim, dim, bias=False)
            # Initialize as identity to preserve information initially
            nn.init.eye_(self.proj.weight)
        else:
            self.register_parameter('proj', None)

    def bind(self, signal: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Bind a signal with a phase key using circular convolution via FFT.
        
        Args:
            signal: Input tensor (real or complex).
            theta: Phase vector (real, in radians).
            
        Returns:
            Bound complex tensor.
        """
        # Ensure signal is complex for FFT operations
        if not signal.is_complex():
            signal = signal.to(torch.cfloat)
            
        sig_f = fft.fft(signal, dim=-1)
        # Create phase vector in the frequency domain
        phase_vec = torch.exp(1j * theta)
        phase_f = fft.fft(phase_vec, dim=-1)
        
        # Circular convolution is pointwise multiplication in frequency domain
        return fft.ifft(sig_f * phase_f, dim=-1)

    def write(self, signal: torch.Tensor, theta: torch.Tensor, persistent: bool = True):
        """
        Superpose new information into the holographic memory.
        
        Args:
            signal: The signal to store.
            theta: The phase key for retrieval.
            persistent: If True, updates the memory parameter.
        """
        bound = self.bind(signal, theta)
        
        # Aggregate across batch if signal has batch dimension
        update_val = bound.mean(dim=0) if bound.dim() > 1 else bound
        
        if persistent:
            with torch.no_grad():
                # Apply decay and add new information
                self.memory.data = self.memory.data * self.decay + update_val
                
                # Normalize to prevent magnitude explosion and restore linear capacity
                # Using a more stable normalization (RMS-like)
                norm = torch.sqrt(torch.mean(self.memory.abs()**2)) + 1e-8
                self.memory.data = self.memory.data / norm
                
                # Optional learned projection for cleanup
                if self.proj is not None:
                    # Apply projection to real and imaginary parts separately
                    real_proj = self.proj(self.memory.data.real)
                    imag_proj = self.proj(self.memory.data.imag)
                    self.memory.data = torch.complex(real_proj, imag_proj)
        else:
            # For non-persistent writes (e.g., temporary scratchpad)
            return self.memory * self.decay + update_val

    def read(self, theta: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Unbind information from memory using an inverse phase key.
        
        Args:
            theta: The phase key used during writing.
            batch_size: If provided, expands the result to a batch.
            
        Returns:
            The retrieved real-valued signal.
        """
        # Create the phase key used during binding
        phase_vec = torch.exp(1j * theta)
        phase_f = fft.fft(phase_vec, dim=-1)
        mem_f = fft.fft(self.memory, dim=-1)
        
        # Unbinding (correlation) is pointwise multiplication with the conjugate in freq domain
        unbound = fft.ifft(mem_f * torch.conj(phase_f), dim=-1)
        result = unbound.real
        
        if batch_size is not None:
            return result.unsqueeze(0).expand(batch_size, -1)
        return result

    def compress_bytes(self, target_bytes: int) -> torch.Tensor:
        """
        Force-fit the memory into a specific byte budget via quantization and folding.
        
        Args:
            target_bytes: Maximum allowed bytes.
            
        Returns:
            Quantized and potentially folded byte tensor.
        """
        # 1. Quantize to int8 (1 byte per element)
        scale = self.memory.abs().max() + 1e-8
        # Store real and imaginary separately (2 bytes per complex number)
        real_q = torch.round(self.memory.real / scale * 127).clamp(-128, 127).to(torch.int8)
        imag_q = torch.round(self.memory.imag / scale * 127).clamp(-128, 127).to(torch.int8)
        
        packed = torch.cat([real_q, imag_q])
        current_bytes = packed.numel()
        
        if current_bytes > target_bytes:
            # 2. Fold/alias dimensions if still too big
            fold = (current_bytes + target_bytes - 1) // target_bytes
            # Truncate to make it divisible by fold for simplicity in this demo
            new_len = (current_bytes // fold) * fold
            packed = packed[:new_len].view(-1, fold).float().mean(dim=1).to(torch.int8)
            
        return packed

class PhaseBrainLayer(nn.Module):
    """
    A 'neuron' or layer that interacts with a shared HolographicBrainMemory.
    """
    def __init__(self, in_dim: int, memory: HolographicBrainMemory):
        super().__init__()
        self.in_dim = in_dim
        self.memory = memory
        # Encoder projects input to the holographic dimension
        self.encoder = nn.Linear(in_dim, memory.dim, bias=False)
        # Unique phase key for this 'neuron'
        self.theta = nn.Parameter(torch.rand(memory.dim) * 2 * torch.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode, write to holographic memory, and read back.
        """
        batch_size = x.shape[0] if x.dim() > 1 else None
        encoded = self.encoder(x)
        # Write the encoded pattern into the shared memory
        self.memory.write(encoded, self.theta)
        # Read back using the same phase key, expanding to batch size
        out = self.memory.read(self.theta, batch_size=batch_size)
        return out
