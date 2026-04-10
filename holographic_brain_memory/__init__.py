"""
Holographic Brain Memory (HBM) Library

A PyTorch-based implementation of a Holographic Brain Memory system,
inspired by Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA).
"""

from .core import HolographicBrainMemory, PhaseBrainLayer
from .real_valued import RealHolographicMemory, RealPhaseBrainLayer
from .spawning import SpawningBrain

__version__ = "0.1.0"
__all__ = [
    "HolographicBrainMemory",
    "PhaseBrainLayer",
    "RealHolographicMemory",
    "RealPhaseBrainLayer",
    "SpawningBrain",
]
