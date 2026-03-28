from .core import HolographicBrainMemory, PhaseBrainLayer
from .spawning import SpawningBrain
from .real_valued import RealHolographicMemory

# InfiniteHolographicBrain lives in solution.py at the Janus root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from solution import InfiniteHolographicBrain
except ImportError:
    InfiniteHolographicBrain = None

__all__ = [
    "HolographicBrainMemory",
    "PhaseBrainLayer",
    "SpawningBrain",
    "RealHolographicMemory",
    "InfiniteHolographicBrain",
]
