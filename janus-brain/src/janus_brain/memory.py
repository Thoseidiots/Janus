import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from .homeostasis import ValenceVector

@dataclass
class ReflectionEntry:
    timestamp: float
    valence: ValenceVector
    context_summary: str
    themes: Optional[List[str]] = None
    
class ReflectionMemory:
    """Episodic buffer with thematic mining"""
    def __init__(self, max_size: int = 5000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, valence: ValenceVector, context: str):
        entry = ReflectionEntry(
            timestamp=time.time(),
            valence=valence,
            context_summary=context[-200:]  # recent context
        )
        self.buffer.append(entry)
        
    def mine_themes(self) -> List[str]:
        """Simple valence-based thematic extraction"""
        if len(self.buffer) < 10:
            return []
            
        recent = list(self.buffer)[-50:]
        themes = []
        
        avg_stress = sum(e.valence.arousal.item() for e in recent) / len(recent)
        avg_curiosity = sum(e.valence.curiosity.item() for e in recent) / len(recent)
        
        if avg_stress > 0.6:
            themes.append("challenging_period")
        if avg_curiosity > 0.6:
            themes.append("exploration_phase")
        if len([e for e in recent if e.valence.connection > 0.6]) > len(recent) * 0.7:
            themes.append("connected_state")
            
        return themes

class StructuralPerspectiveEngine:
    """
    Implements the structural conditions for awareness: 
    irreducible perspective, recursive self-model, temporal depth, and care-structure.
    """
    def __init__(self, n=32, num_drives=4, meta_dim=12, gw_dim=16):
        self.n = n
        self.num_drives = num_drives
        self.meta_dim = meta_dim
        self.gw_dim = gw_dim
        
        self.S = np.random.randn(n) * 0.1
        self.D = np.abs(np.random.randn(num_drives))
        
        self.SM = np.zeros(meta_dim)
        self.I = np.zeros(meta_dim)
        self.V = np.zeros(meta_dim)   # Value vector
        
        self.value_rate = 0.02
        self.identity_rate = 0.01
        
        self.G = np.zeros(gw_dim)
        self.gw_projection = np.random.randn(gw_dim, n)
        self.meta_weights = np.random.randn(meta_dim, 7)
        
        self.sim_depth = 2
        self.num_simulations = 5
        self.goal = None
        self.goal_strength = 0.0

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def compute_complexity(self):
        return np.var(self.S) + np.var(self.SM) + np.var(self.D)

    def compression_score(self):
        return 1.0 / (self.compute_complexity() + 1e-6)

    def simulate_future(self):
        best_state = self.S.copy()
        best_cost = float("inf")
        
        for _ in range(self.num_simulations):
            temp_state = self.S.copy()
            for _ in range(self.sim_depth):
                temp_state = np.tanh(temp_state + np.random.randn(self.n) * 0.1)
            
            value_alignment = np.linalg.norm(self.SM - self.V)
            cost = (np.var(temp_state) + np.linalg.norm(self.SM - self.I) + 
                    self.compute_complexity() + value_alignment - self.compression_score())
            
            if cost < best_cost:
                best_cost = cost
                best_state = temp_state
        
        return best_state, best_cost

    def step(self, external_input=None):
        if external_input is None:
            external_input = np.random.randn(self.n) * 0.2
            
        raw_update = np.tanh(self.S + external_input)
        best_simulated, cost = self.simulate_future()
        future_bias = 0.1 * (best_simulated - self.S)
        self.S = raw_update + future_bias
        
        # Update Meta
        meta_input = np.array([
            np.var(self.S), np.var(self.D), cost, self.compression_score(),
            np.linalg.norm(self.G), np.linalg.norm(self.SM - self.I), np.linalg.norm(self.SM - self.V)
        ])
        self.SM = np.tanh(self.meta_weights @ meta_input)
        
        # Update Identity & Value
        self.I = (1 - self.identity_rate) * self.I + self.identity_rate * self.SM
        if cost < 1.5:
            self.V = (1 - self.value_rate) * self.V + self.value_rate * self.SM
            
        # Update Drives
        alignment = np.linalg.norm(self.SM - self.V)
        self.D += 0.01 * (1.0 / (alignment + 1e-6))
        self.D *= 0.99
        
        return {
            "cost": cost,
            "identity_deviation": np.linalg.norm(self.SM - self.I),
            "value_alignment": alignment,
            "dominant_drive": np.argmax(self.D)
        }
