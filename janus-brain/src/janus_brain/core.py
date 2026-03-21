import torch
import torch.nn as nn
import ray
from typing import Dict, Any, List
from .homeostasis import HomeostasisEngine, ValenceVector
from .llm import ByteLLM, ByteTokenizer, EMBED_DIM
from .memory import ReflectionMemory

class MoodAdapter(nn.Module):
    """Injects valence state into transformer layers"""
    def __init__(self, hidden_dim: int, valence_dim: int = 6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(valence_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
    def forward(self, hidden: torch.Tensor, valence: ValenceVector) -> torch.Tensor:
        cond = self.proj(valence.to_tensor())
        return hidden + cond.unsqueeze(1)

@ray.remote
class AutonomousCore:
    """Full cognitive architecture integrating all components"""
    def __init__(self):
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.homeostasis = HomeostasisEngine()
        self.llm = ByteLLM()
        self.memory = ReflectionMemory()
        
        # Mood adapters for each transformer layer - these are local instances within the actor
        self.mood_adapters = nn.ModuleList([
            MoodAdapter(EMBED_DIM) for _ in range(4)
        ])
        
        # Current state
        self.current_valence = ValenceVector(
            pleasure=torch.tensor(0.5),
            arousal=torch.tensor(0.3),
            curiosity=torch.tensor(0.5),
            autonomy=torch.tensor(0.6),
            connection=torch.tensor(0.4),
            competence=torch.tensor(0.5)
        )
        
    def perceive(self, stimulus_text: str):
        """Update cognitive state based on input"""
        tokenizer = ByteTokenizer()
        stim_ids = torch.tensor([tokenizer.encode(stimulus_text)], dtype=torch.long)
        stim_emb = self.llm.token_emb(stim_ids).mean(dim=1)  # pooled
        
        # Update valence
        self.current_valence = self.homeostasis(stim_emb, self.current_valence)
        
        # Store in episodic memory
        self.memory.add(self.current_valence, stimulus_text)
        
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text conditioned on current mood state"""
        # In a real implementation, the mood_adapters would be hooked into the transformer's forward pass.
        # For this integration, we use the base generate method.
        return self.llm.generate(prompt, max_new=max_tokens)
        
    def reflect(self) -> str:
        """Generate self-reflection based on recent experiences"""
        themes = self.memory.mine_themes()
        v = self.current_valence
        
        reflection = f"State: pleasure={v.pleasure.item():.2f}, curiosity={v.curiosity.item():.2f}. "
        if themes:
            reflection += f"Recent themes: {', '.join(themes)}."
        else:
            reflection += "Still gathering patterns."
            
        return reflection

class SleepEngine:
    """
    Offline consolidation: replays experiences to stabilize memory
    without catastrophic forgetting.
    """
    def __init__(self, core: 'AutonomousCore'): # Use forward reference for type hinting
        self.core = core
        self.replay_buffer = []
        
    def add_experience(self, context: str, response: str, valence: ValenceVector):
        """Store experience for later replay"""
        self.replay_buffer.append({
            'context': context,
            'response': response,
            'valence': valence
        })
        
    def consolidate(self):
        """Replay and stabilize"""
        if not self.replay_buffer:
            return
        # In a real system, this would involve a training loop over the replay buffer
        print(f"Consolidating {len(self.replay_buffer)} experiences...")
        self.replay_buffer = []
