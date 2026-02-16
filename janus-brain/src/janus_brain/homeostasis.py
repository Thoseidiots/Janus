import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ValenceVector:
    """Multi-dimensional affective state driving behavior modulation"""
    pleasure: torch.Tensor      # positive valence core
    arousal: torch.Tensor       # activation level
    curiosity: torch.Tensor     # information gain drive
    autonomy: torch.Tensor      # perceived control
    connection: torch.Tensor    # relational safety
    competence: torch.Tensor    # mastery / efficacy

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([
            self.pleasure, self.arousal, self.curiosity,
            self.autonomy, self.connection, self.competence
        ], dim=-1)

    @classmethod
    def from_tensor(cls, t: torch.Tensor):
        return cls(*t.unbind(-1))

class HomeostasisEngine(nn.Module):
    """
    Recurrent core that evolves valence over time based on stimuli.
    Drives the 'mood' of the system without explicit emotional labels.
    """
    def __init__(self, stim_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.stim_proj = nn.Linear(stim_dim, hidden_dim)
        self.state_proj = nn.Linear(6, hidden_dim)
        self.recurrent = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.valence_head = nn.Linear(hidden_dim, 6)
        self.set_points = nn.Parameter(torch.tensor([0.6, 0.5, 0.4, 0.6, 0.5, 0.5]))

    def forward(self, stimulus: torch.Tensor, prev_valence: ValenceVector) -> ValenceVector:
        stim_emb = self.stim_proj(stimulus)
        prev_emb = self.state_proj(prev_valence.to_tensor())
        fused = torch.cat([stim_emb, prev_emb], dim=-1)
        hidden = self.recurrent(fused)
        
        delta = torch.tanh(self.valence_head(hidden)) * 0.1
        new_val = prev_valence.to_tensor() + delta
        # Homeostatic pull toward set-points
        pull = (self.set_points - new_val) * 0.05
        new_val = torch.tanh(new_val + pull)
        
        return ValenceVector.from_tensor(new_val)
