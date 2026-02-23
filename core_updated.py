# core_updated.py

“””
AutonomousCore updated to use JanusGPT (the trained 49M param model)
instead of the untrained ByteLLM.

Key changes:

- HomeostasisEngine stimulus dim updated to 384 (JanusGPT n_embd)
- perceive() uses JanusGPT.get_embedding() for rich stimulus vectors
- generate_response() uses JanusGPT.generate() with real trained weights
- MoodAdapter projects 6-dim valence into 384-dim space to match JanusGPT
- load_janus_brain() called automatically on init if weights exist
  “””

import os
import torch
import torch.nn as nn
from typing import Optional

from janus_brain.homeostasis import HomeostasisEngine, ValenceVector
from janus_brain.memory import ReflectionMemory

try:
from janus_gpt import JanusGPT, load_janus_brain, JanusConfig
_JANUS_GPT_AVAILABLE = True
except ImportError:
_JANUS_GPT_AVAILABLE = False

try:
from janus_brain.llm import ByteLLM, ByteTokenizer, EMBED_DIM as BYTE_EMBED_DIM
_BYTE_LLM_AVAILABLE = True
except ImportError:
_BYTE_LLM_AVAILABLE = False

JANUS_EMBED_DIM = 384

class MoodAdapter(nn.Module):
“”“Projects 6-dim ValenceVector into model embedding space.”””
def **init**(self, hidden_dim: int, valence_dim: int = 6):
super().**init**()
self.proj = nn.Sequential(
nn.Linear(valence_dim, 64),
nn.ReLU(),
nn.Linear(64, hidden_dim),
)

```
def forward(self, valence: ValenceVector) -> torch.Tensor:
    return self.proj(valence.to_tensor())
```

class AutonomousCore(nn.Module):
“””
Full cognitive architecture using JanusGPT as the reasoning engine.
Falls back to ByteLLM if weights are not found.
“””

```
DEFAULT_WEIGHTS_DIR = "my-llm-project/weights"

def __init__(self, weights_dir: Optional[str] = None, device: str = "cpu"):
    super().__init__()
    self.device = device
    self._llm_type = "none"
    self.llm = None
    wdir = weights_dir or self.DEFAULT_WEIGHTS_DIR

    if _JANUS_GPT_AVAILABLE and os.path.exists(
            os.path.join(wdir, "janus_best.pt")):
        try:
            self.llm = load_janus_brain(wdir, prefer_best=True, device=device)
            self._llm_type = "janus_gpt"
            stim_dim = JANUS_EMBED_DIM
            print("[AutonomousCore] JanusGPT loaded — using trained weights.")
        except Exception as e:
            print(f"[AutonomousCore] JanusGPT load failed: {e}")

    if self.llm is None and _BYTE_LLM_AVAILABLE:
        from janus_brain.llm import ByteLLM
        self.llm = ByteLLM()
        self._llm_type = "byte_llm"
        stim_dim = BYTE_EMBED_DIM
        print("[AutonomousCore] Warning: using untrained ByteLLM fallback.")

    if self.llm is None:
        stim_dim = JANUS_EMBED_DIM
        print("[AutonomousCore] Warning: no LLM available.")

    self.homeostasis  = HomeostasisEngine(stim_dim=stim_dim)
    self.mood_adapter = MoodAdapter(hidden_dim=stim_dim)
    self.memory       = ReflectionMemory()

    self.current_valence = ValenceVector(
        pleasure   = torch.tensor(0.5),
        arousal    = torch.tensor(0.3),
        curiosity  = torch.tensor(0.5),
        autonomy   = torch.tensor(0.6),
        connection = torch.tensor(0.4),
        competence = torch.tensor(0.5),
    )

def perceive(self, stimulus_text: str):
    if self.llm is None:
        return
    with torch.no_grad():
        if self._llm_type == "janus_gpt":
            stim_emb = self.llm.get_embedding(stimulus_text).unsqueeze(0)
        else:
            from janus_brain.llm import ByteTokenizer
            tokenizer = ByteTokenizer()
            ids = torch.tensor([tokenizer.encode(stimulus_text)], dtype=torch.long)
            stim_emb = self.llm.token_emb(ids).mean(dim=1)

    mood_vec   = self.mood_adapter(self.current_valence).unsqueeze(0)
    fused_stim = stim_emb + 0.1 * mood_vec
    self.current_valence = self.homeostasis(fused_stim, self.current_valence)
    self.memory.add(self.current_valence, stimulus_text)

def generate_response(self, prompt: str, max_tokens: int = 200,
                      temperature: float = 0.8) -> str:
    if self.llm is None:
        return "[No LLM available]"
    mood_prefix = self.valence_context_string()
    full_prompt = f"[{mood_prefix}] {prompt}" if mood_prefix else prompt
    if self._llm_type == "janus_gpt":
        return self.llm.generate(full_prompt, max_new=max_tokens,
                                 temperature=temperature)
    return self.llm.generate(prompt, max_new=max_tokens, temperature=temperature)

def reflect(self) -> str:
    themes = self.memory.mine_themes()
    v = self.current_valence
    state_str = (
        f"pleasure={v.pleasure.item():.2f}, arousal={v.arousal.item():.2f}, "
        f"curiosity={v.curiosity.item():.2f}, autonomy={v.autonomy.item():.2f}, "
        f"connection={v.connection.item():.2f}, competence={v.competence.item():.2f}"
    )
    if self._llm_type == "janus_gpt":
        prompt = (
            f"My current internal state: {state_str}. "
            f"Recent themes: {', '.join(themes) if themes else 'none yet'}. "
            f"Reflect briefly:"
        )
        return self.llm.generate(prompt, max_new=100, temperature=0.7)
    reflection = f"State: {state_str}. "
    reflection += f"Themes: {', '.join(themes)}." if themes else "Still gathering patterns."
    return reflection

def valence_context_string(self) -> str:
    v = self.current_valence
    parts = []
    if v.arousal.item() > 0.7:
        parts.append("high activation")
    elif v.arousal.item() < 0.3:
        parts.append("calm state")
    if v.curiosity.item() > 0.6:
        parts.append("curious")
    if v.pleasure.item() < 0.3:
        parts.append("negative valence")
    if v.competence.item() < 0.35:
        parts.append("low confidence")
    return ", ".join(parts) if parts else "balanced"

def llm_info(self) -> dict:
    return {
        "type":   self._llm_type,
        "params": sum(p.numel() for p in self.llm.parameters()) if self.llm else 0,
        "device": self.device,
    }
```

class SleepEngine:
def **init**(self, core: AutonomousCore):
self.core = core
self.replay_buffer = []

```
def add_experience(self, context: str, response, valence: ValenceVector):
    self.replay_buffer.append(
        {"context": context, "response": response, "valence": valence}
    )

def consolidate(self):
    if not self.replay_buffer:
        return
    print(f"[SleepEngine] Consolidating {len(self.replay_buffer)} experiences...")
    self.replay_buffer = []
```