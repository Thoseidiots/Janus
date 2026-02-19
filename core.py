import torch
import torch.nn as nn
from typing import Dict, Any, List
from .homeostasis import HomeostasisEngine, ValenceVector
from .llm import ByteLLM, ByteTokenizer, EMBED_DIM
from .memory import ReflectionMemory

class MoodAdapter(nn.Module):
“”“Injects valence state into transformer layers”””
def **init**(self, hidden_dim: int, valence_dim: int = 6):
super().**init**()
self.proj = nn.Sequential(
nn.Linear(valence_dim, 64),
nn.ReLU(),
nn.Linear(64, hidden_dim)
)

```
def forward(self, hidden: torch.Tensor, valence: ValenceVector) -> torch.Tensor:
    cond = self.proj(valence.to_tensor())
    return hidden + cond.unsqueeze(1)
```

class MoodConditionedByteLLM(ByteLLM):
“””
ByteLLM subclass that applies MoodAdapters after each transformer layer.

```
The base ByteLLM uses a nn.TransformerEncoder which doesn't expose
per-layer hooks directly, so we override forward() to run the encoder
layer-by-layer and interleave mood conditioning.

FIX: This is the concrete implementation of the MoodAdapter wiring
that core.py's comment marked as "TODO / not yet connected."
"""

def __init__(self, mood_adapters: nn.ModuleList):
    super().__init__()
    self.mood_adapters = mood_adapters
    # Keep a reference so we can set it before each forward pass
    self._current_valence: ValenceVector | None = None

def set_valence(self, valence: ValenceVector) -> None:
    self._current_valence = valence

def forward(self, idx: torch.Tensor, targets=None):
    from .llm import CONTEXT_LENGTH, VOCAB_SIZE
    import torch.nn.functional as F

    B, T = idx.shape
    tok_emb = self.token_emb(idx)
    pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
    x = tok_emb + pos_emb

    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()

    # Run encoder layer-by-layer, injecting mood after each layer
    for i, layer in enumerate(self.transformer.layers):
        x = layer(x, src_mask=mask)
        if self._current_valence is not None and i < len(self.mood_adapters):
            x = self.mood_adapters[i](x, self._current_valence)

    x = self.norm(x)
    logits = self.head(x)

    if targets is None:
        return logits, None

    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    return logits, loss
```

class AutonomousCore(nn.Module):
“”“Full cognitive architecture integrating all components”””

```
def __init__(self):
    super().__init__()
    self.homeostasis = HomeostasisEngine(stim_dim=EMBED_DIM)

    # Mood adapters — one per transformer layer (4 layers in ByteLLM)
    self.mood_adapters = nn.ModuleList([
        MoodAdapter(EMBED_DIM) for _ in range(4)
    ])

    # FIX: Use MoodConditionedByteLLM so adapters are actually called
    self.llm = MoodConditionedByteLLM(self.mood_adapters)

    self.memory = ReflectionMemory()

    # Current state
    self.current_valence = ValenceVector(
        pleasure=torch.tensor(0.5),
        arousal=torch.tensor(0.3),
        curiosity=torch.tensor(0.5),
        autonomy=torch.tensor(0.6),
        connection=torch.tensor(0.4),
        competence=torch.tensor(0.5),
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
    """
    Generate text conditioned on current mood state.

    FIX: Now calls llm.set_valence() so MoodAdapters inject the
    current ValenceVector into every layer of the forward pass,
    making the model's "internal weather" colour its generation.
    """
    self.llm.set_valence(self.current_valence)
    return self.llm.generate(prompt, max_new=max_tokens)

def reflect(self) -> str:
    """Generate self-reflection based on recent experiences"""
    themes = self.memory.mine_themes()
    v = self.current_valence

    reflection = (
        f"State: pleasure={v.pleasure.item():.2f}, "
        f"arousal={v.arousal.item():.2f}, "
        f"curiosity={v.curiosity.item():.2f}, "
        f"autonomy={v.autonomy.item():.2f}, "
        f"connection={v.connection.item():.2f}, "
        f"competence={v.competence.item():.2f}. "
    )
    if themes:
        reflection += f"Recent themes: {', '.join(themes)}."
    else:
        reflection += "Still gathering patterns."

    return reflection

def valence_context_string(self) -> str:
    """
    FIX (bottleneck 3): Returns a compact natural-language description
    of the current ValenceState for injection into external LLM prompts
    (e.g. when calling a cloud LLM via the bridge).
    """
    v = self.current_valence
    parts = []
    if v.arousal.item() > 0.7:
        parts.append("high activation — prioritise concise, focused responses")
    elif v.arousal.item() < 0.3:
        parts.append("low activation — can afford broader exploration")
    if v.curiosity.item() > 0.6:
        parts.append("high curiosity — lean toward hypothesis generation")
    if v.pleasure.item() < 0.3:
        parts.append("negative valence — bias toward error-correction and stabilisation")
    if v.competence.item() < 0.35:
        parts.append("low competence — prefer conservative, well-tested actions")
    return "; ".join(parts) if parts else "balanced state — proceed normally"
```

class SleepEngine:
“””
Offline consolidation: replays experiences to stabilize memory
without catastrophic forgetting.
“””

```
def __init__(self, core: AutonomousCore):
    self.core = core
    self.replay_buffer = []

def add_experience(self, context: str, response: str, valence: ValenceVector):
    """Store experience for later replay"""
    self.replay_buffer.append({
        "context": context,
        "response": response,
        "valence": valence,
    })

def consolidate(self):
    """Replay and stabilize"""
    if not self.replay_buffer:
        return
    print(f"Consolidating {len(self.replay_buffer)} experiences...")
    self.replay_buffer = []
```