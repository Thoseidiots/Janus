“””
world_model.py
────────────────────────────────────────────────────────────
Grounded causal world-model for Janus.
During OBSERVE: records (state, action, next_state) transitions.
During SLEEP:   replays + predicts trajectories to improve PLAN.
No external API — lightweight local learning via tabular + simple
gradient-free model (expandable to a tiny neural net).
“””

import json
import math
import time
import random
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# ── State representation ───────────────────────────────────────────────────────

@dataclass
class WorldState:
“”“Snapshot of the world at a point in time.”””
snapshot_id:  str
timestamp:    str
entities:     List[str]          # detected objects / people
scene_type:   str
people_count: int
valence:      Dict[str, float]
active_goals: List[str]
last_action:  Optional[str]
metadata:     Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transition:
“””(state_t, action, state_t+1) triple for world-model training.”””
transition_id: str
state:         WorldState
action:        str
next_state:    WorldState
reward:        float         # derived from valence delta
timestamp:     str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Prediction:
“”“Model’s prediction for next state given current state + action.”””
predicted_entities:  List[str]
predicted_scene:     str
predicted_valence:   Dict[str, float]
confidence:          float
basis:               str         # which training examples supported this

# ── Causal model (tabular + frequency-based) ───────────────────────────────────

class TabularCausalModel:
“””
Lightweight tabular model: for each (scene_type, action) pair,
tracks how often each entity / scene transition occurs.
This is the seed; swap in a small neural net when ready.
“””

```
def __init__(self):
    # (scene, action) → Counter of next_scene
    self._scene_transitions: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # (scene, action) → Counter of entities that appeared
    self._entity_freq: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # (scene, action) → list of valence deltas per dimension
    self._valence_deltas: Dict[Tuple, Dict[str, List[float]]] = \
        defaultdict(lambda: defaultdict(list))
    self._n_samples: Dict[Tuple, int] = defaultdict(int)

def record(self, t: Transition):
    key = (t.state.scene_type, t.action)
    self._n_samples[key] += 1

    # Scene transition
    self._scene_transitions[key][t.next_state.scene_type] += 1

    # Entity frequency
    for ent in t.next_state.entities:
        self._entity_freq[key][ent] += 1

    # Valence delta
    for dim, val in t.next_state.valence.items():
        prev = t.state.valence.get(dim, 0.0)
        self._valence_deltas[key][dim].append(val - prev)

def predict(self, state: WorldState, action: str) -> Prediction:
    key = (state.scene_type, action)
    n   = self._n_samples.get(key, 0)

    if n == 0:
        return Prediction(
            predicted_entities  = state.entities,
            predicted_scene     = state.scene_type,
            predicted_valence   = dict(state.valence),
            confidence          = 0.1,
            basis               = "no_data",
        )

    # Most likely next scene
    sc_counter = self._scene_transitions.get(key, {})
    predicted_scene = max(sc_counter, key=sc_counter.get) if sc_counter else state.scene_type

    # Entities that appeared in >50% of transitions
    ent_counter = self._entity_freq.get(key, {})
    predicted_entities = [e for e, c in ent_counter.items() if c / n > 0.5]

    # Expected valence delta
    predicted_valence = dict(state.valence)
    for dim, deltas in self._valence_deltas.get(key, {}).items():
        avg_delta = sum(deltas) / len(deltas)
        predicted_valence[dim] = max(-1.0, min(1.0, predicted_valence.get(dim, 0) + avg_delta))

    confidence = min(1.0, math.log1p(n) / 5)   # saturates at ~n=148

    return Prediction(
        predicted_entities  = predicted_entities,
        predicted_scene     = predicted_scene,
        predicted_valence   = predicted_valence,
        confidence          = round(confidence, 3),
        basis               = f"{n}_samples",
    )

def to_dict(self) -> dict:
    return {
        "n_samples": dict(self._n_samples),
        "scene_transitions": {str(k): dict(v) for k, v in self._scene_transitions.items()},
    }
```

# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
“”“Circular buffer of Transition records used during sleep replay.”””

```
def __init__(self, maxlen: int = 5000):
    self._buffer: List[Transition] = []
    self._maxlen = maxlen

def push(self, t: Transition):
    self._buffer.append(t)
    if len(self._buffer) > self._maxlen:
        self._buffer = self._buffer[-self._maxlen:]

def sample(self, n: int) -> List[Transition]:
    n = min(n, len(self._buffer))
    return random.sample(self._buffer, n)

def all(self) -> List[Transition]:
    return list(self._buffer)

def __len__(self):
    return len(self._buffer)
```

# ── World model learner ────────────────────────────────────────────────────────

class WorldModelLearner:
“””
Integrates with the cognitive loop:
• OBSERVE phase  → record_transition()
• PLAN phase     → predict() to estimate action outcomes
• SLEEP phase    → offline_replay() to improve model

```
Usage
─────
wm = WorldModelLearner()
wm.record_transition(state_before, action_taken, state_after)
prediction = wm.predict(current_state, candidate_action)
wm.offline_replay(n_steps=200)   # call during sleep
"""

PERSIST_PATH = Path("world_model.json")
TRANSITION_LOG = Path("transitions.jsonl")

def __init__(self):
    self.model  = TabularCausalModel()
    self.buffer = ReplayBuffer()
    self._lock  = threading.Lock()
    self._load()
    print(f"[WorldModel] Ready. Transitions: {len(self.buffer)}")

# ── Online recording ───────────────────────────────────────────────────────
def record_transition(self, state: WorldState, action: str, next_state: WorldState):
    reward = self._compute_reward(state, next_state)
    t = Transition(
        transition_id = f"tr_{int(time.time()*1000)}",
        state         = state,
        action        = action,
        next_state    = next_state,
        reward        = reward,
    )
    with self._lock:
        self.model.record(t)
        self.buffer.push(t)
    self._log_transition(t)

def _compute_reward(self, s: WorldState, ns: WorldState) -> float:
    """Simple reward: sum of valence improvements weighted by dimension."""
    weights = {"pleasure": 1.0, "curiosity": 0.5, "confidence": 0.5,
               "arousal": 0.2, "frustration": -1.0}
    delta = 0.0
    for dim, w in weights.items():
        delta += w * (ns.valence.get(dim, 0) - s.valence.get(dim, 0))
    return round(delta, 4)

# ── Prediction ─────────────────────────────────────────────────────────────
def predict(self, state: WorldState, action: str) -> Prediction:
    with self._lock:
        return self.model.predict(state, action)

def rank_actions(self, state: WorldState, candidates: List[str]) -> List[Tuple[str, Prediction]]:
    """Rank candidate actions by predicted valence outcome."""
    results = []
    for action in candidates:
        pred = self.predict(state, action)
        score = (pred.predicted_valence.get("pleasure", 0)
                 - pred.predicted_valence.get("frustration", 0)
                 + 0.3 * pred.predicted_valence.get("curiosity", 0))
        results.append((score, action, pred))
    results.sort(reverse=True)
    return [(a, p) for _, a, p in results]

# ── Sleep-phase offline replay ─────────────────────────────────────────────
def offline_replay(self, n_steps: int = 200) -> dict:
    """
    Re-train on random transitions from the replay buffer.
    Also runs counterfactual simulation: what if a different action was taken?
    Returns a brief report.
    """
    before_keys = len(self.model._n_samples)
    sample = self.buffer.sample(n_steps)

    with self._lock:
        for t in sample:
            self.model.record(t)

            # Counterfactual: simulate an alternative action
            alt_actions = ["observe_more", "speak", "do_nothing", "query_memory"]
            alt_action  = random.choice([a for a in alt_actions if a != t.action])
            alt_pred    = self.model.predict(t.state, alt_action)
            # If counterfactual prediction is significantly better, log it
            actual_reward  = t.reward
            counter_reward = (alt_pred.predicted_valence.get("pleasure", 0)
                              - alt_pred.predicted_valence.get("frustration", 0))
            if counter_reward > actual_reward + 0.1:
                pass  # Future: inject into planning as a lesson

    after_keys = len(self.model._n_samples)
    report = {
        "ts":              datetime.now().isoformat(),
        "steps_replayed":  len(sample),
        "new_state_pairs": after_keys - before_keys,
        "buffer_size":     len(self.buffer),
    }
    print(f"[WorldModel] Offline replay: {report}")
    self._save()
    return report

# ── Persistence ───────────────────────────────────────────────────────────
def _log_transition(self, t: Transition):
    entry = {
        "id":      t.transition_id,
        "action":  t.action,
        "reward":  t.reward,
        "scene_before": t.state.scene_type,
        "scene_after":  t.next_state.scene_type,
        "ts":      t.timestamp,
    }
    with self.TRANSITION_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def _save(self):
    try:
        self.PERSIST_PATH.write_text(json.dumps(self.model.to_dict(), indent=2))
    except Exception as e:
        print(f"[WorldModel] Save error: {e}")

def _load(self):
    # Replay existing transitions from log
    if not self.TRANSITION_LOG.exists():
        return
    for line in self.TRANSITION_LOG.read_text().splitlines():
        try:
            d = json.loads(line)
            # Lightweight: re-create minimal state objects from log
            state = WorldState(
                snapshot_id  = "loaded",
                timestamp    = d.get("ts", ""),
                entities     = [],
                scene_type   = d.get("scene_before", "unknown"),
                people_count = 0,
                valence      = {},
                active_goals = [],
                last_action  = None,
            )
            next_state = WorldState(
                snapshot_id  = "loaded_next",
                timestamp    = d.get("ts", ""),
                entities     = [],
                scene_type   = d.get("scene_after", "unknown"),
                people_count = 0,
                valence      = {},
                active_goals = [],
                last_action  = d.get("action"),
            )
            t = Transition(
                transition_id = d.get("id", ""),
                state         = state,
                action        = d.get("action", ""),
                next_state    = next_state,
                reward        = d.get("reward", 0.0),
                timestamp     = d.get("ts", ""),
            )
            self.buffer.push(t)
            self.model.record(t)
        except Exception:
            pass
```