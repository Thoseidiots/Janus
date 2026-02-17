“””
cognitive_loop.py
─────────────────────────────────────────────────────────────
Wires unified_perception + multimodal_fusion directly into the
OBSERVE → PLAN → PROPOSE → VERIFY → APPLY cognitive loop.

Drop-in for Janus: import and call CognitiveLoop(consciousness, memory)
Perception events automatically update homeostasis valence in real-time.
“””

import time
import threading
import queue
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

# ── Homeostasis valence dimensions (mirrors janus-brain) ──────────────────────

@dataclass
class ValenceState:
pleasure:   float = 0.0   # -1 … +1
arousal:    float = 0.0   # -1 … +1
curiosity:  float = 0.5   #  0 … +1
frustration:float = 0.0   #  0 … +1
confidence: float = 0.5   #  0 … +1
timestamp:  str   = field(default_factory=lambda: datetime.now().isoformat())

```
def clamp(self):
    self.pleasure    = max(-1.0, min(1.0, self.pleasure))
    self.arousal     = max(-1.0, min(1.0, self.arousal))
    self.curiosity   = max( 0.0, min(1.0, self.curiosity))
    self.frustration = max( 0.0, min(1.0, self.frustration))
    self.confidence  = max( 0.0, min(1.0, self.confidence))
    return self

def to_dict(self) -> dict:
    return asdict(self)
```

class ValenceEngine:
“””
Manages real-time valence updates from perception events.
Each perception signal bumps the relevant dimension by a small delta
and then decays all dimensions toward baseline over time.
“””

```
DECAY_RATE  = 0.02   # per second, toward baseline
BASELINE    = ValenceState()

# Maps (event_source, event_type) → {dim: delta}
BUMP_TABLE = {
    ("vision",  "person_entered"):  {"pleasure": +0.15, "arousal": +0.20, "curiosity": +0.10},
    ("vision",  "person_left"):     {"arousal": -0.10},
    ("vision",  "task_complete"):   {"pleasure": +0.25, "curiosity": +0.15, "frustration": -0.10},
    ("vision",  "scene_change"):    {"curiosity": +0.10, "arousal": +0.05},
    ("vision",  "anomaly"):         {"arousal": +0.30, "curiosity": +0.20},
    ("audio",   "wake_word"):       {"arousal": +0.20, "pleasure": +0.10},
    ("audio",   "speech"):          {"arousal": +0.10, "curiosity": +0.05},
    ("tool",    "success"):         {"pleasure": +0.20, "confidence": +0.10, "frustration": -0.15},
    ("tool",    "failure"):         {"frustration": +0.20, "pleasure": -0.10, "confidence": -0.05},
    ("goal",    "subtask_done"):    {"pleasure": +0.15, "curiosity": +0.05},
    ("goal",    "stalled"):         {"frustration": +0.20, "pleasure": -0.10},
    ("goal",    "completed"):       {"pleasure": +0.40, "confidence": +0.20, "frustration": -0.30},
    ("memory",  "surprise"):        {"curiosity": +0.25, "arousal": +0.15},
}

def __init__(self):
    self.state = ValenceState()
    self._lock = threading.Lock()
    self._history: List[ValenceState] = []
    self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
    self._decay_thread.start()

def bump(self, source: str, event_type: str, scale: float = 1.0):
    key = (source, event_type)
    deltas = self.BUMP_TABLE.get(key, {})
    with self._lock:
        for dim, delta in deltas.items():
            cur = getattr(self.state, dim)
            setattr(self.state, dim, cur + delta * scale)
        self.state.clamp()
        self.state.timestamp = datetime.now().isoformat()
        self._history.append(ValenceState(**asdict(self.state)))
        if len(self._history) > 1000:
            self._history = self._history[-500:]

def _decay_loop(self):
    while True:
        time.sleep(1.0)
        with self._lock:
            for dim in ("pleasure", "arousal", "curiosity", "frustration", "confidence"):
                baseline = getattr(self.BASELINE, dim)
                cur      = getattr(self.state, dim)
                setattr(self.state, dim, cur + (baseline - cur) * self.DECAY_RATE)
            self.state.clamp()

def snapshot(self) -> ValenceState:
    with self._lock:
        return ValenceState(**asdict(self.state))

def history(self, last_n: int = 60) -> List[dict]:
    with self._lock:
        return [asdict(s) for s in self._history[-last_n:]]
```

# ── Cognitive loop phases ──────────────────────────────────────────────────────

class LoopPhase(Enum):
OBSERVE  = “OBSERVE”
PLAN     = “PLAN”
PROPOSE  = “PROPOSE”
VERIFY   = “VERIFY”
APPLY    = “APPLY”
SLEEP    = “SLEEP”

@dataclass
class ObserveBundle:
“”“Structured output of the OBSERVE phase — fed into PLAN.”””
timestamp:        str
visual_summary:   Optional[str]
audio_transcript: Optional[str]
entities:         List[str]
events:           List[Dict]
valence_snapshot: Dict
raw_perception:   Dict = field(default_factory=dict)

class CognitiveLoop:
“””
Central cognitive orchestrator.

```
Usage
─────
loop = CognitiveLoop()
loop.attach_perception(unified_perception_instance)
loop.attach_memory(memory_instance)
loop.attach_consciousness(consciousness_instance)
loop.start()
"""

def __init__(self, state_path: str = "persistent_state.json"):
    self.state_path   = Path(state_path)
    self.valence      = ValenceEngine()
    self.phase        = LoopPhase.OBSERVE
    self.running      = False
    self.cycle_count  = 0
    self.event_log: List[dict] = []

    # Pluggable subsystems (set via attach_*)
    self._perception  = None
    self._memory      = None
    self._consciousness = None

    # Queues for inter-phase comms
    self._observe_q   = queue.Queue()
    self._plan_q      = queue.Queue()

    # Load persistent state
    self._state = self._load_state()

# ── Attachment API ─────────────────────────────────────────────────────────
def attach_perception(self, perception):
    self._perception = perception
    # Wire perception callbacks → valence bumps
    if hasattr(perception, 'fusion') and perception.fusion:
        original = getattr(perception.fusion, 'on_event_detected', None)
        def _hooked_event(event):
            self._on_perception_event(event)
            if original:
                original(event)
        perception.fusion.on_event_detected = _hooked_event

    if hasattr(perception, 'voice') and perception.voice:
        original_ww = getattr(perception.voice, 'on_wake_word', None)
        def _hooked_ww():
            self.valence.bump("audio", "wake_word")
            if original_ww:
                original_ww()
        perception.voice.on_wake_word = _hooked_ww

def attach_memory(self, memory):
    self._memory = memory

def attach_consciousness(self, consciousness):
    self._consciousness = consciousness

# ── Lifecycle ──────────────────────────────────────────────────────────────
def start(self):
    self.running = True
    self._loop_thread = threading.Thread(target=self._main_loop, daemon=True)
    self._loop_thread.start()
    print("[CognitiveLoop] Started.")

def stop(self):
    self.running = False
    self._save_state()
    print("[CognitiveLoop] Stopped.")

# ── Main loop ──────────────────────────────────────────────────────────────
def _main_loop(self):
    while self.running:
        try:
            self.cycle_count += 1
            self._log_event("cycle_start", {"cycle": self.cycle_count, "phase": self.phase.value})

            bundle  = self._phase_observe()
            plan    = self._phase_plan(bundle)
            actions = self._phase_propose(plan)
            safe    = self._phase_verify(actions)
            self._phase_apply(safe)

            # Persist valence + state after every cycle
            self._state["valence"]     = self.valence.snapshot().to_dict()
            self._state["cycle_count"] = self.cycle_count
            self._save_state()

            time.sleep(0.5)  # Throttle — tunable

        except Exception as e:
            self._log_event("loop_error", {"error": str(e)})
            time.sleep(2.0)

# ── OBSERVE ────────────────────────────────────────────────────────────────
def _phase_observe(self) -> ObserveBundle:
    self.phase = LoopPhase.OBSERVE

    visual_summary   = None
    audio_transcript = None
    entities: List[str] = []
    events:   List[dict] = []

    # Pull from perception if available
    if self._perception:
        scene = getattr(self._perception, 'last_scene', None)
        if scene and hasattr(scene, 'summary'):
            visual_summary = scene.summary
            entities += getattr(scene, 'objects', [])
            entities = [getattr(e, 'class_name', str(e)) for e in entities]

        voice = getattr(self._perception, 'voice', None)
        if voice and voice.conversation:
            last = voice.conversation[-1]
            audio_transcript = getattr(last, 'text', None)

        fusion = getattr(self._perception, 'fusion', None)
        if fusion and fusion.detected_events:
            for ev in fusion.detected_events[-10:]:
                events.append({
                    "type":       ev.event_type.value,
                    "desc":       ev.description,
                    "confidence": ev.confidence,
                    "ts":         ev.timestamp,
                })

    bundle = ObserveBundle(
        timestamp        = datetime.now().isoformat(),
        visual_summary   = visual_summary,
        audio_transcript = audio_transcript,
        entities         = list(set(entities)),
        events           = events,
        valence_snapshot = self.valence.snapshot().to_dict(),
    )

    # Bind into episodic memory
    if self._memory:
        try:
            self._memory.store({
                "type":    "observe_bundle",
                "content": asdict(bundle),
            })
        except Exception:
            pass

    self._log_event("observe", {
        "entities": bundle.entities,
        "events":   len(bundle.events),
        "visual":   visual_summary,
    })
    return bundle

# ── PLAN ───────────────────────────────────────────────────────────────────
def _phase_plan(self, bundle: ObserveBundle) -> dict:
    self.phase = LoopPhase.PLAN
    plan = {
        "cycle":      self.cycle_count,
        "based_on":   bundle.timestamp,
        "intent":     "maintain_homeostasis",
        "sub_goals":  [],
    }

    # Valence-driven goal injection
    snap = bundle.valence_snapshot
    if snap.get("frustration", 0) > 0.5:
        plan["sub_goals"].append({"id": "reduce_frustration", "priority": 1})
    if snap.get("curiosity", 0) > 0.7:
        plan["sub_goals"].append({"id": "explore_entities", "entities": bundle.entities, "priority": 2})
    if bundle.events:
        plan["sub_goals"].append({"id": "respond_to_events", "events": bundle.events, "priority": 3})

    self._log_event("plan", plan)
    return plan

# ── PROPOSE ────────────────────────────────────────────────────────────────
def _phase_propose(self, plan: dict) -> List[dict]:
    self.phase = LoopPhase.PROPOSE
    actions = []

    for goal in plan.get("sub_goals", []):
        gid = goal.get("id", "")
        if gid == "reduce_frustration":
            actions.append({"tool": "self_reflect", "args": {"prompt": "What is causing frustration?"}, "risk": "low"})
        elif gid == "explore_entities":
            for ent in goal.get("entities", [])[:2]:
                actions.append({"tool": "memory_query", "args": {"query": ent}, "risk": "low"})
        elif gid == "respond_to_events":
            for ev in goal.get("events", [])[:3]:
                actions.append({"tool": "log_event", "args": ev, "risk": "low"})

    self._log_event("propose", {"action_count": len(actions)})
    return actions

# ── VERIFY ─────────────────────────────────────────────────────────────────
def _phase_verify(self, actions: List[dict]) -> List[dict]:
    self.phase = LoopPhase.VERIFY
    safe = []
    for action in actions:
        risk = action.get("risk", "medium")
        # Only auto-approve low-risk; others need explicit clearance
        if risk == "low":
            safe.append(action)
        else:
            self._log_event("verify_blocked", {"action": action})
    self._log_event("verify", {"safe_count": len(safe), "total": len(actions)})
    return safe

# ── APPLY ──────────────────────────────────────────────────────────────────
def _phase_apply(self, actions: List[dict]):
    self.phase = LoopPhase.APPLY
    for action in actions:
        tool = action.get("tool")
        args = action.get("args", {})
        try:
            result = self._dispatch_tool(tool, args)
            self.valence.bump("tool", "success", scale=0.5)
            self._log_event("apply_ok", {"tool": tool, "result": str(result)[:120]})
        except Exception as e:
            self.valence.bump("tool", "failure", scale=0.5)
            self._log_event("apply_fail", {"tool": tool, "error": str(e)})

def _dispatch_tool(self, tool: str, args: dict) -> Any:
    if tool == "self_reflect":
        return f"Reflection: {args.get('prompt','')}"
    elif tool == "memory_query" and self._memory:
        return self._memory.query(args.get("query", ""))
    elif tool == "log_event":
        self._log_event("agent_log", args)
        return "logged"
    return None

# ── Perception event hook ──────────────────────────────────────────────────
def _on_perception_event(self, event):
    """Called by perception callbacks — updates valence immediately."""
    etype = event.event_type.value if hasattr(event, 'event_type') else str(event)
    modality = getattr(event, 'primary_modality', None)
    source = modality.value if modality else "vision"
    self.valence.bump(source, etype)
    self._log_event("perception_event", {"source": source, "type": etype})

# ── Helpers ────────────────────────────────────────────────────────────────
def _log_event(self, kind: str, data: dict):
    entry = {"ts": datetime.now().isoformat(), "kind": kind, "data": data}
    self.event_log.append(entry)
    if len(self.event_log) > 5000:
        self.event_log = self.event_log[-2500:]

def _load_state(self) -> dict:
    if self.state_path.exists():
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            pass
    return {}

def _save_state(self):
    try:
        existing = self._load_state()
        existing["valence"]     = self.valence.snapshot().to_dict()
        existing["cycle_count"] = self.cycle_count
        existing["last_saved"]  = datetime.now().isoformat()
        self.state_path.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        print(f"[CognitiveLoop] State save failed: {e}")

def get_status(self) -> dict:
    return {
        "phase":       self.phase.value,
        "cycle":       self.cycle_count,
        "valence":     self.valence.snapshot().to_dict(),
        "event_count": len(self.event_log),
        "running":     self.running,
    }
```