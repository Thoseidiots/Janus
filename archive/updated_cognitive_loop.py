# updated_cognitive_loop.py

“””
Full replacement for cognitive_loop.py.
Extends the OBSERVE → PLAN → PROPOSE → VERIFY → APPLY loop with:

- Video observation hooks in OBSERVE phase
- Automatic video learning when curiosity goal is active
- Periodic summary drain from VideoObserver into memory
- Valence nudges from video content type (engaging tutorial = +curiosity/+pleasure)
  “””

from **future** import annotations

import json
import time
import traceback
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

from janus_brain.core import AutonomousCore
from janus_brain.homeostasis import ValenceVector
from goal_planner import Action, Goal, GoalPlanner, RiskLevel
from video_observer import get_observer, VideoObserver
from updated_enhanced_vision import EnhancedVisionProcessor

_vision = EnhancedVisionProcessor()

# ── Phase enum ────────────────────────────────────────────────────────────────

class Phase(str, Enum):
OBSERVE  = “OBSERVE”
PLAN     = “PLAN”
PROPOSE  = “PROPOSE”
VERIFY   = “VERIFY”
APPLY    = “APPLY”
IDLE     = “IDLE”

# ── Cycle result ──────────────────────────────────────────────────────────────

@dataclass
class CycleResult:
cycle_id:         int
timestamp:        float
phase_reached:    Phase
goals_generated:  int
actions_proposed: int
actions_approved: int
actions_applied:  int
actions_retried:  int
actions_blocked:  int
valence_before:   Dict[str, float]
valence_after:    Dict[str, float]
events:           List[Dict[str, Any]]
video_summaries:  List[str]

```
def to_json(self) -> str:
    return json.dumps(asdict(self), indent=2)
```

# ── Sandbox dry-run ───────────────────────────────────────────────────────────

class SandboxResult:
def **init**(self, success: bool, output: Any, error: Optional[str] = None):
self.success = success
self.output  = output
self.error   = error

def _dry_run(action: Action, executor: “ToolExecutor”) -> SandboxResult:
try:
if action.tool_name == “write_state_snapshot”:
json.dumps({“test”: True})
return SandboxResult(True, “Simulated write ok”)
if action.tool_name == “run_analysis_pipeline”:
import os
vp = action.parameters.get(“video_path”, “”)
if vp and not os.path.exists(vp):
return SandboxResult(False, None, f”video_path not found: {vp}”)
return SandboxResult(True, “Precondition ok”)
if action.tool_name == “watch_video”:
return SandboxResult(True, “URL launch permitted”)
return SandboxResult(True, “No dry-run; assumed safe”)
except Exception as exc:
return SandboxResult(False, None, str(exc))

# ── Tool executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
def **init**(self, core: AutonomousCore,
hub: Optional[Any] = None,
observer: Optional[VideoObserver] = None):
self.core     = core
self.hub      = hub
self.observer = observer or get_observer()

```
    self._dispatch: Dict[str, Callable[[Action], Any]] = {
        "self_reflect":          self._exec_self_reflect,
        "explore_memory":        self._exec_explore_memory,
        "generate_response":     self._exec_generate_response,
        "perceive_input":        self._exec_perceive_input,
        "consolidate_sleep":     self._exec_consolidate_sleep,
        "write_state_snapshot":  self._exec_write_state_snapshot,
    }

    # Dynamically pull hub-registered tools (including video tools)
    if hub is not None:
        for tool in getattr(hub, "_tools", {}).values():
            name = tool.get("name")
            fn   = tool.get("execute")
            if name and fn and name not in self._dispatch:
                # Wrap hub tool to match Action → result signature
                self._dispatch[name] = lambda a, _fn=fn: _fn(a.parameters)

def register(self, name: str, fn: Callable[[Action], Any]) -> None:
    self._dispatch[name] = fn

def execute(self, action: Action) -> Any:
    handler = self._dispatch.get(action.tool_name)
    if handler is None:
        raise NotImplementedError(f"No executor for '{action.tool_name}'")
    return handler(action)

def _exec_self_reflect(self, a): return self.core.reflect()
def _exec_explore_memory(self, a): return self.core.memory.mine_themes()
def _exec_generate_response(self, a):
    return self.core.generate_response(
        a.parameters.get("prompt", "Continue."),
        a.parameters.get("max_tokens", 150))
def _exec_perceive_input(self, a):
    t = a.parameters.get("text", "")
    if t: self.core.perceive(t)
def _exec_consolidate_sleep(self, a):
    from janus_brain.core import SleepEngine
    SleepEngine(self.core).consolidate()
    return "Consolidation complete"
def _exec_write_state_snapshot(self, a):
    import os
    path = a.parameters.get("path", "persistent_state.json")
    v = self.core.current_valence
    snap = {"timestamp": time.time(), "valence": {
        "pleasure":   v.pleasure.item(), "arousal":    v.arousal.item(),
        "curiosity":  v.curiosity.item(), "autonomy":  v.autonomy.item(),
        "connection": v.connection.item(), "competence": v.competence.item(),
    }}
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    return f"Snapshot → {path}"
```

# ── Verify phase ──────────────────────────────────────────────────────────────

_MEDIUM_URGENCY_THRESHOLD = 0.4

def _verify(actions, goals, executor, log):
goal_map = {g.id: g for g in goals}
approved, blocked = [], []
retried = 0
for action in actions:
parent = goal_map.get(action.parent_goal_id)
priority = parent.priority if parent else 0.0

```
    if action.estimated_risk == RiskLevel.LOW:
        approved.append(action)
        log("verify_approved", {"tool": action.tool_name, "reason": "low_risk"})

    elif action.estimated_risk == RiskLevel.MEDIUM:
        if priority >= _MEDIUM_URGENCY_THRESHOLD:
            approved.append(action)
            log("verify_approved", {"tool": action.tool_name, "reason": "sufficient_urgency"})
        else:
            blocked.append(action)
            log("verify_blocked", {"tool": action.tool_name, "reason": "insufficient_urgency"})

    elif action.estimated_risk == RiskLevel.HIGH:
        dry = _dry_run(action, executor)
        if dry.success:
            approved.append(action)
            log("verify_approved", {"tool": action.tool_name, "reason": "dry_run_passed"})
        else:
            fallback = Action(
                tool_name="self_reflect",
                parameters={"note": f"Blocked: {action.tool_name} — {dry.error}"},
                rationale="Downgraded after failed dry-run",
                estimated_risk=RiskLevel.LOW,
                parent_goal_id=action.parent_goal_id,
                utility_score=action.utility_score * 0.3,
            )
            approved.append(fallback)
            retried += 1
            log("verify_downgraded", {"original": action.tool_name, "error": dry.error})

return approved, blocked, retried
```

# ── Video OBSERVE hook ────────────────────────────────────────────────────────

# Keywords that indicate Janus should proactively watch a tutorial

_LEARNING_KEYWORDS = {
“code”, “coding”, “programming”, “tutorial”, “learn”, “hardware”,
“python”, “rust”, “component”, “cpu”, “gpu”, “memory”, “algorithm”,
“architecture”, “build”, “install”, “configure”, “debug”, “explain”,
}

def _goals_suggest_video_learning(goals: List[Goal]) -> bool:
“”“Return True if any active goal description contains learning keywords.”””
for goal in goals:
desc_lower = goal.description.lower()
if any(kw in desc_lower for kw in _LEARNING_KEYWORDS):
return True
return False

def _video_valence_nudge(summary: str) -> Dict[str, float]:
“””
Map video content type keywords to small valence adjustments.
Returns delta dict suitable for _nudge_valence().
“””
nudges: Dict[str, float] = {}
sl = summary.lower()
if “tutorial” in sl or “instructional” in sl or “learning value” in sl:
nudges[“curiosity”]  =  0.06
nudges[“competence”] =  0.04
nudges[“pleasure”]   =  0.02
if “paused” in sl or “static” in sl:
nudges[“arousal”]    = -0.03
if “fast” in sl or “demo” in sl:
nudges[“arousal”]    =  0.04
nudges[“curiosity”]  =  0.03
if “lecture” in sl or “slide” in sl:
nudges[“competence”] =  0.05
nudges[“arousal”]    = -0.02
return nudges

# ── Main cognitive loop ───────────────────────────────────────────────────────

class CognitiveLoop:
def **init**(
self,
core: AutonomousCore,
planner: Optional[GoalPlanner] = None,
hub: Optional[Any] = None,
observer: Optional[VideoObserver] = None,
max_retries: int = 2,
idle_interval: float = 1.0,
auto_video_learning: bool = True,
):
self.core      = core
self.planner   = planner or GoalPlanner()
self.observer  = observer or get_observer()
self.executor  = ToolExecutor(core, hub, self.observer)
self.max_retries   = max_retries
self.idle_interval = idle_interval
self.auto_video_learning = auto_video_learning
self._cycle_id     = 0
self._event_log: List[Dict[str, Any]] = []
self._last_video_start = 0.0
self._video_cooldown   = 120.0   # don’t auto-start video more than once per 2min

```
# ── Public API ────────────────────────────────────────────────────────────

def step(self, stimulus: Optional[str] = None) -> CycleResult:
    self._cycle_id += 1
    self._event_log = []
    video_summaries: List[str] = []
    start_valence = self._valence_snapshot()

    # ── OBSERVE ──────────────────────────────────────────────────
    self._log(Phase.OBSERVE, "phase_start", {})
    if stimulus:
        self.core.perceive(stimulus)
        self._log(Phase.OBSERVE, "stimulus_ingested", {"length": len(stimulus)})

    # Drain any accumulated video summaries
    new_summaries = self.observer.drain_summaries()
    for s in new_summaries:
        video_summaries.append(s)
        self.core.perceive(f"[VIDEO_LEARNING] {s}")
        self._log(Phase.OBSERVE, "video_summary_ingested", {"summary": s[:120]})

        # Apply valence nudges from video content
        nudges = _video_valence_nudge(s)
        if nudges:
            self._nudge_valence(**nudges)

    # Snapshot valence after all perception
    current_valence_tensor = self.core.current_valence.to_tensor()
    self._log(Phase.OBSERVE, "valence_sampled", self._valence_snapshot())

    # ── PLAN ─────────────────────────────────────────────────────
    self._log(Phase.PLAN, "phase_start", {})
    goals = self.planner.derive_goals(current_valence_tensor)
    self._log(Phase.PLAN, "goals_derived", {
        "count": len(goals),
        "goals": [{"id": g.id, "priority": round(g.priority, 3)} for g in goals]
    })

    # Auto-start video learning if curiosity is high and goals suggest it
    if (self.auto_video_learning
            and _goals_suggest_video_learning(goals)
            and not self.observer.is_playing()
            and time.time() - self._last_video_start > self._video_cooldown):
        self._auto_start_learning_video(goals)

    if not goals:
        self._log(Phase.IDLE, "no_goals", {"reason": "all dimensions near set-point"})
        return self._make_result(
            phase=Phase.IDLE, goals=goals,
            proposed=[], approved=[], applied=0, retried=0, blocked=0,
            start_valence=start_valence, video_summaries=video_summaries)

    # ── PROPOSE ──────────────────────────────────────────────────
    self._log(Phase.PROPOSE, "phase_start", {})
    proposed = self.planner.propose_actions(goals)
    self._log(Phase.PROPOSE, "actions_proposed", {
        "count": len(proposed),
        "actions": [{"tool": a.tool_name, "risk": a.estimated_risk} for a in proposed]
    })

    # ── VERIFY ───────────────────────────────────────────────────
    self._log(Phase.VERIFY, "phase_start", {})
    approved, blocked, retried = _verify(
        proposed, goals, self.executor,
        log=lambda e, d: self._log(Phase.VERIFY, e, d))
    self._log(Phase.VERIFY, "summary",
              {"approved": len(approved), "blocked": len(blocked)})

    # ── APPLY ────────────────────────────────────────────────────
    self._log(Phase.APPLY, "phase_start", {})
    applied = 0
    retry_count = 0
    for action in approved:
        ok = self._apply_with_retry(action)
        if ok:
            applied += 1
        else:
            retry_count += 1

    end_valence = self._valence_snapshot()
    self._log(Phase.APPLY, "phase_complete", {"applied": applied})

    return self._make_result(
        phase=Phase.APPLY, goals=goals,
        proposed=proposed, approved=approved,
        applied=applied, retried=retry_count, blocked=len(blocked),
        start_valence=start_valence, end_valence=end_valence,
        video_summaries=video_summaries)

def run(self, stimuli: Optional[List[str]] = None, cycles: int = 1) -> List[CycleResult]:
    results = []
    for i in range(cycles):
        stim = stimuli[i] if stimuli and i < len(stimuli) else None
        results.append(self.step(stim))
    return results

# ── Auto video learning ───────────────────────────────────────────────────

def _auto_start_learning_video(self, goals: List[Goal]):
    """
    Pick a learning topic from active goals and open a YouTube search.
    Uses the most urgent goal's target_dimension as the search seed.
    """
    topic_map = {
        "competence": "python programming tutorial beginner",
        "curiosity":  "how computers work hardware explained",
        "autonomy":   "software architecture patterns tutorial",
        "pleasure":   "beginner coding projects fun",
    }
    for goal in goals:
        query = topic_map.get(goal.target_dimension)
        if query:
            import urllib.parse
            url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
            started = self.observer.watch_video(url)
            if started:
                self._last_video_start = time.time()
                self._log(Phase.OBSERVE, "auto_video_learning_started",
                          {"goal": goal.id, "query": query, "url": url})
            break

# ── Internal helpers ──────────────────────────────────────────────────────

def _apply_with_retry(self, action: Action) -> bool:
    attempts = 0
    current  = action
    while attempts <= self.max_retries:
        try:
            result = self.executor.execute(current)
            self._log(Phase.APPLY, "action_success", {
                "tool": current.tool_name, "attempt": attempts + 1,
                "result": str(result)[:100]})
            self._nudge_valence(competence=+0.05, pleasure=+0.02)
            return True
        except Exception as exc:
            attempts += 1
            self._log(Phase.APPLY, "action_failed", {
                "tool": current.tool_name, "attempt": attempts, "error": str(exc)})
            self._nudge_valence(competence=-0.08, pleasure=-0.03)
            if attempts <= self.max_retries:
                current = Action(
                    tool_name="self_reflect",
                    parameters={"note": f"Retry after {action.tool_name} failed: {exc}"},
                    rationale="Auto retry fallback",
                    estimated_risk=RiskLevel.LOW,
                    parent_goal_id=action.parent_goal_id,
                    utility_score=action.utility_score * 0.5)
    return False

def _nudge_valence(self, **kwargs: float):
    v = self.core.current_valence
    fields = {
        "pleasure": v.pleasure, "arousal": v.arousal,
        "curiosity": v.curiosity, "autonomy": v.autonomy,
        "connection": v.connection, "competence": v.competence,
    }
    updates = {}
    for dim, delta in kwargs.items():
        if dim in fields:
            updates[dim] = torch.clamp(fields[dim] + delta, -1.0, 1.0)
    if updates:
        self.core.current_valence = type(v)(
            pleasure  = updates.get("pleasure",   v.pleasure),
            arousal   = updates.get("arousal",    v.arousal),
            curiosity = updates.get("curiosity",  v.curiosity),
            autonomy  = updates.get("autonomy",   v.autonomy),
            connection= updates.get("connection", v.connection),
            competence= updates.get("competence", v.competence),
        )

def _valence_snapshot(self) -> Dict[str, float]:
    v = self.core.current_valence
    return {k: round(getattr(v, k).item(), 4)
            for k in ("pleasure", "arousal", "curiosity", "autonomy", "connection", "competence")}

def _log(self, phase: Phase, event: str, data: Dict[str, Any]):
    self._event_log.append({
        "cycle": self._cycle_id, "phase": phase.value,
        "event": event, "timestamp": round(time.time(), 4), **data})

def _make_result(self, phase, goals, proposed, approved,
                 applied, retried, blocked, start_valence,
                 end_valence=None, video_summaries=None) -> CycleResult:
    return CycleResult(
        cycle_id=self._cycle_id, timestamp=time.time(),
        phase_reached=phase,
        goals_generated=len(goals),
        actions_proposed=len(proposed), actions_approved=len(approved),
        actions_applied=applied, actions_retried=retried, actions_blocked=blocked,
        valence_before=start_valence, valence_after=end_valence or start_valence,
        events=list(self._event_log),
        video_summaries=video_summaries or [])
```