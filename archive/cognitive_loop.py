“””
cognitive_loop.py — Janus OBSERVE → PLAN → PROPOSE → VERIFY → APPLY loop.

This is the missing orchestration layer that glues janus-brain together.

Key design decisions (addressing the bottlenecks from the architecture review):

1. PROPOSE uses GoalPlanner (valence-deficit scoring) instead of if/elif.
   Every tool competes on a utility score; no goal is hardcoded to a tool.
1. VERIFY implements a tiered gate rather than a binary low/high block:
- LOW risk  → auto-approved
- MEDIUM risk → approved if urgency above threshold
- HIGH risk  → sandboxed dry-run first; approved only on simulated success
1. ValenceState is injected into every phase as context so the loop is
   “aware of its own feelings” throughout planning, not just after acting.
1. APPLY has retry-with-modification logic: if a tool raises an exception,
   the loop downgrades the action, logs the failure, and updates valence
   negatively (competence drop) before re-entering the cycle.
1. The bridge (bridge.py) can drive this loop by adding a “loop_step” command,
   keeping the Rust ↔ Python protocol unchanged.
   “””

from **future** import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

from .core import AutonomousCore
from .homeostasis import ValenceVector
from .goal_planner import Action, Goal, GoalPlanner, RiskLevel

# —————————————————————————

# Loop phases as an enum for logging clarity

# —————————————————————————

class Phase(str, Enum):
OBSERVE  = “OBSERVE”
PLAN     = “PLAN”
PROPOSE  = “PROPOSE”
VERIFY   = “VERIFY”
APPLY    = “APPLY”
IDLE     = “IDLE”

# —————————————————————————

# Cycle result — returned after each full loop iteration

# —————————————————————————

@dataclass
class CycleResult:
cycle_id: int
timestamp: float
phase_reached: Phase
goals_generated: int
actions_proposed: int
actions_approved: int
actions_applied: int
actions_retried: int
actions_blocked: int
valence_before: Dict[str, float]
valence_after: Dict[str, float]
events: List[Dict[str, Any]]

```
def to_json(self) -> str:
    return json.dumps(asdict(self), indent=2)
```

# —————————————————————————

# Sandbox / dry-run stub

# —————————————————————————

class SandboxResult:
“”“Result of a dry-run execution used by the Verify phase.”””
def **init**(self, success: bool, output: Any, error: Optional[str] = None):
self.success = success
self.output  = output
self.error   = error

def _dry_run(action: Action, executor: “ToolExecutor”) -> SandboxResult:
“””
Lightweight simulation of an action without real side effects.

```
In production this would spin up a WASM subprocess.  Here we do a
best-effort Python-level simulation:
  - For HIGH risk filesystem actions, we check preconditions.
  - For everything else, we treat the dry-run as always-safe.
"""
try:
    if action.tool_name == "write_state_snapshot":
        # Simulate: can we serialise the state?
        state_str = json.dumps({"test": True})
        return SandboxResult(success=True, output=f"Simulated write: {len(state_str)} bytes")

    if action.tool_name == "run_analysis_pipeline":
        video_path = action.parameters.get("video_path", "")
        import os
        if video_path and not os.path.exists(video_path):
            return SandboxResult(
                success=False, output=None,
                error=f"Precondition failed: video_path '{video_path}' not found"
            )
        return SandboxResult(success=True, output="Precondition check passed")

    # Default: assume dry-run passes for unrecognised tools
    return SandboxResult(success=True, output="No dry-run logic; assumed safe")

except Exception as exc:
    return SandboxResult(success=False, output=None, error=str(exc))
```

# —————————————————————————

# Tool executor — maps action.tool_name → AutonomousCore method calls

# —————————————————————————

class ToolExecutor:
“””
Calls the appropriate AutonomousCore (or FinalJanusSystem) method
for a given Action and returns the raw result.

```
Add new tool handlers by extending _DISPATCH or calling register().
"""

def __init__(self, core: AutonomousCore, janus_system: Optional[Any] = None):
    self.core   = core
    self.janus  = janus_system   # FinalJanusSystem, optional

    self._dispatch: Dict[str, Callable[[Action], Any]] = {
        "self_reflect":          self._exec_self_reflect,
        "explore_memory":        self._exec_explore_memory,
        "generate_response":     self._exec_generate_response,
        "perceive_input":        self._exec_perceive_input,
        "consolidate_sleep":     self._exec_consolidate_sleep,
        "write_state_snapshot":  self._exec_write_state_snapshot,
        "run_analysis_pipeline": self._exec_run_analysis_pipeline,
    }

def register(self, name: str, fn: Callable[[Action], Any]) -> None:
    self._dispatch[name] = fn

def execute(self, action: Action) -> Any:
    handler = self._dispatch.get(action.tool_name)
    if handler is None:
        raise NotImplementedError(f"No executor for tool '{action.tool_name}'")
    return handler(action)

# --- individual handlers -------------------------------------------

def _exec_self_reflect(self, action: Action) -> str:
    return self.core.reflect()

def _exec_explore_memory(self, action: Action) -> List[str]:
    window = action.parameters.get("window", 50)
    return self.core.memory.mine_themes()

def _exec_generate_response(self, action: Action) -> str:
    prompt = action.parameters.get("prompt", "Continue coherently.")
    max_tokens = action.parameters.get("max_tokens", 150)
    return self.core.generate_response(prompt, max_tokens)

def _exec_perceive_input(self, action: Action) -> None:
    text = action.parameters.get("text", "")
    if text:
        self.core.perceive(text)

def _exec_consolidate_sleep(self, action: Action) -> str:
    from .core import SleepEngine
    engine = SleepEngine(self.core)
    engine.consolidate()
    return "Consolidation complete"

def _exec_write_state_snapshot(self, action: Action) -> str:
    import json, os
    path = action.parameters.get("path", "persistent_state.json")
    v = self.core.current_valence
    snapshot = {
        "timestamp": time.time(),
        "valence": {
            "pleasure":   v.pleasure.item(),
            "arousal":    v.arousal.item(),
            "curiosity":  v.curiosity.item(),
            "autonomy":   v.autonomy.item(),
            "connection": v.connection.item(),
            "competence": v.competence.item(),
        }
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return f"State written to {path}"

def _exec_run_analysis_pipeline(self, action: Action) -> str:
    if self.janus is None:
        raise RuntimeError("FinalJanusSystem not attached to ToolExecutor")
    video_path  = action.parameters.get("video_path", "")
    output_file = action.parameters.get("output_file", "janus_report.txt")
    self.janus.analyze_video(video_path, output_file)
    return f"Analysis written to {output_file}"
```

# —————————————————————————

# Verify phase

# —————————————————————————

_MEDIUM_URGENCY_THRESHOLD = 0.4   # minimum goal priority to approve MEDIUM risk

def _verify(
actions: List[Action],
goals: List[Goal],
executor: ToolExecutor,
log: Callable[[str, Dict], None],
) -> tuple[List[Action], List[Action], int]:
“””
Returns (approved_actions, blocked_actions, retried_count).

```
Tiered logic:
  LOW    → always approved
  MEDIUM → approved when the parent goal has priority ≥ threshold
  HIGH   → approved only if dry-run succeeds
"""
goal_map = {g.id: g for g in goals}
approved: List[Action] = []
blocked:  List[Action] = []
retried = 0

for action in actions:
    parent_goal = goal_map.get(action.parent_goal_id)
    parent_priority = parent_goal.priority if parent_goal else 0.0

    if action.estimated_risk == RiskLevel.LOW:
        approved.append(action)
        log("verify_approved", {"tool": action.tool_name, "risk": action.estimated_risk, "reason": "low_risk"})

    elif action.estimated_risk == RiskLevel.MEDIUM:
        if parent_priority >= _MEDIUM_URGENCY_THRESHOLD:
            approved.append(action)
            log("verify_approved", {"tool": action.tool_name, "risk": action.estimated_risk, "reason": "sufficient_urgency"})
        else:
            blocked.append(action)
            log("verify_blocked", {"tool": action.tool_name, "risk": action.estimated_risk, "reason": "insufficient_urgency"})

    elif action.estimated_risk == RiskLevel.HIGH:
        dry = _dry_run(action, executor)
        if dry.success:
            approved.append(action)
            log("verify_approved", {
                "tool": action.tool_name,
                "risk": action.estimated_risk,
                "reason": "dry_run_passed",
                "dry_run_output": str(dry.output)
            })
        else:
            # Attempt one downgrade before blocking
            downgraded = Action(
                tool_name="self_reflect",
                parameters={"note": f"Blocked high-risk action '{action.tool_name}': {dry.error}"},
                rationale=f"Downgraded from '{action.tool_name}' after failed dry-run",
                estimated_risk=RiskLevel.LOW,
                parent_goal_id=action.parent_goal_id,
                utility_score=action.utility_score * 0.3,
            )
            approved.append(downgraded)
            retried += 1
            log("verify_downgraded", {
                "original_tool": action.tool_name,
                "new_tool": "self_reflect",
                "dry_run_error": dry.error
            })

return approved, blocked, retried
```

# —————————————————————————

# Main cognitive loop

# —————————————————————————

class CognitiveLoop:
“””
Runs the OBSERVE → PLAN → PROPOSE → VERIFY → APPLY cycle.

```
Parameters
----------
core : AutonomousCore
planner : GoalPlanner
janus_system : optional FinalJanusSystem (for vision/audio pipeline)
max_retries : int
    How many times to retry a failed APPLY before giving up.
idle_interval : float
    Seconds to sleep when no goals are active.
"""

def __init__(
    self,
    core: AutonomousCore,
    planner: Optional[GoalPlanner] = None,
    janus_system: Optional[Any] = None,
    max_retries: int = 2,
    idle_interval: float = 1.0,
):
    self.core          = core
    self.planner       = planner or GoalPlanner()
    self.executor      = ToolExecutor(core, janus_system)
    self.max_retries   = max_retries
    self.idle_interval = idle_interval
    self._cycle_id     = 0
    self._event_log: List[Dict[str, Any]] = []

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def step(self, stimulus: Optional[str] = None) -> CycleResult:
    """Execute one full OBSERVE → APPLY cycle and return a CycleResult."""
    self._cycle_id += 1
    self._event_log = []
    start_valence = self._valence_snapshot()

    # ── OBSERVE ──────────────────────────────────────────────────
    self._log(Phase.OBSERVE, "phase_start", {})
    if stimulus:
        self.core.perceive(stimulus)
        self._log(Phase.OBSERVE, "stimulus_ingested", {"length": len(stimulus)})

    current_valence_tensor = self.core.current_valence.to_tensor()
    self._log(Phase.OBSERVE, "valence_sampled", self._valence_snapshot())

    # ── PLAN ─────────────────────────────────────────────────────
    self._log(Phase.PLAN, "phase_start", {})
    goals = self.planner.derive_goals(current_valence_tensor)
    self._log(Phase.PLAN, "goals_derived", {
        "count": len(goals),
        "goals": [{"id": g.id, "priority": round(g.priority, 3)} for g in goals]
    })

    if not goals:
        self._log(Phase.IDLE, "no_goals", {"reason": "all dimensions near set-point"})
        return self._make_result(
            phase=Phase.IDLE,
            goals=goals,
            proposed=[], approved=[], applied=0, retried=0, blocked=0,
            start_valence=start_valence,
        )

    # ── PROPOSE ──────────────────────────────────────────────────
    self._log(Phase.PROPOSE, "phase_start", {})
    proposed_actions = self.planner.propose_actions(goals)
    self._log(Phase.PROPOSE, "actions_proposed", {
        "count": len(proposed_actions),
        "actions": [
            {"tool": a.tool_name, "risk": a.estimated_risk, "utility": round(a.utility_score, 3)}
            for a in proposed_actions
        ]
    })

    # ── VERIFY ───────────────────────────────────────────────────
    self._log(Phase.VERIFY, "phase_start", {})
    approved, blocked, retried = _verify(
        proposed_actions, goals, self.executor,
        log=lambda event, data: self._log(Phase.VERIFY, event, data)
    )
    self._log(Phase.VERIFY, "verify_summary", {
        "approved": len(approved), "blocked": len(blocked), "retried": retried
    })

    # ── APPLY ────────────────────────────────────────────────────
    self._log(Phase.APPLY, "phase_start", {})
    applied_count = 0
    retry_count = 0

    for action in approved:
        success = self._apply_with_retry(action)
        if success:
            applied_count += 1
        else:
            retry_count += 1

    end_valence = self._valence_snapshot()
    self._log(Phase.APPLY, "phase_complete", {
        "applied": applied_count,
        "failed": len(approved) - applied_count,
    })

    return self._make_result(
        phase=Phase.APPLY,
        goals=goals,
        proposed=proposed_actions,
        approved=approved,
        applied=applied_count,
        retried=retry_count,
        blocked=len(blocked),
        start_valence=start_valence,
        end_valence=end_valence,
    )

def run(self, stimuli: Optional[List[str]] = None, cycles: int = 1) -> List[CycleResult]:
    """Run N cycles, optionally feeding a list of stimuli (one per cycle)."""
    results = []
    for i in range(cycles):
        stim = stimuli[i] if stimuli and i < len(stimuli) else None
        results.append(self.step(stim))
    return results

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _apply_with_retry(self, action: Action) -> bool:
    """
    Try to execute action.  On failure, apply a negative valence nudge
    (competence drop) and retry up to max_retries times with a
    fallback to self_reflect.
    """
    attempts = 0
    current_action = action

    while attempts <= self.max_retries:
        try:
            result = self.executor.execute(current_action)
            self._log(Phase.APPLY, "action_success", {
                "tool": current_action.tool_name,
                "attempt": attempts + 1,
                "result_preview": str(result)[:120],
            })
            # Positive competence nudge on success
            self._nudge_valence(competence=+0.05, pleasure=+0.02)
            return True

        except Exception as exc:
            attempts += 1
            self._log(Phase.APPLY, "action_failed", {
                "tool": current_action.tool_name,
                "attempt": attempts,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            })
            # Competence drop on failure
            self._nudge_valence(competence=-0.08, pleasure=-0.03)

            if attempts <= self.max_retries:
                # Downgrade to a safe fallback
                current_action = Action(
                    tool_name="self_reflect",
                    parameters={"note": f"Retry fallback after '{action.tool_name}' failed: {exc}"},
                    rationale="Auto-retry with low-risk fallback",
                    estimated_risk=RiskLevel.LOW,
                    parent_goal_id=action.parent_goal_id,
                    utility_score=action.utility_score * 0.5,
                )
                self._log(Phase.APPLY, "retry_with_fallback", {
                    "original": action.tool_name,
                    "attempt": attempts + 1,
                })

    return False

def _nudge_valence(self, **kwargs: float) -> None:
    """Apply a small direct delta to specific valence dimensions."""
    v = self.core.current_valence
    mapping = {
        "pleasure":   "pleasure",
        "arousal":    "arousal",
        "curiosity":  "curiosity",
        "autonomy":   "autonomy",
        "connection": "connection",
        "competence": "competence",
    }
    updates = {}
    for dim, delta in kwargs.items():
        if dim in mapping:
            current = getattr(v, mapping[dim])
            updates[mapping[dim]] = torch.clamp(current + delta, -1.0, 1.0)
    if updates:
        self.core.current_valence = type(v)(
            pleasure   = updates.get("pleasure",   v.pleasure),
            arousal    = updates.get("arousal",    v.arousal),
            curiosity  = updates.get("curiosity",  v.curiosity),
            autonomy   = updates.get("autonomy",   v.autonomy),
            connection = updates.get("connection", v.connection),
            competence = updates.get("competence", v.competence),
        )

def _valence_snapshot(self) -> Dict[str, float]:
    v = self.core.current_valence
    return {
        "pleasure":   round(v.pleasure.item(), 4),
        "arousal":    round(v.arousal.item(), 4),
        "curiosity":  round(v.curiosity.item(), 4),
        "autonomy":   round(v.autonomy.item(), 4),
        "connection": round(v.connection.item(), 4),
        "competence": round(v.competence.item(), 4),
    }

def _log(self, phase: Phase, event: str, data: Dict[str, Any]) -> None:
    entry = {
        "cycle":     self._cycle_id,
        "phase":     phase.value,
        "event":     event,
        "timestamp": round(time.time(), 4),
        **data,
    }
    self._event_log.append(entry)

def _make_result(
    self,
    phase: Phase,
    goals: List[Goal],
    proposed: List[Action],
    approved: List[Action],
    applied: int,
    retried: int,
    blocked: int,
    start_valence: Dict[str, float],
    end_valence: Optional[Dict[str, float]] = None,
) -> CycleResult:
    return CycleResult(
        cycle_id         = self._cycle_id,
        timestamp        = time.time(),
        phase_reached    = phase,
        goals_generated  = len(goals),
        actions_proposed = len(proposed),
        actions_approved = len(approved),
        actions_applied  = applied,
        actions_retried  = retried,
        actions_blocked  = blocked,
        valence_before   = start_valence,
        valence_after    = end_valence or start_valence,
        events           = list(self._event_log),
    )
```