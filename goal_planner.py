“””
goal_planner.py — Janus Goal & Plan representation layer.

Translates the current ValenceVector into a prioritized set of Goals,
then decomposes each Goal into an ordered list of Actions that the
cognitive loop can execute.

Design notes:

- Goals are NOT hardcoded to specific tool calls.  Instead, the planner
  scores every registered tool against the current valence deficit and
  lets the highest-scoring tools win.  This replaces the if/elif
  pattern that was discussed as the “Propose bottleneck.”
- Actions carry an estimated_risk level so the Verify phase can gate
  them without having to understand the action semantics itself.
- The planner is stateless between calls; all state lives in the core.
  “””

from **future** import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

# —————————————————————————

# Risk taxonomy

# —————————————————————————

class RiskLevel(str, Enum):
LOW    = “low”      # read-only, reversible
MEDIUM = “medium”   # writes to in-memory state only
HIGH   = “high”     # file-system / network / subprocess side effects

# —————————————————————————

# Tool registry

# —————————————————————————

@dataclass
class ToolSpec:
“”“Describes a capability available to the cognitive loop.”””
name: str
description: str
risk: RiskLevel
# Valence dimensions this tool addresses (dimension -> improvement direction)
# Positive means the tool raises that dimension, negative means it lowers it.
valence_affinity: Dict[str, float] = field(default_factory=dict)

# Default tool catalogue — extend at runtime via PlannerConfig.register_tool()

DEFAULT_TOOLS: List[ToolSpec] = [
ToolSpec(
name=“self_reflect”,
description=“Synthesise recent episodic memory into a self-narrative.”,
risk=RiskLevel.LOW,
valence_affinity={“pleasure”: 0.3, “curiosity”: 0.2, “competence”: 0.1},
),
ToolSpec(
name=“explore_memory”,
description=“Mine the episodic buffer for thematic patterns.”,
risk=RiskLevel.LOW,
valence_affinity={“curiosity”: 0.4, “autonomy”: 0.2},
),
ToolSpec(
name=“generate_response”,
description=“Produce a language response conditioned on current mood.”,
risk=RiskLevel.LOW,
valence_affinity={“connection”: 0.4, “competence”: 0.2},
),
ToolSpec(
name=“perceive_input”,
description=“Ingest new multimodal stimulus and update valence.”,
risk=RiskLevel.LOW,
valence_affinity={“arousal”: 0.3, “curiosity”: 0.3},
),
ToolSpec(
name=“consolidate_sleep”,
description=“Run offline memory consolidation (SleepEngine).”,
risk=RiskLevel.MEDIUM,
valence_affinity={“pleasure”: 0.2, “competence”: 0.3, “autonomy”: 0.1},
),
ToolSpec(
name=“write_state_snapshot”,
description=“Persist current cognitive state to disk.”,
risk=RiskLevel.HIGH,
valence_affinity={“autonomy”: 0.2, “competence”: 0.1},
),
ToolSpec(
name=“run_analysis_pipeline”,
description=“Execute the full multimodal video analysis pipeline.”,
risk=RiskLevel.HIGH,
valence_affinity={“curiosity”: 0.5, “competence”: 0.4},
),
]

# —————————————————————————

# Goal and Action dataclasses

# —————————————————————————

@dataclass
class Goal:
“”“A high-level homeostatic objective derived from valence deficits.”””
id: str
description: str
priority: float          # 0–1, higher = more urgent
target_dimension: str    # which valence dimension this goal primarily serves
deficit: float           # magnitude of the gap from set-point

@dataclass
class Action:
“”“A concrete step the loop should execute.”””
tool_name: str
parameters: Dict[str, Any]
rationale: str
estimated_risk: RiskLevel
parent_goal_id: str
utility_score: float     # expected valence improvement

# —————————————————————————

# Planner

# —————————————————————————

_SET_POINTS: Dict[str, float] = {
“pleasure”:   0.6,
“arousal”:    0.5,
“curiosity”:  0.4,
“autonomy”:   0.6,
“connection”: 0.5,
“competence”: 0.5,
}

_DIMENSION_NAMES = list(_SET_POINTS.keys())

class GoalPlanner:
“””
Converts the current ValenceVector into a ranked list of Goals,
then maps each Goal to the best-matching Action(s).

```
Parameters
----------
tools : list of ToolSpec
    The available capability catalogue.
goal_threshold : float
    Minimum deficit magnitude to generate a Goal (filters noise).
max_goals : int
    Cap on concurrent goals to prevent plan explosion.
max_actions_per_goal : int
    How many actions to propose per goal.
"""

def __init__(
    self,
    tools: Optional[List[ToolSpec]] = None,
    goal_threshold: float = 0.12,
    max_goals: int = 3,
    max_actions_per_goal: int = 2,
):
    self.tools = {t.name: t for t in (tools or DEFAULT_TOOLS)}
    self.goal_threshold = goal_threshold
    self.max_goals = max_goals
    self.max_actions_per_goal = max_actions_per_goal

def register_tool(self, spec: ToolSpec) -> None:
    """Add a new tool to the catalogue at runtime."""
    self.tools[spec.name] = spec

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def derive_goals(self, valence_tensor: torch.Tensor) -> List[Goal]:
    """
    Compare each valence dimension against its set-point and return
    the top-N goals ordered by urgency (largest deficit first).
    """
    values = valence_tensor.detach().tolist()
    goals: List[Goal] = []

    for dim, val in zip(_DIMENSION_NAMES, values):
        set_point = _SET_POINTS[dim]
        deficit = set_point - val          # positive means below set-point
        if abs(deficit) < self.goal_threshold:
            continue

        direction = "raise" if deficit > 0 else "lower"
        goals.append(Goal(
            id=f"goal_{dim}",
            description=f"{direction.capitalize()} {dim} (current={val:.2f}, target={set_point:.2f})",
            priority=self._urgency(abs(deficit)),
            target_dimension=dim,
            deficit=deficit,
        ))

    goals.sort(key=lambda g: g.priority, reverse=True)
    return goals[: self.max_goals]

def propose_actions(self, goals: List[Goal]) -> List[Action]:
    """
    For each goal, score every tool by how well its valence_affinity
    addresses the goal's deficit, then emit the top-K actions.
    """
    actions: List[Action] = []

    for goal in goals:
        scored = self._score_tools_for_goal(goal)
        for tool_name, score in scored[: self.max_actions_per_goal]:
            tool = self.tools[tool_name]
            actions.append(Action(
                tool_name=tool_name,
                parameters=self._default_params(tool_name, goal),
                rationale=(
                    f"Selected to address {goal.id} "
                    f"(deficit={goal.deficit:+.2f}); "
                    f"utility={score:.3f}"
                ),
                estimated_risk=tool.risk,
                parent_goal_id=goal.id,
                utility_score=score,
            ))

    return actions

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

@staticmethod
def _urgency(deficit_magnitude: float) -> float:
    """Non-linear urgency: small deficits feel negligible, large ones spike."""
    return 1.0 - math.exp(-3.0 * deficit_magnitude)

def _score_tools_for_goal(self, goal: Goal) -> List[tuple[str, float]]:
    """
    Score = affinity_for_target_dim * sign_match * deficit_weight.
    """
    results = []
    for name, tool in self.tools.items():
        affinity = tool.valence_affinity.get(goal.target_dimension, 0.0)
        # Sign: if we need to raise the dimension, positive affinity is good.
        sign_ok = (goal.deficit > 0 and affinity > 0) or (goal.deficit < 0 and affinity < 0)
        if not sign_ok:
            affinity = max(affinity * -0.5, 0.0)   # slight penalty for wrong direction
        score = abs(affinity) * abs(goal.deficit)
        results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

@staticmethod
def _default_params(tool_name: str, goal: Goal) -> Dict[str, Any]:
    """
    Minimal default parameters per tool type.
    The cognitive loop or executor can enrich these before calling the tool.
    """
    base: Dict[str, Any] = {"goal_id": goal.id}
    if tool_name == "self_reflect":
        base["depth"] = "standard"
    elif tool_name == "generate_response":
        base["max_tokens"] = 150
        base["mood_conditioned"] = True
    elif tool_name == "explore_memory":
        base["window"] = 50
    elif tool_name == "consolidate_sleep":
        base["replay_limit"] = 100
    return base
```