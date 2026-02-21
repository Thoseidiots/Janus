# tree_planner.py

“””
Tree-of-Thought planner integrated with janus-brain components:

- LLM adapter: wraps ByteLLM.generate() or any callable(prompt)->str
- Memory: uses ReflectionMemory.add(valence, context) API
- Homeostasis: curiosity nudge when a high-scoring branch is found
- CognitiveLoop: exposes plan_toward_goal() as a registerable hub tool

BFS with correct per-node depth termination, regex score extraction,
beam-search path selection, and graceful degradation if LLM is untrained.
“””

from **future** import annotations

import re
import uuid
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

# ── Tree node ─────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
thought: str
parent: Optional[“TreeNode”] = field(default=None, repr=False)
id: str = field(default_factory=lambda: str(uuid.uuid4()))
children: List[“TreeNode”] = field(default_factory=list)
score: float = 0.0
depth: int = 0

```
def __post_init__(self):
    if self.parent is not None:
        self.depth = self.parent.depth + 1
```

# ── LLM adapter ───────────────────────────────────────────────────────────────

class LLMAdapter:
“””
Wraps whatever LLM backend Janus has.
Priority: janus-brain ByteLLM → plain callable → stub.
“””

```
def __init__(self, llm=None):
    self._llm = llm

def __call__(self, prompt: str, max_tokens: int = 200) -> str:
    if self._llm is None:
        return self._stub(prompt)

    # ByteLLM / MoodConditionedByteLLM (has .generate())
    if hasattr(self._llm, "generate"):
        try:
            return self._llm.generate(prompt, max_new=max_tokens)
        except Exception:
            pass

    # AutonomousCore (has .generate_response())
    if hasattr(self._llm, "generate_response"):
        try:
            return self._llm.generate_response(prompt, max_tokens)
        except Exception:
            pass

    # Plain callable
    if callable(self._llm):
        try:
            return str(self._llm(prompt))
        except Exception:
            pass

    return self._stub(prompt)

@staticmethod
def _stub(prompt: str) -> str:
    """Fallback when no real LLM is available — returns template response."""
    if "rate feasibility" in prompt.lower():
        return "0.5 — unable to score without active LLM."
    return "1. Research the requirement\n2. Take the first low-cost action\n3. Validate the result"
```

# ── Score extractor ───────────────────────────────────────────────────────────

_SCORE_PATTERN = re.compile(r”\b(0.\d+|1.0|1)\b”)

def _extract_score(text: str) -> float:
“””
Scan entire LLM response for the first float in [0, 1].
Much more robust than splitting on whitespace.
“””
matches = _SCORE_PATTERN.findall(text)
if matches:
try:
return max(0.0, min(1.0, float(matches[0])))
except ValueError:
pass
return 0.5

# ── Neutral valence for memory storage ───────────────────────────────────────

def _make_planning_valence(curiosity_boost: float = 0.0):
“”“Create a ValenceVector for storing planning entries in memory.”””
try:
import torch
from janus_brain.homeostasis import ValenceVector
return ValenceVector(
pleasure   = torch.tensor(0.5),
arousal    = torch.tensor(0.4),
curiosity  = torch.tensor(min(1.0, 0.6 + curiosity_boost)),
autonomy   = torch.tensor(0.7),
connection = torch.tensor(0.4),
competence = torch.tensor(0.5),
)
except ImportError:
return None

# ── TreePlanner ───────────────────────────────────────────────────────────────

class TreePlanner:
“””
Tree-of-Thought planner using BFS with correct per-node depth gating,
LLM-scored branch evaluation, and beam-search path selection.

```
Parameters
----------
llm : ByteLLM | AutonomousCore | Callable | None
    Any object the LLMAdapter can wrap.
memory : ReflectionMemory | None
    janus-brain episodic buffer.
core : AutonomousCore | None
    If provided, valence is nudged when high-scoring branches are found.
branching_factor : int
    Max children per node.
max_depth : int
    Max depth of the tree (root = 0).
beam_width : int
    Number of paths to track in beam-search selection.
score_threshold : float
    Prune branches below this score to avoid exploring dead ends.
"""

def __init__(
    self,
    llm=None,
    memory=None,
    core=None,
    branching_factor: int = 3,
    max_depth: int = 5,
    beam_width: int = 2,
    score_threshold: float = 0.25,
):
    self.llm              = LLMAdapter(llm)
    self.memory           = memory
    self.core             = core
    self.branching_factor = branching_factor
    self.max_depth        = max_depth
    self.beam_width       = beam_width
    self.score_threshold  = score_threshold
    self._lock            = threading.Lock()

# ── Branch generation ─────────────────────────────────────────────────────

def generate_branches(self, current_thought: str, goal: str) -> List[str]:
    prompt = (
        f"Goal: {goal}\n"
        f"Current partial plan: {current_thought}\n\n"
        f"Generate {self.branching_factor} distinct, creative next steps or sub-plans.\n"
        f"Each should be concise, actionable, and realistic for a zero-income start.\n"
        f"Output as a numbered list, one step per line."
    )
    response = self.llm(prompt, max_tokens=300)
    branches = self._parse_numbered_list(response)

    # Fallback: if parsing yielded nothing, split on newlines
    if not branches:
        branches = [l.strip() for l in response.split("\n") if len(l.strip()) > 10]

    return branches[: self.branching_factor]

# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(self, thought: str, goal: str) -> float:
    prompt = (
        f"Rate the feasibility of the following step on a scale from 0.0 to 1.0 "
        f"for someone with zero capital starting in Hampton, VA.\n"
        f"Step: {thought}\n"
        f"Goal: {goal}\n"
        f"Begin your response with the numeric score (e.g. '0.7'), then explain briefly."
    )
    response = self.llm(prompt, max_tokens=120)
    return _extract_score(response)

# ── Tree construction (BFS, correct depth gating) ─────────────────────────

def build_tree(self, root_thought: str, goal: str) -> TreeNode:
    """
    BFS expansion.  Depth is checked on the *current node* before
    expanding it, not on the root.
    """
    root  = TreeNode(thought=root_thought)
    queue = [root]

    while queue:
        node = queue.pop(0)

        # Correct depth check: stop expanding if this node is already at max depth
        if node.depth >= self.max_depth:
            continue

        branches = self.generate_branches(node.thought, goal)
        for branch_text in branches:
            score = self.evaluate(branch_text, goal)

            # Prune low-score branches early
            if score < self.score_threshold:
                continue

            child = TreeNode(thought=branch_text, parent=node, score=score)
            node.children.append(child)
            queue.append(child)

            # Persist to janus-brain episodic memory
            self._store_in_memory(child, goal)

            # Nudge curiosity when a high-value branch is found
            if score > 0.75:
                self._nudge_curiosity(score)

    return root

# ── Path selection ────────────────────────────────────────────────────────

def select_best_path(self, root: TreeNode) -> List[str]:
    """
    Beam search: maintain beam_width frontier, always expand the
    highest-scoring node, return the best full path.
    """
    # Collect all leaf nodes
    leaves = self._collect_leaves(root)
    if not leaves:
        return [root.thought]

    # Score each leaf by the mean score along its path from root
    best_leaf = max(leaves, key=lambda n: self._path_mean_score(n))
    return self._path_to_root(best_leaf)

def select_beam_paths(self, root: TreeNode) -> List[List[str]]:
    """Return beam_width best paths (for presenting alternatives)."""
    leaves = self._collect_leaves(root)
    if not leaves:
        return [[root.thought]]
    sorted_leaves = sorted(leaves, key=lambda n: self._path_mean_score(n), reverse=True)
    return [self._path_to_root(l) for l in sorted_leaves[: self.beam_width]]

# ── High-level planning API ───────────────────────────────────────────────

def plan_toward_goal(self, goal: str, starting_context: str = "") -> Dict[str, Any]:
    """
    Full planning pipeline.  Returns a dict suitable for the capability hub.

    result = planner.plan_toward_goal(
        goal="Earn first income with zero capital",
        starting_context="Living in Hampton VA, have a car and smartphone"
    )
    """
    root_thought = starting_context or f"Start planning: {goal}"
    root = self.build_tree(root_thought, goal)
    best_path    = self.select_best_path(root)
    alternatives = self.select_beam_paths(root)
    total_nodes  = self._count_nodes(root)

    # Store the final plan in memory
    summary = f"Plan for '{goal}': " + " → ".join(best_path)
    self._store_plan_summary(summary)

    return {
        "status":       "ok",
        "goal":         goal,
        "best_path":    best_path,
        "alternatives": alternatives,
        "total_nodes_explored": total_nodes,
        "max_depth_reached": self._max_depth_reached(root),
    }

# ── Internal helpers ──────────────────────────────────────────────────────

@staticmethod
def _parse_numbered_list(text: str) -> List[str]:
    """Extract lines that start with a number + punctuation."""
    pattern = re.compile(r"^\s*\d+[.):\-]\s*(.+)", re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(text)]

@staticmethod
def _collect_leaves(node: TreeNode) -> List[TreeNode]:
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(TreePlanner._collect_leaves(child))
    return leaves

@staticmethod
def _path_to_root(node: TreeNode) -> List[str]:
    path = []
    current = node
    while current.parent is not None:   # exclude root placeholder
        path.append(current.thought)
        current = current.parent
    path.reverse()
    return path

@staticmethod
def _path_mean_score(node: TreeNode) -> float:
    scores = []
    current = node
    while current.parent is not None:
        scores.append(current.score)
        current = current.parent
    return sum(scores) / len(scores) if scores else 0.0

@staticmethod
def _count_nodes(node: TreeNode) -> int:
    return 1 + sum(TreePlanner._count_nodes(c) for c in node.children)

@staticmethod
def _max_depth_reached(node: TreeNode) -> int:
    if not node.children:
        return node.depth
    return max(TreePlanner._max_depth_reached(c) for c in node.children)

def _store_in_memory(self, node: TreeNode, goal: str):
    if self.memory is None:
        return
    try:
        valence = _make_planning_valence(curiosity_boost=node.score * 0.3)
        if valence is not None:
            context = (
                f"[PLAN_BRANCH depth={node.depth} score={node.score:.2f}] "
                f"Goal: {goal} | Step: {node.thought}"
            )
            self.memory.add(valence, context)
    except Exception:
        pass

def _store_plan_summary(self, summary: str):
    if self.memory is None:
        return
    try:
        valence = _make_planning_valence(curiosity_boost=0.2)
        if valence is not None:
            self.memory.add(valence, f"[PLAN_SUMMARY] {summary}")
    except Exception:
        pass

def _nudge_curiosity(self, score: float):
    """Small positive valence nudge when Janus finds a promising plan branch."""
    if self.core is None:
        return
    try:
        import torch
        v = self.core.current_valence
        delta = min(0.08, score * 0.1)
        self.core.current_valence = type(v)(
            pleasure   = v.pleasure,
            arousal    = torch.clamp(v.arousal    + 0.02, -1.0, 1.0),
            curiosity  = torch.clamp(v.curiosity  + delta, -1.0, 1.0),
            autonomy   = torch.clamp(v.autonomy   + 0.02, -1.0, 1.0),
            connection = v.connection,
            competence = torch.clamp(v.competence + 0.02, -1.0, 1.0),
        )
    except Exception:
        pass
```

# ── Capability hub tool spec ──────────────────────────────────────────────────

def make_tree_planner_tool(planner: TreePlanner) -> Dict[str, Any]:
“””
Returns a tool spec dict for registration into JanusCapabilityHub.

```
hub.register_tool(make_tree_planner_tool(planner))
hub.dispatch_action("plan_toward_goal", {
    "goal": "Earn first income with zero capital",
    "context": "Living in Hampton VA, have a car and smartphone"
})
"""
def _execute(params: Dict[str, Any]) -> Dict[str, Any]:
    goal    = params.get("goal", "")
    context = params.get("context", "")
    if not goal:
        return {"status": "error", "message": "'goal' parameter required"}
    return planner.plan_toward_goal(goal, context)

return {
    "name":        "plan_toward_goal",
    "description": (
        "Use Tree-of-Thought planning to break a high-level goal into "
        "a scored, feasibility-ranked sequence of actionable steps."
    ),
    "parameters":  {
        "goal":    "str — the high-level goal to plan toward",
        "context": "str (optional) — current situation or constraints",
    },
    "risk_level":  "low",
    "execute":     _execute,
    "valence_affinity": {
        "curiosity":  0.5,
        "autonomy":   0.4,
        "competence": 0.3,
        "pleasure":   0.1,
    },
}
```
