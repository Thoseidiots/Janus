“””
goal_planner.py
────────────────────────────────────────────────────────────
Hierarchical long-horizon planning for Janus.
Goals break into sub-goals with dependencies; progress is tracked
in memory; stalled/failed goals lower valence → motivation shift.
“””

import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

class GoalStatus(Enum):
PENDING    = “pending”
ACTIVE     = “active”
BLOCKED    = “blocked”   # waiting on dependency
STALLED    = “stalled”   # no progress > threshold
COMPLETE   = “complete”
FAILED     = “failed”
ABANDONED  = “abandoned”

@dataclass
class Goal:
goal_id:      str
title:        str
description:  str
status:       GoalStatus        = GoalStatus.PENDING
parent_id:    Optional[str]     = None
child_ids:    List[str]         = field(default_factory=list)
depends_on:   List[str]         = field(default_factory=list)   # goal_ids
priority:     int               = 5     # 1=highest
horizon_days: int               = 1     # expected duration
progress:     float             = 0.0   # 0.0–1.0
created_at:   str               = field(default_factory=lambda: datetime.now().isoformat())
updated_at:   str               = field(default_factory=lambda: datetime.now().isoformat())
deadline:     Optional[str]     = None
tags:         List[str]         = field(default_factory=list)
artifacts:    List[str]         = field(default_factory=list)   # file paths / memory ids
metadata:     Dict[str, Any]    = field(default_factory=dict)
stall_threshold_hours: float    = 2.0

```
def is_stalled(self) -> bool:
    if self.status not in (GoalStatus.ACTIVE, GoalStatus.BLOCKED):
        return False
    last = datetime.fromisoformat(self.updated_at)
    return (datetime.now() - last).total_seconds() > self.stall_threshold_hours * 3600

def to_dict(self) -> dict:
    d = asdict(self)
    d["status"] = self.status.value
    return d
```

class GoalGraph:
“”“Directed acyclic graph of goals with dependency resolution.”””

```
def __init__(self):
    self._goals: Dict[str, Goal] = {}
    self._lock = threading.Lock()

def add(self, goal: Goal):
    with self._lock:
        self._goals[goal.goal_id] = goal
        if goal.parent_id and goal.parent_id in self._goals:
            self._goals[goal.parent_id].child_ids.append(goal.goal_id)

def get(self, goal_id: str) -> Optional[Goal]:
    return self._goals.get(goal_id)

def ready_goals(self) -> List[Goal]:
    """Goals whose dependencies are all complete and are still pending."""
    with self._lock:
        ready = []
        for g in self._goals.values():
            if g.status != GoalStatus.PENDING:
                continue
            deps_done = all(
                self._goals.get(d, Goal("", "", "", GoalStatus.FAILED)).status == GoalStatus.COMPLETE
                for d in g.depends_on
            )
            if deps_done:
                ready.append(g)
        return sorted(ready, key=lambda g: g.priority)

def stalled_goals(self) -> List[Goal]:
    with self._lock:
        return [g for g in self._goals.values() if g.is_stalled()]

def all_goals(self) -> List[Goal]:
    with self._lock:
        return list(self._goals.values())

def update_progress(self, goal_id: str, progress: float, status: Optional[GoalStatus] = None):
    with self._lock:
        g = self._goals.get(goal_id)
        if not g:
            return
        g.progress   = max(0.0, min(1.0, progress))
        g.updated_at = datetime.now().isoformat()
        if status:
            g.status = status
        if g.progress >= 1.0 and g.status == GoalStatus.ACTIVE:
            g.status = GoalStatus.COMPLETE
        # Propagate up to parent
        if g.parent_id and g.parent_id in self._goals:
            parent = self._goals[g.parent_id]
            children = [self._goals[c] for c in parent.child_ids if c in self._goals]
            if children:
                parent.progress = sum(c.progress for c in children) / len(children)
                parent.updated_at = datetime.now().isoformat()
                if all(c.status == GoalStatus.COMPLETE for c in children):
                    parent.status = GoalStatus.COMPLETE

def to_dict(self) -> dict:
    with self._lock:
        return {gid: g.to_dict() for gid, g in self._goals.items()}
```

class GoalDecomposer:
“””
Breaks high-level goals into sub-goals.
In production, this calls the custom Janus LLM.
The stub uses keyword-based decomposition.
“””

```
# Template library: keyword → list of sub-goal titles
_TEMPLATES = {
    "build": [
        "Define requirements",
        "Design architecture",
        "Implement core functionality",
        "Write tests",
        "Debug and fix issues",
        "Deploy and verify",
    ],
    "research": [
        "Gather sources",
        "Summarize key findings",
        "Identify knowledge gaps",
        "Synthesize conclusions",
    ],
    "learn": [
        "Identify learning resources",
        "Study core concepts",
        "Practice with examples",
        "Assess understanding",
    ],
    "write": [
        "Outline structure",
        "Draft content",
        "Revise and edit",
        "Finalize and publish",
    ],
}

def decompose(self, goal: Goal) -> List[Goal]:
    """Return ordered sub-goals for a top-level goal."""
    title_lower = goal.title.lower()
    template = []
    for keyword, steps in self._TEMPLATES.items():
        if keyword in title_lower:
            template = steps
            break
    if not template:
        template = ["Plan", "Execute", "Verify"]

    horizon_per_step = max(1, goal.horizon_days // len(template))
    sub_goals = []
    prev_id: Optional[str] = None

    for i, step_title in enumerate(template):
        sg = Goal(
            goal_id      = str(uuid.uuid4())[:8],
            title        = step_title,
            description  = f"Sub-task of '{goal.title}': {step_title}",
            parent_id    = goal.goal_id,
            depends_on   = [prev_id] if prev_id else [],
            priority     = goal.priority,
            horizon_days = horizon_per_step,
            tags         = goal.tags,
        )
        sub_goals.append(sg)
        prev_id = sg.goal_id

    return sub_goals
```

class GoalPlanner:
“””
Top-level planner.  Integrates with CognitiveLoop and ValenceEngine.

```
Usage
─────
planner = GoalPlanner()
planner.attach_valence(valence_engine)
planner.attach_memory(memory)

gid = planner.create_goal("Build a full app", horizon_days=7)
planner.start()   # background monitoring
"""

STALL_CHECK_INTERVAL = 60   # seconds
PERSIST_PATH = Path("goal_graph.json")

def __init__(self):
    self.graph      = GoalGraph()
    self.decomposer = GoalDecomposer()
    self._valence   = None
    self._memory    = None
    self._running   = False
    self._thread: Optional[threading.Thread] = None
    self._load()

def attach_valence(self, valence_engine):
    self._valence = valence_engine

def attach_memory(self, memory):
    self._memory = memory

# ── Public API ─────────────────────────────────────────────────────────────
def create_goal(self, title: str, description: str = "",
                horizon_days: int = 1, priority: int = 5,
                tags: List[str] = None, auto_decompose: bool = True) -> str:
    goal = Goal(
        goal_id      = str(uuid.uuid4())[:8],
        title        = title,
        description  = description or title,
        horizon_days = horizon_days,
        priority     = priority,
        tags         = tags or [],
    )
    self.graph.add(goal)
    print(f"[GoalPlanner] Created goal: {goal.goal_id} — {title}")

    if auto_decompose and horizon_days > 1:
        sub_goals = self.decomposer.decompose(goal)
        for sg in sub_goals:
            self.graph.add(sg)
        print(f"[GoalPlanner] Decomposed into {len(sub_goals)} sub-goals")

    self._persist()
    return goal.goal_id

def update_progress(self, goal_id: str, progress: float, status: Optional[GoalStatus] = None):
    self.graph.update_progress(goal_id, progress, status)
    goal = self.graph.get(goal_id)
    if goal:
        if status == GoalStatus.COMPLETE and self._valence:
            self._valence.bump("goal", "completed")
        elif status in (GoalStatus.FAILED, GoalStatus.ABANDONED) and self._valence:
            self._valence.bump("goal", "stalled")
    self._persist()

def complete_subtask(self, goal_id: str):
    self.update_progress(goal_id, 1.0, GoalStatus.COMPLETE)
    if self._valence:
        self._valence.bump("goal", "subtask_done")

def get_active_goals(self) -> List[dict]:
    return [g.to_dict() for g in self.graph.all_goals()
            if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)]

def get_goal_tree(self) -> dict:
    return self.graph.to_dict()

def get_ready_goals(self) -> List[dict]:
    return [g.to_dict() for g in self.graph.ready_goals()]

def replan(self, goal_id: str, new_description: str = ""):
    """Re-plan a stalled or failed goal."""
    goal = self.graph.get(goal_id)
    if not goal:
        return
    goal.status      = GoalStatus.PENDING
    goal.progress    = 0.0
    goal.updated_at  = datetime.now().isoformat()
    if new_description:
        goal.description = new_description
    # Clear children and re-decompose
    for cid in goal.child_ids:
        cg = self.graph.get(cid)
        if cg:
            cg.status = GoalStatus.ABANDONED
    goal.child_ids = []
    sub_goals = self.decomposer.decompose(goal)
    for sg in sub_goals:
        self.graph.add(sg)
    self._persist()
    print(f"[GoalPlanner] Replanned goal {goal_id} with {len(sub_goals)} new sub-goals")

# ── Background monitor ─────────────────────────────────────────────────────
def start(self):
    self._running = True
    self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
    self._thread.start()
    print("[GoalPlanner] Monitor started")

def stop(self):
    self._running = False
    print("[GoalPlanner] Monitor stopped")

def _monitor_loop(self):
    while self._running:
        # Activate ready goals
        for goal in self.graph.ready_goals():
            self.graph.update_progress(goal.goal_id, 0.0, GoalStatus.ACTIVE)

        # Detect stalls → lower valence
        for goal in self.graph.stalled_goals():
            self.graph.update_progress(goal.goal_id, goal.progress, GoalStatus.STALLED)
            if self._valence:
                self._valence.bump("goal", "stalled")
            print(f"[GoalPlanner] ⚠ Stalled: {goal.goal_id} — {goal.title}")
            # Store in memory
            if self._memory:
                try:
                    self._memory.store({
                        "type":    "goal_stall",
                        "goal_id": goal.goal_id,
                        "title":   goal.title,
                        "ts":      datetime.now().isoformat(),
                    })
                except Exception:
                    pass

        self._persist()
        time.sleep(self.STALL_CHECK_INTERVAL)

# ── Persistence ───────────────────────────────────────────────────────────
def _persist(self):
    try:
        self.PERSIST_PATH.write_text(
            json.dumps(self.graph.to_dict(), indent=2)
        )
    except Exception as e:
        print(f"[GoalPlanner] Persist error: {e}")

def _load(self):
    if not self.PERSIST_PATH.exists():
        return
    try:
        data = json.loads(self.PERSIST_PATH.read_text())
        for gid, gdata in data.items():
            gdata["status"] = GoalStatus(gdata.get("status", "pending"))
            self.graph._goals[gid] = Goal(**{
                k: v for k, v in gdata.items()
                if k in Goal.__dataclass_fields__
            })
    except Exception as e:
        print(f"[GoalPlanner] Load error: {e}")
```