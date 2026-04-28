"""
multi_step_planner.py
=====================
Multi-step planning for the Janus Reasoning Engine.

Breaks complex goals into ordered, actionable steps with contingency handling.
Uses JanusGPT for LLM-based planning with a heuristic fallback when the model
is unavailable or returns unparseable output.

Requirements: REQ-4.1, REQ-5.1
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class PlanStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single actionable step within a plan."""
    id: str
    description: str
    tool_type: str                          # e.g. "browser", "code_execution"
    estimated_minutes: float
    dependencies: List[str] = field(default_factory=list)   # step IDs
    status: StepStatus = StepStatus.PENDING
    contingency: Optional[str] = None      # fallback description on failure
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Plan:
    """A multi-step plan for achieving a goal."""
    id: str
    goal_description: str
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def pending_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def next_step(self) -> Optional[PlanStep]:
        """Return the first pending step whose dependencies are all completed."""
        completed_ids = {
            s.id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                return step
        return None

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)


# ── Heuristic templates ───────────────────────────────────────────────────────

_KEYWORD_TEMPLATES: List[Dict[str, Any]] = [
    {
        "keywords": ["research", "find", "search", "discover", "explore"],
        "steps": [
            ("Search for relevant information online", "browser", 15),
            ("Evaluate and filter results", "browser", 10),
            ("Summarise findings", "code_execution", 10),
        ],
    },
    {
        "keywords": ["code", "develop", "build", "implement", "program", "write"],
        "steps": [
            ("Analyse requirements and design solution", "code_execution", 20),
            ("Implement core functionality", "code_execution", 60),
            ("Test and validate implementation", "code_execution", 20),
            ("Document and deliver", "file_manipulation", 10),
        ],
    },
    {
        "keywords": ["earn", "money", "job", "freelance", "work", "client"],
        "steps": [
            ("Search for relevant opportunities", "browser", 15),
            ("Evaluate and select best opportunity", "browser", 10),
            ("Prepare and submit application or proposal", "browser", 20),
            ("Complete the work", "autonomous_worker", 60),
            ("Deliver and collect payment", "browser", 10),
        ],
    },
    {
        "keywords": ["learn", "study", "skill", "tutorial", "course"],
        "steps": [
            ("Find learning resources", "browser", 10),
            ("Study core concepts", "browser", 45),
            ("Practice with examples", "code_execution", 30),
            ("Store knowledge for future use", "file_manipulation", 5),
        ],
    },
    {
        "keywords": ["file", "document", "report", "write", "create"],
        "steps": [
            ("Gather required information", "browser", 15),
            ("Draft content", "code_execution", 30),
            ("Review and refine", "code_execution", 15),
            ("Save and deliver", "file_manipulation", 5),
        ],
    },
]

_DEFAULT_STEPS = [
    ("Analyse the goal and gather context", "browser", 15),
    ("Plan detailed approach", "code_execution", 10),
    ("Execute primary action", "autonomous_worker", 45),
    ("Verify results", "code_execution", 10),
    ("Document outcome", "file_manipulation", 5),
]


def _heuristic_steps(goal: str) -> List[Dict[str, Any]]:
    """Return step templates based on keyword matching against the goal."""
    goal_lower = goal.lower()
    for template in _KEYWORD_TEMPLATES:
        if any(kw in goal_lower for kw in template["keywords"]):
            return [
                {"description": d, "tool_type": t, "estimated_minutes": m}
                for d, t, m in template["steps"]
            ]
    return [
        {"description": d, "tool_type": t, "estimated_minutes": m}
        for d, t, m in _DEFAULT_STEPS
    ]


# ── MultiStepPlanner ──────────────────────────────────────────────────────────

class MultiStepPlanner:
    """
    Generates multi-step plans for goals.

    Attempts to use JanusGPT for richer plans; falls back to keyword-based
    heuristics when the model is unavailable or returns invalid output.
    """

    def __init__(self, janus_gpt=None) -> None:
        self._gpt = janus_gpt

    # ── Public API ────────────────────────────────────────────────────────────

    def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """
        Create a multi-step plan for *goal*.

        Tries JanusGPT first; falls back to heuristics on any error.

        Args:
            goal:    High-level goal description.
            context: Optional context dict (skills, resources, etc.).

        Returns:
            A Plan with 3-5 ordered PlanStep objects.
        """
        context = context or {}
        steps_data: Optional[List[Dict[str, Any]]] = None

        if self._gpt is not None:
            steps_data = self._llm_steps(goal, context)

        if not steps_data:
            logger.debug("Using heuristic fallback for goal: %s", goal)
            steps_data = _heuristic_steps(goal)

        steps = self._build_steps(steps_data)
        plan = Plan(
            id=str(uuid.uuid4()),
            goal_description=goal,
            steps=steps,
            status=PlanStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={"context": context, "source": "llm" if self._gpt else "heuristic"},
        )
        logger.info("Created plan %s with %d steps for goal: %s", plan.id, len(steps), goal)
        return plan

    def add_contingency(self, step: PlanStep, failure_scenario: str) -> PlanStep:
        """
        Attach a contingency description to *step*.

        Args:
            step:             The PlanStep to augment.
            failure_scenario: Human-readable description of the fallback action.

        Returns:
            The same PlanStep with contingency set (mutated in-place and returned).
        """
        step.contingency = failure_scenario
        logger.debug("Added contingency to step %s: %s", step.id, failure_scenario)
        return step

    # ── Private helpers ───────────────────────────────────────────────────────

    def _llm_steps(
        self,
        goal: str,
        context: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Ask JanusGPT to generate steps; return None on any failure."""
        prompt = (
            f"Create a step-by-step plan to achieve the following goal:\n"
            f"Goal: {goal}\n"
            f"Context: {json.dumps(context)}\n\n"
            "Return a JSON array of steps. Each step must have:\n"
            '  "description": string\n'
            '  "tool_type": one of browser|code_execution|file_manipulation|computer_use|autonomous_worker\n'
            '  "estimated_minutes": number\n'
            "Return ONLY the JSON array, no other text."
        )
        try:
            raw = self._gpt.generate(prompt)
            # Extract JSON array from the response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                logger.warning("JanusGPT response contained no JSON array")
                return None
            parsed = json.loads(raw[start:end])
            if not isinstance(parsed, list) or not parsed:
                return None
            # Validate each step has required fields
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "description" in item:
                    valid.append({
                        "description": str(item.get("description", "")),
                        "tool_type": str(item.get("tool_type", "browser")),
                        "estimated_minutes": float(item.get("estimated_minutes", 15)),
                    })
            return valid if valid else None
        except Exception as exc:
            logger.warning("JanusGPT planning failed (%s); using heuristics", exc)
            return None

    @staticmethod
    def _build_steps(steps_data: List[Dict[str, Any]]) -> List[PlanStep]:
        """Convert raw step dicts into PlanStep objects with sequential dependencies."""
        steps: List[PlanStep] = []
        for i, data in enumerate(steps_data):
            step = PlanStep(
                id=str(uuid.uuid4()),
                description=data.get("description", f"Step {i + 1}"),
                tool_type=data.get("tool_type", "browser"),
                estimated_minutes=float(data.get("estimated_minutes", 15)),
                # Each step depends on the previous one (sequential by default)
                dependencies=[steps[i - 1].id] if i > 0 else [],
            )
            steps.append(step)
        return steps
