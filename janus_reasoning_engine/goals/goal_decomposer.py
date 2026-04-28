"""
Goal decomposition using JanusGPT with heuristic fallback.

Breaks abstract high-level goals into concrete strategies and sub-goals.
Satisfies REQ-1.2 and REQ-5.3.
"""

import uuid
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from janus_reasoning_engine.core.interfaces import (
    Goal,
    GoalStatus,
    Strategy,
    StrategyStatus,
)
from janus_reasoning_engine.goals.goal_manager import GoalManagerImpl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic strategy templates used when JanusGPT is unavailable
# ---------------------------------------------------------------------------

_HEURISTIC_TEMPLATES: List[Dict[str, Any]] = [
    {
        "description": "Research and plan: gather information, identify requirements, create a detailed plan",
        "expected_value_factor": 0.3,
        "time_estimate": 2.0,
        "success_probability": 0.85,
        "steps": [
            "Research the goal domain and gather relevant information",
            "Identify key requirements and constraints",
            "Create a detailed action plan with milestones",
            "Review and refine the plan",
        ],
    },
    {
        "description": "Iterative execution: break into small tasks, execute incrementally, review after each step",
        "expected_value_factor": 0.7,
        "time_estimate": 8.0,
        "success_probability": 0.70,
        "steps": [
            "Break the goal into small, concrete tasks",
            "Execute the first task and review results",
            "Adjust approach based on feedback",
            "Continue iterating until goal is achieved",
        ],
    },
    {
        "description": "Direct approach: attempt the goal directly using available tools and knowledge",
        "expected_value_factor": 1.0,
        "time_estimate": 4.0,
        "success_probability": 0.55,
        "steps": [
            "Assess available tools and resources",
            "Attempt the goal directly",
            "Monitor progress and handle obstacles",
            "Verify completion and document outcome",
        ],
    },
]


class GoalDecomposer:
    """
    Decomposes high-level goals into strategies and sub-goals.

    Uses JanusGPT for LLM-powered decomposition when available, falling
    back to heuristic templates when the model is unavailable or fails.
    """

    def __init__(
        self,
        goal_manager: GoalManagerImpl,
        janus_gpt=None,
        num_strategies: int = 3,
    ):
        """
        Initialize the decomposer.

        Args:
            goal_manager: GoalManagerImpl for persisting sub-goals/strategies.
            janus_gpt: Optional JanusGPT instance for LLM decomposition.
            num_strategies: Number of strategies to generate per goal.
        """
        self.goal_manager = goal_manager
        self.janus_gpt = janus_gpt
        self.num_strategies = num_strategies

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose_goal(self, goal: Goal) -> List[Goal]:
        """
        Break a high-level goal into sub-goals.

        Each sub-goal corresponds to a concrete strategy option. The
        sub-goals are persisted and returned.

        Args:
            goal: High-level goal to decompose.

        Returns:
            List of sub-goals (one per strategy option).
        """
        strategies = self.generate_strategies(goal)
        sub_goals: List[Goal] = []

        for strategy in strategies:
            sub_goal = self.goal_manager.create_goal(
                description=f"[Strategy] {strategy.description}",
                priority=goal.priority * strategy.success_probability,
                expected_value=strategy.expected_value,
                parent_goal_id=goal.id,
                feasibility=strategy.success_probability,
                metadata={
                    "strategy_id": strategy.id,
                    "time_estimate": strategy.time_estimate,
                    "steps": strategy.steps,
                },
            )
            sub_goals.append(sub_goal)

        logger.info(f"Decomposed goal '{goal.description}' into {len(sub_goals)} sub-goals")
        return sub_goals

    def generate_strategies(self, goal: Goal) -> List[Strategy]:
        """
        Generate multiple strategy options for a goal.

        Tries JanusGPT first; falls back to heuristics on failure.

        Args:
            goal: Goal to generate strategies for.

        Returns:
            List of Strategy objects (not yet persisted).
        """
        strategies: List[Strategy] = []

        if self.janus_gpt is not None:
            try:
                strategies = self._generate_with_llm(goal)
                logger.info(f"Generated {len(strategies)} strategies via JanusGPT for '{goal.description}'")
            except Exception as exc:
                logger.warning(f"JanusGPT strategy generation failed ({exc}); using heuristics")
                strategies = []

        if not strategies:
            strategies = self._generate_heuristic(goal)
            logger.info(f"Generated {len(strategies)} heuristic strategies for '{goal.description}'")

        # Persist strategies
        for s in strategies:
            self.goal_manager.save_strategy(s)

        return strategies

    def evaluate_strategies(self, strategies: List[Strategy]) -> List[Tuple[Strategy, float]]:
        """
        Score each strategy by expected utility.

        Utility = expected_value * success_probability / max(time_estimate, 0.1)

        Args:
            strategies: Strategies to evaluate.

        Returns:
            List of (strategy, score) tuples sorted by score descending.
        """
        scored: List[Tuple[Strategy, float]] = []
        for s in strategies:
            score = self._utility_score(s)
            scored.append((s, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_best_strategy(self, strategies: List[Strategy]) -> Optional[Strategy]:
        """
        Select the highest-utility strategy.

        Args:
            strategies: Candidate strategies.

        Returns:
            Best strategy or None if the list is empty.
        """
        if not strategies:
            return None
        scored = self.evaluate_strategies(strategies)
        best, score = scored[0]
        logger.info(f"Selected strategy '{best.description}' (score={score:.3f})")
        return best

    # ------------------------------------------------------------------
    # LLM-based decomposition
    # ------------------------------------------------------------------

    def _generate_with_llm(self, goal: Goal) -> List[Strategy]:
        """Use JanusGPT to generate strategies via text generation."""
        prompt = self._build_decomposition_prompt(goal)

        # JanusGPT.generate returns a string
        raw_output: str = self.janus_gpt.generate(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
        )

        return self._parse_llm_strategies(raw_output, goal)

    def _build_decomposition_prompt(self, goal: Goal) -> str:
        return (
            f"You are a strategic planning assistant.\n"
            f"Goal: {goal.description}\n"
            f"Priority: {goal.priority:.2f}, Expected value: {goal.expected_value:.2f}\n\n"
            f"Generate {self.num_strategies} distinct strategies to achieve this goal.\n"
            f"For each strategy output a JSON object on its own line with keys:\n"
            f"  description, expected_value, time_estimate_hours, success_probability, steps (list)\n"
            f"Output only the JSON lines, no other text.\n"
        )

    def _parse_llm_strategies(self, raw: str, goal: Goal) -> List[Strategy]:
        """Parse JSON strategy objects from LLM output."""
        strategies: List[Strategy] = []
        now = datetime.utcnow()

        # Try to find JSON objects in the output
        json_pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)
        matches = json_pattern.findall(raw)

        for match in matches[:self.num_strategies]:
            try:
                data = json.loads(match)
                strategy = Strategy(
                    id=str(uuid.uuid4()),
                    goal_id=goal.id,
                    description=str(data.get("description", "LLM strategy")),
                    expected_value=float(data.get("expected_value", goal.expected_value * 0.8)),
                    time_estimate=float(data.get("time_estimate_hours", 4.0)),
                    success_probability=float(data.get("success_probability", 0.6)),
                    resource_requirements={},
                    status=StrategyStatus.PROPOSED,
                    steps=list(data.get("steps", [])),
                    metadata={"source": "janus_gpt"},
                    created_at=now,
                )
                strategies.append(strategy)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.debug(f"Failed to parse LLM strategy chunk: {exc}")

        return strategies

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _generate_heuristic(self, goal: Goal) -> List[Strategy]:
        """Generate strategies from built-in templates."""
        now = datetime.utcnow()
        strategies: List[Strategy] = []

        templates = _HEURISTIC_TEMPLATES[: self.num_strategies]
        for tmpl in templates:
            strategy = Strategy(
                id=str(uuid.uuid4()),
                goal_id=goal.id,
                description=tmpl["description"],
                expected_value=goal.expected_value * tmpl["expected_value_factor"],
                time_estimate=tmpl["time_estimate"],
                success_probability=tmpl["success_probability"],
                resource_requirements={},
                status=StrategyStatus.PROPOSED,
                steps=list(tmpl["steps"]),
                metadata={"source": "heuristic"},
                created_at=now,
            )
            strategies.append(strategy)

        return strategies

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _utility_score(strategy: Strategy) -> float:
        """
        Compute expected utility for a strategy.

        Utility = (expected_value * success_probability) / time_estimate
        """
        time = max(strategy.time_estimate, 0.1)
        return (strategy.expected_value * strategy.success_probability) / time
