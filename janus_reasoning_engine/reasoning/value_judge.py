"""
Value Judgment and Tradeoffs module for the Janus Autonomous Reasoning Engine.

Evaluates options across multiple value dimensions (money, skills, reputation,
time) and selects the best option for long-term success.

**Validates: Requirements REQ-5.3**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default weights for the composite score
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "money": 0.40,
    "skills": 0.25,
    "reputation": 0.20,
    "time": 0.15,
}


@dataclass
class ValueScore:
    """
    Multi-dimensional value score for an option.

    All individual scores are in [0.0, 1.0].
    ``composite`` is the weighted sum.
    """
    money_score: float = 0.0
    skill_score: float = 0.0
    reputation_score: float = 0.0
    time_score: float = 0.0
    composite: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ValueScore(composite={self.composite:.3f}, "
            f"money={self.money_score:.2f}, skill={self.skill_score:.2f}, "
            f"reputation={self.reputation_score:.2f}, time={self.time_score:.2f})"
        )


class ValueJudge:
    """
    Evaluates options across money, skills, reputation, and time dimensions.

    Usage::

        judge = ValueJudge()
        score = judge.evaluate({"money": 0.8, "skills": 0.5, "reputation": 0.6, "time": 0.7})
        ranked = judge.compare([option_a, option_b, option_c])
        best = judge.optimize_long_term([option_a, option_b], horizon_days=60)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            weights: Custom weights dict with keys money/skills/reputation/time.
                     Values are normalised to sum to 1.0 automatically.
        """
        raw = weights or _DEFAULT_WEIGHTS.copy()
        self.weights = self._normalise_weights(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        option: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> ValueScore:
        """
        Evaluate a single option and return its ValueScore.

        The option dict should contain any subset of:
        - ``money`` (float 0–1): earning potential
        - ``skills`` (float 0–1): skill development value
        - ``reputation`` (float 0–1): reputation impact
        - ``time`` (float 0–1): time efficiency (1 = very efficient)

        Missing keys default to 0.0.

        Args:
            option: Dict describing the option's value dimensions.
            weights: Override instance weights for this evaluation.

        Returns:
            ValueScore with individual and composite scores.
        """
        w = self._normalise_weights(weights) if weights else self.weights

        money = float(option.get("money", 0.0))
        skills = float(option.get("skills", 0.0))
        reputation = float(option.get("reputation", 0.0))
        time = float(option.get("time", 0.0))

        # Clamp to [0, 1]
        money = max(0.0, min(1.0, money))
        skills = max(0.0, min(1.0, skills))
        reputation = max(0.0, min(1.0, reputation))
        time = max(0.0, min(1.0, time))

        composite = (
            w.get("money", 0.0) * money
            + w.get("skills", 0.0) * skills
            + w.get("reputation", 0.0) * reputation
            + w.get("time", 0.0) * time
        )

        return ValueScore(
            money_score=round(money, 4),
            skill_score=round(skills, 4),
            reputation_score=round(reputation, 4),
            time_score=round(time, 4),
            composite=round(composite, 4),
        )

    def compare(
        self,
        options: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Dict[str, Any], ValueScore]]:
        """
        Evaluate and rank a list of options, best first.

        Args:
            options: List of option dicts.
            weights: Optional weight override.

        Returns:
            List of (option, ValueScore) tuples sorted by composite score desc.
        """
        scored = [(opt, self.evaluate(opt, weights)) for opt in options]
        scored.sort(key=lambda pair: pair[1].composite, reverse=True)
        return scored

    def optimize_long_term(
        self,
        options: List[Dict[str, Any]],
        horizon_days: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the option with the best long-term compounding value.

        Options with higher ``skills`` scores are preferred over longer
        horizons because skill development compounds over time.

        The long-term composite is computed as::

            lt_score = composite + skill_bonus * (horizon_days / 365)

        where ``skill_bonus = skill_score * 0.5``.

        Args:
            options: List of option dicts.
            horizon_days: Planning horizon in days.

        Returns:
            The option with the highest long-term score, or None if empty.
        """
        if not options:
            return None

        horizon_factor = horizon_days / 365.0

        best_option = None
        best_lt_score = -1.0

        for opt in options:
            score = self.evaluate(opt)
            skill_bonus = score.skill_score * 0.5 * horizon_factor
            lt_score = score.composite + skill_bonus

            logger.debug(
                f"Option {opt.get('name', '?')!r}: composite={score.composite:.3f}, "
                f"lt_score={lt_score:.3f} (horizon={horizon_days}d)"
            )

            if lt_score > best_lt_score:
                best_lt_score = lt_score
                best_option = opt

        return best_option

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalise weights so they sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return _DEFAULT_WEIGHTS.copy()
        return {k: v / total for k, v in weights.items()}
