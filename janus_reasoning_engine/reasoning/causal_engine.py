"""
Causal Reasoning Engine for the Janus Autonomous Reasoning Engine.

Stores cause→effect relationships as a directed graph and predicts outcomes
of actions based on learned causal models. Optionally integrates with
janus_causal_horizon for holographic causal modeling.

**Validates: Requirements REQ-5.1, REQ-13.1**
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional integration with janus_causal_horizon
try:
    from janus_causal_horizon import CausalHBMPocketDimension
    _CAUSAL_HORIZON_AVAILABLE = True
except ImportError:
    _CAUSAL_HORIZON_AVAILABLE = False
    logger.debug("janus_causal_horizon not available; using built-in causal model only.")


@dataclass
class CausalEdge:
    """A directed cause→effect relationship with a strength weight."""
    cause: str
    effect: str
    strength: float = 1.0  # 0.0 (no relation) to 1.0 (certain)
    observations: int = 1  # how many times this edge was observed

    def reinforce(self, delta: float = 0.1) -> None:
        """Increase strength (capped at 1.0)."""
        self.strength = min(1.0, self.strength + delta)
        self.observations += 1

    def weaken(self, delta: float = 0.2) -> None:
        """Decrease strength (floored at 0.0)."""
        self.strength = max(0.0, self.strength - delta)
        self.observations += 1


class CausalModel:
    """
    Directed graph of cause→effect relationships.

    Internally stored as:
        _graph[cause] = {effect: CausalEdge, ...}
    """

    def __init__(self) -> None:
        self._graph: Dict[str, Dict[str, CausalEdge]] = defaultdict(dict)

    def add_edge(self, cause: str, effect: str, strength: float = 1.0) -> CausalEdge:
        """Add or update a cause→effect edge."""
        if effect in self._graph[cause]:
            edge = self._graph[cause][effect]
            edge.reinforce()
        else:
            edge = CausalEdge(cause=cause, effect=effect, strength=strength)
            self._graph[cause][effect] = edge
        return edge

    def get_effects(self, cause: str) -> List[CausalEdge]:
        """Return all known effects for a given cause, sorted by strength desc."""
        return sorted(self._graph.get(cause, {}).values(), key=lambda e: e.strength, reverse=True)

    def weaken_edge(self, cause: str, effect: str, delta: float = 0.2) -> None:
        """Weaken a specific cause→effect edge (e.g., after a mistake)."""
        if cause in self._graph and effect in self._graph[cause]:
            self._graph[cause][effect].weaken(delta)

    def all_causes(self) -> List[str]:
        return list(self._graph.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            cause: {
                effect: {"strength": edge.strength, "observations": edge.observations}
                for effect, edge in effects.items()
            }
            for cause, effects in self._graph.items()
        }


@dataclass
class PredictedOutcome:
    """A predicted effect of an action with a confidence score."""
    effect: str
    confidence: float  # 0.0–1.0
    source: str = "causal_model"  # "causal_model" or "causal_horizon"

    def __repr__(self) -> str:
        return f"PredictedOutcome(effect={self.effect!r}, confidence={self.confidence:.2f})"


class CausalEngine:
    """
    Causal reasoning engine that learns cause→effect relationships from
    experience and predicts outcomes of actions.

    Optionally uses janus_causal_horizon for holographic signal propagation
    when available.

    Usage::

        engine = CausalEngine()
        engine.observe("send_proposal", "client_responds", strength=0.8)
        outcomes = engine.predict_outcome("send_proposal")
        engine.learn_from_mistake("send_proposal", "client_responds", "no_response")
    """

    def __init__(self, use_causal_horizon: bool = True) -> None:
        self.model = CausalModel()
        self._horizon: Optional[Any] = None

        if use_causal_horizon and _CAUSAL_HORIZON_AVAILABLE:
            try:
                self._horizon = CausalHBMPocketDimension(dim=256, propagation_speed=0.3)
                logger.info("CausalHBMPocketDimension integrated.")
            except Exception as exc:
                logger.warning(f"Failed to initialise CausalHBMPocketDimension: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, cause: str, effect: str, strength: float = 1.0) -> None:
        """
        Record a cause-effect pair.

        Args:
            cause: The action or event that caused something.
            effect: The observed effect.
            strength: How strongly this cause leads to this effect (0.0–1.0).
        """
        strength = max(0.0, min(1.0, strength))
        self.model.add_edge(cause, effect, strength)

        # Optionally emit a signal through the causal horizon
        if self._horizon is not None:
            try:
                self._horizon.emit_signal(cause, effect, {"strength": strength})
                self._horizon.update()
            except Exception as exc:
                logger.debug(f"Causal horizon signal failed (non-fatal): {exc}")

        logger.debug(f"Observed: {cause!r} → {effect!r} (strength={strength:.2f})")

    def predict_outcome(self, action: str) -> List[PredictedOutcome]:
        """
        Predict the effects of an action based on the causal model.

        Args:
            action: The action to predict outcomes for.

        Returns:
            List of PredictedOutcome sorted by confidence (highest first).
        """
        edges = self.model.get_effects(action)
        outcomes = [
            PredictedOutcome(effect=edge.effect, confidence=edge.strength)
            for edge in edges
        ]

        if not outcomes:
            logger.debug(f"No causal knowledge for action: {action!r}")

        return outcomes

    def learn_from_mistake(
        self,
        action: str,
        expected: str,
        actual: str,
        penalty: float = 0.2,
    ) -> None:
        """
        Update the causal model when a prediction was wrong.

        Weakens the (action → expected) edge and reinforces (action → actual).

        Args:
            action: The action that was taken.
            expected: The effect that was predicted/expected.
            actual: The effect that actually occurred.
            penalty: How much to weaken the wrong prediction (0.0–1.0).
        """
        # Weaken the incorrect prediction
        self.model.weaken_edge(action, expected, delta=penalty)

        # Reinforce the actual outcome
        self.model.add_edge(action, actual, strength=0.5)

        logger.info(
            f"Learned from mistake: {action!r} → expected {expected!r}, "
            f"got {actual!r}. Weakened expected, reinforced actual."
        )

    def get_model_summary(self) -> Dict[str, Any]:
        """Return a summary of the current causal model."""
        return {
            "total_causes": len(self.model.all_causes()),
            "causal_horizon_active": self._horizon is not None,
            "graph": self.model.to_dict(),
        }
