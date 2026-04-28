"""Social simulation bridge for multi-agent ToM modeling and relationship management.

Requirements: REQ-14.2, REQ-14.3
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Optional import of social_sim from workspace root
try:
    import social_sim as _social_sim  # type: ignore
    _HAS_SOCIAL_SIM = True
except (ImportError, SyntaxError, Exception):
    _social_sim = None
    _HAS_SOCIAL_SIM = False


@dataclass
class SimResult:
    """Result of a social simulation run."""

    scenario: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    tom_updates: Dict[str, Any] = field(default_factory=dict)


class SocialSimBridge:
    """Bridge to social simulation and relationship management.

    Wraps social_sim.py when available; falls back to lightweight stubs.
    Relationships are stored in-memory (and optionally persisted to
    relationships.json).
    """

    # Default relationship values
    _DEFAULT_TRUST = 0.5
    _DEFAULT_FAMILIARITY = 0.0
    _DEFAULT_TONE = "neutral"

    _TRUST_DELTA = 0.1

    def __init__(self, relationships_path: str = "relationships.json") -> None:
        self._relationships_path = relationships_path
        self._relationships: Dict[str, Dict[str, Any]] = {}
        self._load_relationships()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulation(self, scenario: str, agent_ids: List[str]) -> SimResult:
        """Run a multi-agent social simulation for the given scenario."""
        if _HAS_SOCIAL_SIM and hasattr(_social_sim, "run_simulation"):
            raw = _social_sim.run_simulation(scenario, agent_ids)
            # Normalise whatever social_sim returns into SimResult
            if isinstance(raw, dict):
                return SimResult(
                    scenario=scenario,
                    turns=raw.get("turns", []),
                    outcome=str(raw.get("outcome", "")),
                    tom_updates=raw.get("tom_updates", {}),
                )

        # Stub: generate synthetic turns
        turns: List[Dict[str, Any]] = []
        for i, agent_id in enumerate(agent_ids):
            turns.append(
                {
                    "turn": i + 1,
                    "agent": agent_id,
                    "action": f"Agent {agent_id} responds to scenario: {scenario}",
                }
            )

        tom_updates: Dict[str, Any] = {aid: {"belief_updated": True} for aid in agent_ids}
        return SimResult(
            scenario=scenario,
            turns=turns,
            outcome=f"Simulation of '{scenario}' completed with {len(agent_ids)} agents.",
            tom_updates=tom_updates,
        )

    # ------------------------------------------------------------------
    # Relationship management
    # ------------------------------------------------------------------

    def update_relationship(self, agent_id: str, outcome: str) -> None:
        """Update relationship metrics based on interaction outcome.

        'positive' increases trust; 'negative' decreases trust.
        """
        rel = self._get_or_create(agent_id)
        if outcome == "positive":
            rel["trust"] = min(1.0, rel["trust"] + self._TRUST_DELTA)
            rel["familiarity"] = min(1.0, rel["familiarity"] + self._TRUST_DELTA)
        elif outcome == "negative":
            rel["trust"] = max(0.0, rel["trust"] - self._TRUST_DELTA)
        self._save_relationships()

    def get_relationship(self, agent_id: str) -> Dict[str, Any]:
        """Return trust, familiarity, and tone for the given agent."""
        return dict(self._get_or_create(agent_id))

    def list_relationships(self) -> List[Dict[str, Any]]:
        """Return all tracked relationships."""
        return [
            {"agent_id": aid, **rel}
            for aid, rel in self._relationships.items()
        ]

    def sleep_phase_refinement(self) -> Dict[str, Any]:
        """Run sleep-phase ToM refinement across all known relationships."""
        if _HAS_SOCIAL_SIM and hasattr(_social_sim, "sleep_phase_refinement"):
            return _social_sim.sleep_phase_refinement()

        # Stub: simulate refinement
        tom_updates: Dict[str, Any] = {}
        for agent_id, rel in self._relationships.items():
            tom_updates[agent_id] = {"trust_refined": rel["trust"]}

        return {
            "simulations_run": len(self._relationships),
            "tom_updates": tom_updates,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._relationships:
            self._relationships[agent_id] = {
                "trust": self._DEFAULT_TRUST,
                "familiarity": self._DEFAULT_FAMILIARITY,
                "tone": self._DEFAULT_TONE,
            }
        return self._relationships[agent_id]

    def _load_relationships(self) -> None:
        if os.path.exists(self._relationships_path):
            try:
                with open(self._relationships_path, "r", encoding="utf-8") as fh:
                    self._relationships = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._relationships = {}

    def _save_relationships(self) -> None:
        try:
            with open(self._relationships_path, "w", encoding="utf-8") as fh:
                json.dump(self._relationships, fh, indent=2)
        except OSError:
            pass
