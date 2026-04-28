"""
opportunity_scorer.py
=====================
Opportunity evaluation and exploration/exploitation strategy for the
Janus Reasoning Engine.

Defines the Opportunity dataclass and OpportunityScorer which:
- Scores opportunities on multiple criteria
- Queries episodic memory for similar past experiences
- Ranks opportunities by composite score
- Implements epsilon-greedy exploration vs exploitation
- Tracks niche expertise (platform + skill → success rate)
- Adds curiosity bonus for novel opportunity types

**Validates: Requirements REQ-2.2, REQ-2.3, REQ-5.3, REQ-5.4, REQ-6.1**
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("janus.discovery.scorer")


# ---------------------------------------------------------------------------
# Opportunity dataclass
# ---------------------------------------------------------------------------

@dataclass
class Opportunity:
    """
    Represents a discovered earning opportunity.

    Attributes:
        id: Unique identifier.
        title: Short title of the opportunity.
        description: Full description.
        platform: Source platform (e.g. "upwork", "fiverr", "github").
        url: Direct URL to the opportunity.
        earning_potential: Estimated earnings in USD (0 if unknown).
        required_skills: Skills needed to complete the work.
        time_estimate: Estimated hours to complete (0 if unknown).
        competition_level: 0.0 (no competition) – 1.0 (very competitive).
        reputation_impact: -1.0 (harmful) – 1.0 (very positive).
        raw_data: Original data from the source adapter.
        score: Composite score assigned by OpportunityScorer (set after scoring).
        discovered_at: When this opportunity was found.
    """

    id: str
    title: str
    description: str
    platform: str
    url: str
    earning_potential: float = 0.0
    required_skills: List[str] = field(default_factory=list)
    time_estimate: float = 0.0          # hours
    competition_level: float = 0.5      # 0–1
    reputation_impact: float = 0.0      # -1 to 1
    raw_data: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Opportunity":
        """Create an Opportunity from a raw dict returned by a source adapter."""
        return cls(
            id=raw.get("id") or str(uuid.uuid4()),
            title=raw.get("title", "Untitled"),
            description=raw.get("description", ""),
            platform=raw.get("platform", "unknown"),
            url=raw.get("url", ""),
            earning_potential=float(raw.get("budget", raw.get("earning_potential", 0)) or 0),
            required_skills=raw.get("required_skills", []),
            time_estimate=float(raw.get("time_estimate", 0) or 0),
            competition_level=float(raw.get("competition_level", 0.5)),
            reputation_impact=float(raw.get("reputation_impact", 0.0)),
            raw_data=raw.get("raw_data", {}),
        )


# ---------------------------------------------------------------------------
# Scoring weights (configurable)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "earning_potential": 0.30,
    "skill_match":       0.25,
    "learning_value":    0.15,
    "time_efficiency":   0.15,
    "competition":       0.10,
    "reputation":        0.05,
}


# ---------------------------------------------------------------------------
# OpportunityScorer
# ---------------------------------------------------------------------------

class OpportunityScorer:
    """
    Scores and ranks opportunities using multiple criteria.

    Scoring criteria:
    - Earning potential (normalised against a configurable ceiling)
    - Skill match (0–1, fraction of required skills Janus has)
    - Learning value (estimated from skill novelty)
    - Time efficiency (earning_potential / time_estimate)
    - Competition level (inverted — lower competition is better)
    - Reputation impact (normalised from -1..1 to 0..1)

    Also queries episodic memory for similar past opportunities to adjust
    scores based on historical success rates.

    **Validates: Requirements REQ-2.2, REQ-5.3, REQ-6.1**
    """

    def __init__(
        self,
        known_skills: Optional[List[str]] = None,
        episodic_memory: Any = None,
        weights: Optional[Dict[str, float]] = None,
        earning_ceiling: float = 1000.0,
    ):
        """
        Args:
            known_skills: Skills Janus currently has.
            episodic_memory: EpisodicMemory instance for past-experience lookup.
            weights: Scoring weights (defaults to DEFAULT_WEIGHTS).
            earning_ceiling: USD value treated as "maximum" for normalisation.
        """
        self.known_skills: List[str] = [s.lower() for s in (known_skills or [])]
        self.episodic_memory = episodic_memory
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.earning_ceiling = max(earning_ceiling, 1.0)

        # Niche expertise: (platform, skill_category) → success_rate
        # Updated by record_outcome()
        self._niche_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

        # Seen opportunity types for curiosity bonus
        self._seen_types: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_opportunity(self, opp: Opportunity) -> float:
        """
        Calculate and assign a composite score to an opportunity.

        Args:
            opp: Opportunity to score.

        Returns:
            Composite score in [0, 1].
        """
        components = self._compute_components(opp)
        composite = sum(
            self.weights.get(k, 0.0) * v for k, v in components.items()
        )
        composite = max(0.0, min(1.0, composite))

        # Adjust with episodic memory if available
        memory_boost = self._memory_boost(opp)
        composite = min(1.0, composite + memory_boost)

        opp.score = composite
        return composite

    def score_and_rank(self, opportunities: List[Opportunity]) -> List[Opportunity]:
        """
        Score all opportunities and return them sorted best-first.

        Args:
            opportunities: List of opportunities to evaluate.

        Returns:
            Sorted list (highest score first).
        """
        for opp in opportunities:
            self.score_opportunity(opp)
        return sorted(opportunities, key=lambda o: o.score, reverse=True)

    def record_outcome(
        self,
        opportunity: Opportunity,
        success: bool,
        earnings: float = 0.0,
    ) -> None:
        """
        Record the outcome of pursuing an opportunity.

        Updates niche expertise stats used for future scoring.

        Args:
            opportunity: The opportunity that was pursued.
            success: Whether it was successful.
            earnings: Actual earnings (USD).
        """
        for skill in opportunity.required_skills or ["general"]:
            key = (opportunity.platform, skill.lower())
            if key not in self._niche_stats:
                self._niche_stats[key] = {"wins": 0, "total": 0}
            self._niche_stats[key]["total"] += 1
            if success:
                self._niche_stats[key]["wins"] += 1

        logger.debug(
            f"[OpportunityScorer] Recorded outcome for '{opportunity.title}': "
            f"success={success}, earnings={earnings}"
        )

    def niche_success_rate(self, platform: str, skill: str) -> float:
        """
        Return historical success rate for a platform+skill niche.

        Args:
            platform: Platform name.
            skill: Skill category.

        Returns:
            Success rate in [0, 1]. Returns 0.5 (neutral) if no data.
        """
        key = (platform, skill.lower())
        stats = self._niche_stats.get(key)
        if not stats or stats["total"] == 0:
            return 0.5  # neutral prior
        return stats["wins"] / stats["total"]

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _compute_components(self, opp: Opportunity) -> Dict[str, float]:
        """Compute individual scoring components (all in [0, 1])."""
        return {
            "earning_potential": self._score_earning(opp),
            "skill_match":       self._score_skill_match(opp),
            "learning_value":    self._score_learning(opp),
            "time_efficiency":   self._score_time(opp),
            "competition":       self._score_competition(opp),
            "reputation":        self._score_reputation(opp),
        }

    def _score_earning(self, opp: Opportunity) -> float:
        """Normalise earning potential against ceiling."""
        return min(opp.earning_potential / self.earning_ceiling, 1.0)

    def _score_skill_match(self, opp: Opportunity) -> float:
        """Fraction of required skills Janus already has."""
        if not opp.required_skills:
            return 0.5  # unknown requirements — neutral
        matched = sum(
            1 for s in opp.required_skills if s.lower() in self.known_skills
        )
        return matched / len(opp.required_skills)

    def _score_learning(self, opp: Opportunity) -> float:
        """
        Learning value: high when skills are new (not yet known).
        Complement of skill_match — novel skills = high learning value.
        """
        if not opp.required_skills:
            return 0.0
        novel = sum(
            1 for s in opp.required_skills if s.lower() not in self.known_skills
        )
        return novel / len(opp.required_skills)

    def _score_time(self, opp: Opportunity) -> float:
        """
        Time efficiency: earning_potential per hour, normalised.
        If time_estimate is 0, assume 1 hour to avoid division by zero.
        """
        hours = opp.time_estimate if opp.time_estimate > 0 else 1.0
        hourly = opp.earning_potential / hours
        # Normalise: $100/hr = 1.0
        return min(hourly / 100.0, 1.0)

    def _score_competition(self, opp: Opportunity) -> float:
        """Invert competition level — lower competition is better."""
        return 1.0 - max(0.0, min(1.0, opp.competition_level))

    def _score_reputation(self, opp: Opportunity) -> float:
        """Normalise reputation_impact from [-1, 1] to [0, 1]."""
        return (opp.reputation_impact + 1.0) / 2.0

    def _memory_boost(self, opp: Opportunity) -> float:
        """
        Query episodic memory for similar past opportunities and return
        a small score adjustment based on historical success rate.

        Returns a value in [-0.1, +0.1].
        """
        if self.episodic_memory is None:
            return 0.0

        try:
            query = f"{opp.platform} {opp.title} {' '.join(opp.required_skills[:3])}"
            similar = self.episodic_memory.retrieve_similar_experiences(
                query, limit=5, similarity_threshold=0.2
            )
            if not similar:
                return 0.0

            from janus_reasoning_engine.memory.episodic_memory import OutcomeType
            successes = sum(
                1 for e in similar if e.outcome_type == OutcomeType.SUCCESS
            )
            rate = successes / len(similar)
            # Map [0, 1] → [-0.1, +0.1]
            return (rate - 0.5) * 0.2
        except Exception as exc:
            logger.debug(f"[OpportunityScorer] Memory boost failed: {exc}")
            return 0.0


# ---------------------------------------------------------------------------
# Exploration vs Exploitation
# ---------------------------------------------------------------------------

class ExplorationStrategy:
    """
    Epsilon-greedy exploration vs exploitation strategy.

    - With probability (1 - epsilon): exploit — pick the highest-scored
      opportunity (adjusted by niche expertise).
    - With probability epsilon: explore — pick a random opportunity,
      with a curiosity bonus for types not seen before.

    Also tracks niche expertise (platform + skill → success rate) and
    applies a curiosity bonus for novel opportunity types.

    **Validates: Requirements REQ-2.3, REQ-5.4**
    """

    def __init__(
        self,
        scorer: OpportunityScorer,
        epsilon: float = 0.15,
        curiosity_bonus: float = 0.1,
    ):
        """
        Args:
            scorer: OpportunityScorer instance (shares niche stats).
            epsilon: Exploration probability (0–1). Default 0.15.
            curiosity_bonus: Score bonus for novel opportunity types.
        """
        self.scorer = scorer
        self.epsilon = max(0.0, min(1.0, epsilon))
        self.curiosity_bonus = curiosity_bonus
        self._seen_types: set = set()

    def select_opportunity(
        self,
        opportunities: List[Opportunity],
        epsilon: Optional[float] = None,
    ) -> Optional[Opportunity]:
        """
        Select an opportunity balancing exploitation and exploration.

        Args:
            opportunities: Ranked list of opportunities (scored).
            epsilon: Override epsilon for this call (uses instance default if None).

        Returns:
            Selected opportunity, or None if list is empty.
        """
        if not opportunities:
            return None

        eps = epsilon if epsilon is not None else self.epsilon

        if random.random() > eps:
            # Exploit: pick best opportunity, boosted by niche expertise
            return self._exploit(opportunities)
        else:
            # Explore: pick a random opportunity, preferring novel types
            return self._explore(opportunities)

    def mark_seen(self, opportunity: Opportunity) -> None:
        """
        Mark an opportunity type as seen (reduces future curiosity bonus).

        Args:
            opportunity: Opportunity that was pursued.
        """
        opp_type = self._opportunity_type(opportunity)
        self._seen_types.add(opp_type)

    def is_novel(self, opportunity: Opportunity) -> bool:
        """Return True if this opportunity type has not been seen before."""
        return self._opportunity_type(opportunity) not in self._seen_types

    def curiosity_score(self, opportunity: Opportunity) -> float:
        """
        Return curiosity bonus for an opportunity.

        Novel types get +curiosity_bonus; known types get 0.
        """
        return self.curiosity_bonus if self.is_novel(opportunity) else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exploit(self, opportunities: List[Opportunity]) -> Opportunity:
        """
        Pick the best opportunity, adjusted by niche expertise.

        Adds niche success rate as a tiebreaker on top of the base score.
        """
        def exploit_score(opp: Opportunity) -> float:
            niche = 0.0
            for skill in opp.required_skills or ["general"]:
                niche = max(niche, self.scorer.niche_success_rate(opp.platform, skill))
            # Blend base score (80%) with niche expertise (20%)
            return 0.8 * opp.score + 0.2 * niche

        return max(opportunities, key=exploit_score)

    def _explore(self, opportunities: List[Opportunity]) -> Opportunity:
        """
        Pick a random opportunity, preferring novel types.

        Novel opportunities get a curiosity bonus added to their score
        for the purpose of this selection.
        """
        def explore_score(opp: Opportunity) -> float:
            return opp.score + self.curiosity_score(opp)

        # Weighted random selection by explore_score
        scores = [max(explore_score(o), 0.001) for o in opportunities]
        total = sum(scores)
        r = random.random() * total
        cumulative = 0.0
        for opp, s in zip(opportunities, scores):
            cumulative += s
            if r <= cumulative:
                return opp
        return opportunities[-1]

    @staticmethod
    def _opportunity_type(opp: Opportunity) -> str:
        """Derive a type key from platform + primary skill."""
        primary_skill = opp.required_skills[0].lower() if opp.required_skills else "general"
        return f"{opp.platform}:{primary_skill}"
