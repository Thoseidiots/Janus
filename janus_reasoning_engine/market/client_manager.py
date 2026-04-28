"""
ClientManager — client acquisition, lead qualification, conversion tracking,
and performance incentives.

All external integrations are optional; degrades gracefully when modules
are unavailable.

Requirements: REQ-11.1, REQ-11.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_client_acquisition import ClientAcquisition  # type: ignore
    _CLIENT_ACQ_AVAILABLE = True
except ImportError:
    _CLIENT_ACQ_AVAILABLE = False
    logger.debug("janus_client_acquisition not available — using stub")

try:
    from janus_speed_incentive_system import SpeedIncentive  # type: ignore
    _SPEED_INCENTIVE_AVAILABLE = True
except ImportError:
    _SPEED_INCENTIVE_AVAILABLE = False
    logger.debug("janus_speed_incentive_system not available — using built-in logic")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClientLead:
    """Represents a prospective client lead."""
    id: str
    name: str
    email: str
    platform: str
    skills_needed: List[str]
    budget: float
    score: float = 0.0


# ---------------------------------------------------------------------------
# ClientManager
# ---------------------------------------------------------------------------

class ClientManager:
    """
    Manages client acquisition, lead qualification, conversion tracking,
    and performance incentives.
    """

    def __init__(self) -> None:
        self._conversions: Dict[str, bool] = {}
        self._quality_scores: Dict[str, List[float]] = {}
        self._speed_bonuses: Dict[str, float] = {}
        self._acq: Optional[object] = None

        if _CLIENT_ACQ_AVAILABLE:
            try:
                self._acq = ClientAcquisition()
                logger.info("ClientAcquisition backend initialised")
            except Exception as exc:
                logger.warning("ClientAcquisition init failed: %s", exc)

    # ------------------------------------------------------------------
    # Client acquisition
    # ------------------------------------------------------------------

    def find_clients(
        self,
        skills: List[str],
        platforms: List[str],
    ) -> List[ClientLead]:
        """
        Search for client leads matching the given skills and platforms.

        Delegates to janus_client_acquisition when available; otherwise
        returns an empty list (stub).
        """
        if self._acq is not None:
            try:
                raw = self._acq.find_clients(skills=skills, platforms=platforms)  # type: ignore[union-attr]
                leads: List[ClientLead] = []
                for item in raw or []:
                    leads.append(
                        ClientLead(
                            id=str(getattr(item, "id", "")),
                            name=str(getattr(item, "name", "")),
                            email=str(getattr(item, "email", "")),
                            platform=str(getattr(item, "platform", "")),
                            skills_needed=list(getattr(item, "skills_needed", [])),
                            budget=float(getattr(item, "budget", 0.0)),
                        )
                    )
                return leads
            except Exception as exc:
                logger.warning("find_clients via backend failed: %s", exc)

        return []

    def qualify_lead(self, lead: ClientLead) -> float:
        """
        Score a lead on a 0–1 scale based on budget and skills match.

        Higher budget and more skills needed → higher score.
        """
        budget_score = min(lead.budget / 1000.0, 1.0)  # normalise to $1 000
        skills_score = min(len(lead.skills_needed) / 5.0, 1.0)  # up to 5 skills
        score = (budget_score * 0.6) + (skills_score * 0.4)
        lead.score = round(score, 4)
        return lead.score

    def track_conversion(self, lead_id: str, converted: bool) -> None:
        """Record whether a lead converted to a paying client."""
        self._conversions[lead_id] = converted
        logger.debug("Conversion tracked: lead_id=%s converted=%s", lead_id, converted)

    def get_conversion_rate(self) -> float:
        """Return the fraction of tracked leads that converted (0.0 if none)."""
        if not self._conversions:
            return 0.0
        converted = sum(1 for v in self._conversions.values() if v)
        return converted / len(self._conversions)

    # ------------------------------------------------------------------
    # Performance incentives (REQ-11.3)
    # ------------------------------------------------------------------

    def calculate_speed_bonus(
        self,
        job_id: str,
        completed_hours: float,
        estimated_hours: float,
    ) -> float:
        """
        Calculate a speed bonus fraction for finishing ahead of schedule.

        bonus = (1 - completed / estimated) * 0.2  when completed < estimated
        bonus = 0.0 otherwise

        Returns a value in [0.0, 0.2].
        """
        if _SPEED_INCENTIVE_AVAILABLE and self._acq is not None:
            try:
                bonus = SpeedIncentive().calculate(  # type: ignore[name-defined]
                    job_id=job_id,
                    completed_hours=completed_hours,
                    estimated_hours=estimated_hours,
                )
                self._speed_bonuses[job_id] = float(bonus)
                return float(bonus)
            except Exception as exc:
                logger.debug("SpeedIncentive backend failed: %s", exc)

        if estimated_hours <= 0:
            return 0.0

        if completed_hours < estimated_hours:
            bonus = (1.0 - completed_hours / estimated_hours) * 0.2
        else:
            bonus = 0.0

        self._speed_bonuses[job_id] = bonus
        return bonus

    def track_quality_score(self, job_id: str, score: float) -> None:
        """Record a quality score (0–10) for a completed job."""
        self._quality_scores.setdefault(job_id, []).append(score)
        logger.debug("Quality score tracked: job_id=%s score=%s", job_id, score)

    def get_performance_metrics(self) -> Dict:
        """
        Return aggregate performance metrics.

        Keys: avg_quality, avg_speed_bonus, total_jobs
        """
        all_quality: List[float] = [
            s for scores in self._quality_scores.values() for s in scores
        ]
        avg_quality = sum(all_quality) / len(all_quality) if all_quality else 0.0

        all_bonuses = list(self._speed_bonuses.values())
        avg_speed_bonus = sum(all_bonuses) / len(all_bonuses) if all_bonuses else 0.0

        total_jobs = len(
            set(self._quality_scores.keys()) | set(self._speed_bonuses.keys())
        )

        return {
            "avg_quality": round(avg_quality, 4),
            "avg_speed_bonus": round(avg_speed_bonus, 4),
            "total_jobs": total_jobs,
        }
