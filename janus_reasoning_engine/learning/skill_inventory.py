"""
Skill Inventory Management for the Janus Reasoning Engine.

Maintains a dynamic list of Janus's capabilities backed by SemanticMemory.
Provides methods to add/update skills, query by domain, identify strengths
and weaknesses, and plan strategic skill development.

**Validates: Requirements REQ-3.3, REQ-6.2**
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillInventory:
    """
    High-level skill inventory backed by SemanticMemory.

    Wraps SemanticMemory to provide a focused API for skill management:
    confidence tracking, domain queries, strength/weakness identification,
    and development planning.
    """

    def __init__(self, semantic_memory) -> None:
        """
        Args:
            semantic_memory: A SemanticMemory instance used for persistence.
        """
        self.semantic_memory = semantic_memory
        # Local cache: skill_name (lower) -> skill_id
        self._name_to_id: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Skill CRUD
    # ------------------------------------------------------------------

    def add_skill(
        self,
        name: str,
        confidence: float,
        domains: Optional[List[str]] = None,
        description: str = "",
    ) -> str:
        """
        Add a new skill or update confidence if it already exists.

        Args:
            name: Human-readable skill name.
            confidence: Initial confidence level (0.0–1.0).
            domains: Domain tags (e.g. ["programming", "web"]).
            description: Optional description.

        Returns:
            Skill ID.
        """
        from janus_reasoning_engine.memory.semantic_memory import SkillLevel

        confidence = max(0.0, min(1.0, confidence))
        existing_id = self._resolve_id(name)

        if existing_id:
            # Update existing skill's confidence
            self.semantic_memory.update_skill(existing_id, confidence=confidence)
            return existing_id

        level = self._confidence_to_level(confidence)
        skill_id = self.semantic_memory.add_skill(
            name=name,
            description=description or f"Skill: {name}",
            level=level,
            confidence=confidence,
            domains=domains or [],
        )
        self._name_to_id[name.lower()] = skill_id
        return skill_id

    def update_confidence(self, name: str, delta: float) -> None:
        """
        Adjust a skill's confidence by delta (positive or negative).

        If the skill doesn't exist it is created with confidence = max(0, delta).

        Args:
            name: Skill name.
            delta: Amount to add to current confidence.
        """
        skill_id = self._resolve_id(name)
        if skill_id is None:
            # Create with initial confidence
            self.add_skill(name, max(0.0, delta))
            return

        skill = self.semantic_memory.get_skill(skill_id)
        if skill is None:
            return

        new_confidence = max(0.0, min(1.0, skill.confidence + delta))
        level = self._confidence_to_level(new_confidence)
        self.semantic_memory.update_skill(skill_id, confidence=new_confidence, level=level)

    def get_skill_confidence(self, name: str) -> float:
        """
        Return the current confidence for a skill (0.0 if unknown).

        Args:
            name: Skill name.

        Returns:
            Confidence in [0.0, 1.0].
        """
        skill_id = self._resolve_id(name)
        if skill_id is None:
            return 0.0
        skill = self.semantic_memory.get_skill(skill_id)
        return skill.confidence if skill else 0.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_skills_by_domain(self, domain: str) -> list:
        """
        Return all skills belonging to a domain.

        Args:
            domain: Domain name.

        Returns:
            List of Skill objects.
        """
        return self.semantic_memory.get_skills_by_domain(domain)

    def identify_strengths(self, threshold: float = 0.7) -> list:
        """
        Return skills with confidence >= threshold.

        Args:
            threshold: Minimum confidence to be considered a strength.

        Returns:
            List of Skill objects.
        """
        all_skills = self._get_all_skills()
        return [s for s in all_skills if s.confidence >= threshold]

    def identify_weaknesses(self, threshold: float = 0.4) -> list:
        """
        Return skills with confidence <= threshold.

        Args:
            threshold: Maximum confidence to be considered a weakness.

        Returns:
            List of Skill objects.
        """
        all_skills = self._get_all_skills()
        return [s for s in all_skills if s.confidence <= threshold]

    def plan_development(self, target_domains: List[str]) -> List[str]:
        """
        Suggest skills to learn for the given target domains.

        Returns skills that are either:
        - Not yet in the inventory for those domains, or
        - Present but with low confidence (< 0.5).

        Args:
            target_domains: Domains to focus development on.

        Returns:
            List of skill names to learn/improve.
        """
        to_learn: List[str] = []
        for domain in target_domains:
            domain_skills = self.get_skills_by_domain(domain)
            weak = [s.name for s in domain_skills if s.confidence < 0.5]
            to_learn.extend(weak)

        # Deduplicate while preserving order
        seen: set = set()
        result: List[str] = []
        for name in to_learn:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def as_dict(self) -> Dict[str, float]:
        """
        Return a flat mapping of skill_name -> confidence for all skills.

        Useful for passing to SkillGapDetector.
        """
        all_skills = self._get_all_skills()
        return {s.name: s.confidence for s in all_skills}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, name: str) -> Optional[str]:
        """Look up skill ID by name (case-insensitive)."""
        key = name.lower()
        if key in self._name_to_id:
            return self._name_to_id[key]

        # Try searching semantic memory
        results = self.semantic_memory.search_skills(name, limit=5)
        for skill in results:
            if skill.name.lower() == key:
                self._name_to_id[key] = skill.skill_id
                return skill.skill_id
        return None

    def _get_all_skills(self) -> list:
        """Retrieve all skills from semantic memory."""
        try:
            inventory = self.semantic_memory.get_skill_inventory()
            # get_skill_inventory returns top_skills as Skill objects
            top = inventory.get("top_skills", [])
            # top_skills may be Skill objects or dicts depending on implementation
            from janus_reasoning_engine.memory.semantic_memory import Skill
            result = []
            for item in top:
                if isinstance(item, Skill):
                    result.append(item)
                elif isinstance(item, dict):
                    try:
                        result.append(Skill.from_dict(item))
                    except Exception:
                        pass
            return result
        except Exception as exc:
            logger.warning("Failed to retrieve all skills: %s", exc)
            return []

    @staticmethod
    def _confidence_to_level(confidence: float):
        """Map a confidence float to a SkillLevel enum value."""
        from janus_reasoning_engine.memory.semantic_memory import SkillLevel

        if confidence >= 0.9:
            return SkillLevel.EXPERT
        if confidence >= 0.75:
            return SkillLevel.ADVANCED
        if confidence >= 0.5:
            return SkillLevel.INTERMEDIATE
        if confidence >= 0.25:
            return SkillLevel.BEGINNER
        return SkillLevel.NOVICE
