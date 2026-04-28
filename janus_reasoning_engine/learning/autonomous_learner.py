"""
Autonomous Learner for the Janus Reasoning Engine.

Orchestrates the process of learning a new skill:
  1. Search for tutorials via the computer-use engine (optional)
  2. Watch/read video content via janus_video_comprehension (optional)
  3. Extract knowledge from text using JanusGPT or heuristics
  4. Store learned knowledge in SemanticMemory

All external integrations are optional and fail gracefully.

**Validates: Requirements REQ-3.2, REQ-6.2, REQ-8.1**
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """Result of a learning session for a single skill."""

    skill_name: str
    success: bool
    knowledge_items: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    learning_duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def knowledge_count(self) -> int:
        return len(self.knowledge_items)


class AutonomousLearner:
    """
    Orchestrates autonomous skill learning.

    Dependencies are all optional:
    - janus_gpt: JanusGPT instance for knowledge extraction
    - computer_use_engine: for searching tutorials online
    - semantic_memory: SemanticMemory instance for storing knowledge
    """

    def __init__(
        self,
        semantic_memory=None,
        janus_gpt=None,
        computer_use_engine=None,
    ) -> None:
        self.semantic_memory = semantic_memory
        self.janus_gpt = janus_gpt
        self.computer_use_engine = computer_use_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_skill(self, skill_name: str, context: Dict[str, Any]) -> LearningResult:
        """
        Orchestrate learning a skill end-to-end.

        Steps:
          1. Search for tutorials (computer-use engine, graceful if unavailable)
          2. Fetch video/article content (janus_video_comprehension, graceful)
          3. Extract knowledge from gathered text
          4. Store knowledge in semantic memory

        Args:
            skill_name: Name of the skill to learn.
            context: Additional context (e.g. {"goal": "...", "domain": "..."}).

        Returns:
            LearningResult summarising what was learned.
        """
        start = datetime.utcnow()
        knowledge_items: List[str] = []
        sources_used: List[str] = []

        # Step 1 – search for tutorials
        tutorial_texts = self._search_tutorials(skill_name, context)
        sources_used.extend([f"tutorial:{i}" for i in range(len(tutorial_texts))])

        # Step 2 – video comprehension
        video_texts = self._fetch_video_content(skill_name, context)
        sources_used.extend([f"video:{i}" for i in range(len(video_texts))])

        all_texts = tutorial_texts + video_texts

        # Step 3 – extract knowledge
        for text in all_texts:
            items = self.extract_knowledge_from_text(text, skill_name)
            knowledge_items.extend(items)

        # Deduplicate while preserving order
        seen: set = set()
        unique_items: List[str] = []
        for item in knowledge_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        knowledge_items = unique_items

        # Step 4 – store in semantic memory
        if knowledge_items and self.semantic_memory is not None:
            self._store_knowledge(skill_name, knowledge_items, context)

        duration = (datetime.utcnow() - start).total_seconds()

        result = LearningResult(
            skill_name=skill_name,
            success=len(knowledge_items) > 0,
            knowledge_items=knowledge_items,
            sources_used=sources_used,
            learning_duration_seconds=duration,
        )
        logger.info(
            "Learned skill '%s': %d knowledge items in %.1fs",
            skill_name,
            result.knowledge_count,
            duration,
        )
        return result

    def extract_knowledge_from_text(self, text: str, skill: str) -> List[str]:
        """
        Extract a list of knowledge statements from raw text.

        Uses JanusGPT when available; falls back to a simple heuristic
        that splits on sentence boundaries and filters for relevance.

        Args:
            text: Raw text to extract knowledge from.
            skill: Skill name (used to focus extraction).

        Returns:
            List of knowledge strings.
        """
        if not text or not text.strip():
            return []

        if self.janus_gpt is not None:
            try:
                prompt = (
                    f"Extract the key facts and concepts about '{skill}' from the "
                    f"following text. List each fact on a new line starting with '- '.\n\n"
                    f"{text[:800]}"
                )
                response = self.janus_gpt.generate(prompt, max_new=200)
                items = self._parse_bullet_list(response)
                if items:
                    return items
            except Exception as exc:
                logger.debug("JanusGPT knowledge extraction failed: %s", exc)

        # Heuristic fallback: split into sentences, keep those mentioning the skill
        return self._heuristic_extract(text, skill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_tutorials(
        self, skill_name: str, context: Dict[str, Any]
    ) -> List[str]:
        """Search for tutorial content using the computer-use engine."""
        if self.computer_use_engine is None:
            logger.debug("Computer-use engine unavailable; skipping tutorial search.")
            return []
        try:
            query = f"how to learn {skill_name} tutorial"
            result = self.computer_use_engine.search(query)
            if isinstance(result, list):
                return [str(r) for r in result]
            if isinstance(result, str):
                return [result]
            return []
        except Exception as exc:
            logger.debug("Tutorial search failed: %s", exc)
            return []

    def _fetch_video_content(
        self, skill_name: str, context: Dict[str, Any]
    ) -> List[str]:
        """Fetch video transcripts via janus_video_comprehension."""
        try:
            import janus_video_comprehension as jvc  # type: ignore

            query = f"{skill_name} tutorial"
            result = jvc.get_transcript(query)
            if isinstance(result, str) and result.strip():
                return [result]
            if isinstance(result, list):
                return [str(r) for r in result if r]
        except ImportError:
            logger.debug("janus_video_comprehension not available.")
        except Exception as exc:
            logger.debug("Video comprehension failed: %s", exc)
        return []

    def _store_knowledge(
        self,
        skill_name: str,
        knowledge_items: List[str],
        context: Dict[str, Any],
    ) -> None:
        """Persist extracted knowledge into SemanticMemory."""
        try:
            from janus_reasoning_engine.memory.semantic_memory import KnowledgeType

            domain = context.get("domain", "general")
            for item in knowledge_items:
                self.semantic_memory.add_knowledge(
                    knowledge_type=KnowledgeType.FACT,
                    name=f"{skill_name}: {item[:60]}",
                    content={"fact": item, "skill": skill_name},
                    source="autonomous_learner",
                    tags=[skill_name.lower().replace(" ", "-")],
                    domains=[domain],
                )
        except Exception as exc:
            logger.warning("Failed to store knowledge in semantic memory: %s", exc)

    @staticmethod
    def _parse_bullet_list(text: str) -> List[str]:
        """Extract bullet-point items from LLM output."""
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if item:
                    items.append(item)
        return items

    @staticmethod
    def _heuristic_extract(text: str, skill: str) -> List[str]:
        """Simple sentence-based extraction heuristic."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        skill_lower = skill.lower()
        relevant = []
        for sentence in sentences:
            s = sentence.strip()
            if not s or len(s) < 20:
                continue
            if skill_lower in s.lower() or len(relevant) < 3:
                relevant.append(s)
            if len(relevant) >= 10:
                break
        return relevant
