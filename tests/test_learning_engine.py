"""
tests/test_learning_engine.py
==============================
Property-based tests for LearningEngine.

# Feature: janus-autonomous-worker-completion, Property 5
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from janus_worker_core import LearningEngine, SkillLevel, SkillRecord, WorkerDatabase


# ── Property 5: Concept extraction returns valid skill mappings ───────────────
# Feature: janus-autonomous-worker-completion, Property 5
# Validates: Requirements 2.4, 2.5


@given(
    content=st.text(min_size=1),
    skill=st.text(min_size=1),
)
@settings(max_examples=100)
def test_concept_extraction_returns_valid_skill_mappings(
    content: str, skill: str
) -> None:
    """
    Property 5: Concept extraction returns valid skill mappings.

    For any non-empty content and skill, _extract_concepts() SHALL return a
    non-empty list, and every key returned by _map_concepts_to_skills() SHALL
    exist in the known skills registry (WorkerDatabase.list_skills()).

    # Feature: janus-autonomous-worker-completion, Property 5
    Validates: Requirements 2.4, 2.5
    """
    # Set up in-memory DB with some pre-populated skills
    db = WorkerDatabase(":memory:")
    known_skill_names = ["python", "javascript", "machine learning", "data analysis"]
    for name in known_skill_names:
        db.upsert_skill(SkillRecord(
            name=name,
            level=SkillLevel.BEGINNER,
            experience_pts=0,
            success_rate=0.5,
            last_used=None,
            last_improved=None,
        ))

    # Mock brain.ask() to return a JSON list of concept strings
    brain = MagicMock()
    brain.ask = MagicMock(return_value='["concept1", "concept2"]')

    # Mock ComputerUseEngine with AsyncMock
    engine = AsyncMock()

    learning_engine = LearningEngine(engine=engine, brain=brain, db=db)

    # _extract_concepts must return a non-empty list
    concepts = asyncio.run(learning_engine._extract_concepts(content, skill))
    assert isinstance(concepts, list), "_extract_concepts must return a list"
    assert len(concepts) > 0, "_extract_concepts must return a non-empty list"

    # _map_concepts_to_skills: every returned key must exist in the known skills registry
    skill_map = learning_engine._map_concepts_to_skills(concepts)
    assert isinstance(skill_map, dict), "_map_concepts_to_skills must return a dict"

    known_names_set = {s.name for s in db.list_skills()}
    for key in skill_map:
        assert key in known_names_set, (
            f"Key {key!r} returned by _map_concepts_to_skills is not in the "
            f"known skills registry: {known_names_set}"
        )
