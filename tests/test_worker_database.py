"""
tests/test_worker_database.py
Property-based tests for WorkerDatabase persistence.
"""

import math
import uuid

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from janus_worker_core import WorkerDatabase, JobRecord, SkillLevel, SkillRecord


# ── Property 13: Job persistence round-trip preserves all fields ──────────────
# Feature: janus-autonomous-worker-completion, Property 13
# Validates: Requirements 9.3

@given(
    title=st.text(),
    description=st.text(),
    platform=st.text(),
    budget=st.floats(min_value=0),
)
@settings(max_examples=100)
def test_job_persistence_round_trip(title, description, platform, budget):
    """
    Property 13: For any JobRecord written to WorkerDatabase, reading it back
    by id SHALL return a record where id, title, description, platform, budget,
    and status are identical to the original.

    Validates: Requirements 9.3
    """
    assume(math.isfinite(budget))

    db = WorkerDatabase(":memory:")

    job_id = str(uuid.uuid4())
    original = JobRecord(
        id=job_id,
        title=title,
        description=description,
        platform=platform,
        budget=budget,
        status="available",
        claimed_at=None,
        completed_at=None,
        quality_score=None,
        payment_amount=None,
        metadata={},
    )

    db.insert_job(original)
    retrieved = db.get_job(job_id)

    assert retrieved is not None, "get_job returned None after insert"
    assert retrieved.id == original.id
    assert retrieved.title == original.title
    assert retrieved.description == original.description
    assert retrieved.platform == original.platform
    assert retrieved.budget == original.budget
    assert retrieved.status == original.status


# ── Property 14: Skill persistence round-trip preserves all fields ────────────
# Feature: janus-autonomous-worker-completion, Property 14
# Validates: Requirements 9.4

@given(
    name=st.text(min_size=1),
    level=st.sampled_from(SkillLevel),
    experience_pts=st.integers(min_value=0, max_value=2**63 - 1),
    success_rate=st.floats(0.0, 1.0),
)
@settings(max_examples=100)
def test_skill_persistence_round_trip(name, level, experience_pts, success_rate):
    """
    Property 14: For any Skill object written to WorkerDatabase, reading it back
    by name SHALL return a record where name, level, experience_pts, and
    success_rate are identical to the original.

    Validates: Requirements 9.4
    """
    assume(math.isfinite(success_rate))

    db = WorkerDatabase(":memory:")

    original = SkillRecord(
        name=name,
        level=level,
        experience_pts=experience_pts,
        success_rate=success_rate,
        last_used=None,
        last_improved=None,
    )

    db.upsert_skill(original)
    retrieved = db.get_skill(name)

    assert retrieved is not None, "get_skill returned None after upsert"
    assert retrieved.name == original.name
    assert retrieved.level == original.level
    assert retrieved.experience_pts == original.experience_pts
    assert retrieved.success_rate == original.success_rate
