"""
tests/test_market_analyzer.py
Property-based tests for MarketAnalyzer.

# Feature: janus-autonomous-worker-completion, Property 16
"""

import unittest.mock
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janus_worker_core import (
    JobRecord,
    MarketAnalysis,
    MarketAnalyzer,
    WorkerDatabase,
)


# ── Composite strategy for JobRecord ─────────────────────────────────────────

@st.composite
def job_record_strategy(draw) -> JobRecord:
    """Generate arbitrary JobRecord instances for property testing."""
    job_id = draw(st.text(min_size=1, max_size=64))
    title = draw(st.text(max_size=200))
    description = draw(st.text(max_size=500))
    platform = draw(st.sampled_from(["upwork", "fiverr", "freelancer", "toptal"]))
    budget = draw(st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False))
    status = draw(st.sampled_from(["available", "claimed", "completed", "failed"]))
    payment_amount = draw(
        st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        )
    )
    quality_score = draw(
        st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    metadata = draw(
        st.fixed_dictionaries({
            "required_skills": st.lists(st.text(min_size=1, max_size=30), max_size=5),
            "job_type": st.sampled_from(["code", "document", "design", "general"]),
        })
    )

    return JobRecord(
        id=job_id,
        title=title,
        description=description,
        platform=platform,
        budget=budget,
        status=status,
        claimed_at=None,
        completed_at=None,
        quality_score=quality_score,
        payment_amount=payment_amount,
        metadata=metadata,
    )


# ── Property 16: Market analysis always returns required keys ─────────────────
# Feature: janus-autonomous-worker-completion, Property 16
# Validates: Requirements 19.1

@given(st.lists(job_record_strategy()))
@settings(max_examples=100)
def test_market_analysis_required_keys(job_history: List[JobRecord]) -> None:
    """
    Property 16: For any job history list (including empty) passed to
    MarketAnalyzer.analyze(), the returned MarketAnalysis SHALL have non-None
    values for trending_skills, high_paying_job_types, emerging_opportunities,
    skill_roi, and recommendations.

    Validates: Requirements 19.1
    """
    brain = unittest.mock.MagicMock()
    brain.ask.return_value = "item one\nitem two\nitem three"

    db = WorkerDatabase(":memory:")
    analyzer = MarketAnalyzer(db=db, brain=brain)

    result = analyzer.analyze(job_history)

    assert isinstance(result, MarketAnalysis), "analyze() must return a MarketAnalysis instance"

    assert result.trending_skills is not None, "trending_skills must not be None"
    assert result.high_paying_job_types is not None, "high_paying_job_types must not be None"
    assert result.emerging_opportunities is not None, "emerging_opportunities must not be None"
    assert result.skill_roi is not None, "skill_roi must not be None"
    assert result.recommendations is not None, "recommendations must not be None"

    assert isinstance(result.trending_skills, list), "trending_skills must be a list"
    assert isinstance(result.high_paying_job_types, list), "high_paying_job_types must be a list"
    assert isinstance(result.emerging_opportunities, list), "emerging_opportunities must be a list"
    assert isinstance(result.skill_roi, dict), "skill_roi must be a dict"
    assert isinstance(result.recommendations, list), "recommendations must be a list"
