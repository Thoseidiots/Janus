"""
tests/test_work_generator.py
============================
Property-based tests for WorkGenerator.

# Feature: janus-autonomous-worker-completion, Property 1
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janus_worker_core import BrowserJob, WorkGenerator

# ── strategies ────────────────────────────────────────────────────────────────

non_empty_text = st.text(min_size=1)


# ── Property 1: Work prompt contains all job context fields ───────────────────
# Validates: Requirements 1.2


@given(
    title=non_empty_text,
    description=non_empty_text,
    skills=st.lists(non_empty_text, min_size=1),
)
@settings(max_examples=100)
def test_build_prompt_contains_all_job_context_fields(
    title: str, description: str, skills: list
) -> None:
    """
    Property 1: Work prompt contains all job context fields.

    For any BrowserJob with non-empty title, description, and required_skills,
    _build_prompt() SHALL contain the job title, description, and each required
    skill as substrings.

    # Feature: janus-autonomous-worker-completion, Property 1
    Validates: Requirements 1.2
    """
    brain = MagicMock()
    generator = WorkGenerator(brain)

    job = BrowserJob(
        id="test-id",
        title=title,
        description=description,
        platform="upwork",
        budget=100.0,
        required_skills=skills,
    )

    prompt = generator._build_prompt(job)

    assert title in prompt, f"Title {title!r} not found in prompt"
    assert description in prompt, f"Description {description!r} not found in prompt"
    for skill in skills:
        assert skill in prompt, f"Skill {skill!r} not found in prompt"


# ── Strategy ──────────────────────────────────────────────────────────────────

from datetime import datetime

from hypothesis.strategies import composite


@composite
def browser_job_strategy(draw):
    """Generate arbitrary BrowserJob instances for WorkGenerator tests."""
    job_id = draw(st.text(min_size=1, max_size=50))
    title = draw(st.text(max_size=100))
    description = draw(st.text(max_size=500))
    platform = draw(st.text(max_size=50))
    budget = draw(st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False))
    required_skills = draw(st.lists(st.text(min_size=1, max_size=30), max_size=10))
    job_type = draw(st.sampled_from(["general", "code", "document", "design"]))
    deadline_choice = draw(st.one_of(
        st.none(),
        st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2100, 1, 1),
        ),
    ))
    return BrowserJob(
        id=job_id,
        title=title,
        description=description,
        platform=platform,
        budget=budget,
        required_skills=required_skills,
        deadline=deadline_choice,
        job_type=job_type,
    )


# ── Property 4: Work metrics always contain required fields ───────────────────
# Feature: janus-autonomous-worker-completion, Property 4
# Validates: Requirements 1.5, 8.3

import asyncio

from janus_worker_core import WorkResult


@given(browser_job_strategy())
@settings(max_examples=100)
def test_work_result_required_fields(job) -> None:
    """
    Property 4: Work metrics always contain required fields.

    For any completed call to WorkGenerator.generate(), the returned WorkResult
    SHALL have non-None values for quality_score, generation_time_seconds, and
    attempts, with quality_score in [0.0, 1.0] and generation_time_seconds >= 0.

    # Feature: janus-autonomous-worker-completion, Property 4
    Validates: Requirements 1.5, 8.3
    """
    brain = MagicMock()
    # Return a non-empty string so generation succeeds and quality > 0
    brain.ask.return_value = (
        "This is a detailed response that addresses the job requirements fully. "
        "It contains enough content to pass the length check and keyword overlap."
    )

    generator = WorkGenerator(brain)
    result: WorkResult = asyncio.run(generator.generate(job))

    # Required fields must be non-None
    assert result.quality_score is not None, "quality_score must not be None"
    assert result.generation_time_seconds is not None, "generation_time_seconds must not be None"
    assert result.attempts is not None, "attempts must not be None"

    # quality_score in [0.0, 1.0]
    assert 0.0 <= result.quality_score <= 1.0, (
        f"quality_score {result.quality_score} is outside [0.0, 1.0]"
    )

    # generation_time_seconds >= 0
    assert result.generation_time_seconds >= 0, (
        f"generation_time_seconds {result.generation_time_seconds} is negative"
    )


# ── Property 2: Quality validator score is always in [0.0, 1.0] ───────────────
# Feature: janus-autonomous-worker-completion, Property 2
# Validates: Requirements 1.3, 14.1

from janus_worker_core import QualityAssurance, QAResult


@given(st.text(), browser_job_strategy())
@settings(max_examples=100)
def test_qa_score_bounds(work_content: str, job) -> None:
    """
    Property 2: Quality validator score is always in [0.0, 1.0].

    For any work string and job object passed to QualityAssurance.validate(),
    the returned QAResult.score SHALL be a float in the closed interval [0.0, 1.0],
    and each sub-score (completeness, relevance, format_score) SHALL also be in [0.0, 1.0].

    # Feature: janus-autonomous-worker-completion, Property 2
    Validates: Requirements 1.3, 14.1
    """
    qa = QualityAssurance()
    work = WorkResult(
        content=work_content,
        job_type=job.job_type,
        quality_score=0.0,
        generation_time_seconds=0.0,
        attempts=1,
    )

    result: QAResult = qa.validate(work, job)

    assert 0.0 <= result.score <= 1.0, (
        f"QAResult.score {result.score} is outside [0.0, 1.0]"
    )
    assert 0.0 <= result.completeness <= 1.0, (
        f"QAResult.completeness {result.completeness} is outside [0.0, 1.0]"
    )
    assert 0.0 <= result.relevance <= 1.0, (
        f"QAResult.relevance {result.relevance} is outside [0.0, 1.0]"
    )
    assert 0.0 <= result.format_score <= 1.0, (
        f"QAResult.format_score {result.format_score} is outside [0.0, 1.0]"
    )
