"""
tests/test_decision_engine.py
Property-based tests for DecisionEngine.
"""

from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from janus_worker_core import BrowserJob, DecisionEngine, SkillLevel


# ── Strategies ────────────────────────────────────────────────────────────────

@composite
def browser_job_strategy(draw):
    """Generate arbitrary BrowserJob instances."""
    job_id = draw(st.text(min_size=1, max_size=50))
    title = draw(st.text(max_size=100))
    description = draw(st.text(max_size=500))
    platform = draw(st.text(max_size=50))
    budget = draw(st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False))
    required_skills = draw(st.lists(st.text(min_size=1, max_size=30), max_size=10))
    job_type = draw(st.sampled_from(["general", "code", "document", "design"]))

    # Deadline: None, past, or future
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


@composite
def skill_dict_strategy(draw):
    """Generate arbitrary Dict[str, SkillLevel] skill dictionaries."""
    keys = draw(st.lists(st.text(min_size=1, max_size=30), max_size=15, unique=True))
    values = draw(st.lists(st.sampled_from(SkillLevel), min_size=len(keys), max_size=len(keys)))
    return dict(zip(keys, values))


# ── Property 7: Job score is always in [0.0, 1.0] ────────────────────────────
# Feature: janus-autonomous-worker-completion, Property 7
# Validates: Requirements 6.1

@given(browser_job_strategy(), skill_dict_strategy())
@settings(max_examples=100)
def test_job_score_bounds(job, skills):
    """
    Property 7: For any BrowserJob and skill dictionary passed to
    DecisionEngine.score_job(), the returned score SHALL be a float in the
    closed interval [0.0, 1.0].

    Validates: Requirements 6.1
    """
    engine = DecisionEngine()
    score = engine.score_job(job, skills)

    assert isinstance(score, float), f"score_job must return a float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score {score} is outside [0.0, 1.0]"


# ── Property 8: Weighted job score equals the formula ────────────────────────
# Feature: janus-autonomous-worker-completion, Property 8
# Validates: Requirements 6.2

import math
from unittest.mock import patch


@given(
    st.floats(0, 1),
    st.floats(0, 1),
    st.floats(0, 1),
    st.floats(0, 1),
)
@settings(max_examples=100)
def test_weighted_score_formula(s_match, s_budget, s_deadline, s_learning):
    """
    Property 8: For any four component scores s_match, s_budget, s_deadline,
    s_learning each in [0.0, 1.0], the total score computed by
    DecisionEngine.score_job() SHALL equal
    0.40*s_match + 0.30*s_budget + 0.20*s_deadline + 0.10*s_learning
    within floating-point tolerance (1e-9).

    **Validates: Requirements 6.2**
    """
    from hypothesis import assume

    assume(all(math.isfinite(s) for s in [s_match, s_budget, s_deadline, s_learning]))

    engine = DecisionEngine()

    # Minimal job — content doesn't matter since all sub-score methods are mocked
    job = BrowserJob(
        id="test",
        title="Test Job",
        description="desc",
        platform="upwork",
        budget=50.0,
        required_skills=[],
    )

    with patch.object(engine, "_skill_match_score", return_value=s_match), \
         patch.object(engine, "_budget_score", return_value=s_budget), \
         patch.object(engine, "_deadline_score", return_value=s_deadline), \
         patch.object(engine, "_learning_score", return_value=s_learning):

        score = engine.score_job(job, {})

    expected = 0.40 * s_match + 0.30 * s_budget + 0.20 * s_deadline + 0.10 * s_learning
    # score_job clamps to [0,1]; since all sub-scores are in [0,1] and weights sum to 1,
    # expected is already in [0,1], so clamping has no effect here.
    assert abs(score - expected) < 1e-9, (
        f"score {score} != expected {expected} "
        f"(s_match={s_match}, s_budget={s_budget}, s_deadline={s_deadline}, s_learning={s_learning})"
    )


# ── Property 9: Selected jobs are always the top-N by score ──────────────────
# Feature: janus-autonomous-worker-completion, Property 9
# Validates: Requirements 6.3, 6.4


@composite
def unique_id_job_list_strategy(draw):
    """Generate a list of BrowserJob instances with guaranteed unique IDs."""
    jobs_raw = draw(st.lists(browser_job_strategy(), min_size=0, max_size=20))
    # Assign unique sequential IDs so score lookups by ID are unambiguous
    unique_jobs = []
    for i, job in enumerate(jobs_raw):
        unique_jobs.append(BrowserJob(
            id=f"job-{i}",
            title=job.title,
            description=job.description,
            platform=job.platform,
            budget=job.budget,
            required_skills=job.required_skills,
            deadline=job.deadline,
            job_type=job.job_type,
            metadata=job.metadata,
        ))
    return unique_jobs


@given(
    unique_id_job_list_strategy(),
    st.integers(1, 10),
)
@settings(max_examples=100)
def test_select_jobs_top_n(jobs, n):
    """
    Property 9: For any list of scored jobs and a capacity limit N, the jobs
    returned by DecisionEngine.select_jobs() SHALL be exactly the N jobs with
    the highest scores (or all qualifying jobs if fewer than N qualify), and no
    selected job SHALL have a score below 0.5.

    **Validates: Requirements 6.3, 6.4**
    """
    engine = DecisionEngine()
    skills: Dict = {}

    selected = engine.select_jobs(jobs, skills, max_jobs=n)

    # Score every job independently so we can verify the selection
    all_scores = {job.id: engine.score_job(job, skills) for job in jobs}

    # No selected job may have a score below 0.5
    for job in selected:
        score = all_scores[job.id]
        assert score >= 0.5, (
            f"Selected job '{job.id}' has score {score} which is below 0.5"
        )

    # Qualifying jobs are those with score >= 0.5, sorted by score descending
    qualifying = sorted(
        [job for job in jobs if all_scores[job.id] >= 0.5],
        key=lambda j: all_scores[j.id],
        reverse=True,
    )

    # The number of selected jobs must be min(n, len(qualifying))
    expected_count = min(n, len(qualifying))
    assert len(selected) == expected_count, (
        f"Expected {expected_count} selected jobs, got {len(selected)}"
    )

    # Selected jobs must be the top-N qualifying jobs by score
    expected_ids = [job.id for job in qualifying[:n]]
    selected_ids = [job.id for job in selected]
    assert selected_ids == expected_ids, (
        f"Selected jobs {selected_ids} do not match expected top-N {expected_ids}"
    )
