"""
tests/test_work_cycle.py
Property-based tests for WorkCycle.

# Feature: janus-autonomous-worker-completion, Property 10
"""

import asyncio
import math
from datetime import datetime
from decimal import Decimal
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from janus_worker_core import BrowserJob, WorkCycle


# ── Strategies ────────────────────────────────────────────────────────────────

@composite
def browser_job_strategy(draw):
    """Generate arbitrary BrowserJob instances."""
    job_id = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip()))
    title = draw(st.text(max_size=100))
    description = draw(st.text(max_size=200))
    platform = draw(st.sampled_from(["upwork", "fiverr"]))
    budget = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    required_skills = draw(st.lists(st.text(min_size=1, max_size=20), max_size=5))
    job_type = draw(st.sampled_from(["general", "code", "document"]))

    return BrowserJob(
        id=job_id,
        title=title,
        description=description,
        platform=platform,
        budget=budget,
        required_skills=required_skills,
        deadline=None,
        job_type=job_type,
    )


def _make_work_cycle(max_concurrent_jobs: int) -> WorkCycle:
    """Build a WorkCycle with all dependencies mocked."""
    db = MagicMock()
    db.list_skills.return_value = []
    db.insert_job.return_value = None
    db.update_job_status.return_value = None
    db.insert_cycle_summary.return_value = None

    wallet = MagicMock()
    wallet.record_income = MagicMock()

    decision_engine = MagicMock()
    # select_jobs returns all jobs up to max_concurrent_jobs (all qualify)
    decision_engine.select_jobs.side_effect = lambda jobs, skills, max_jobs: jobs[:max_jobs]

    work_generator = MagicMock()
    work_generator.generate = AsyncMock()

    learning_engine = MagicMock()
    learning_engine.learn_skill = AsyncMock(
        return_value=MagicMock(skill_name="general", resources_used=[])
    )

    quality_assurance = MagicMock()

    investment_engine = MagicMock()
    investment_engine.evaluate_and_invest = AsyncMock(return_value=[])

    market_analyzer = MagicMock()

    monitor = MagicMock()
    monitor.log_job_claimed = MagicMock()
    monitor.log_work_generated = MagicMock()
    monitor.log_job_completed = MagicMock()
    monitor.log_payment = MagicMock()
    monitor.log_skill_improved = MagicMock()
    monitor.log_error = MagicMock()

    return WorkCycle(
        db=db,
        wallet=wallet,
        decision_engine=decision_engine,
        work_generator=work_generator,
        learning_engine=learning_engine,
        quality_assurance=quality_assurance,
        investment_engine=investment_engine,
        market_analyzer=market_analyzer,
        monitor=monitor,
        max_concurrent_jobs=max_concurrent_jobs,
    )


# ── Property 10: Active job count never exceeds configured maximum ────────────
# Feature: janus-autonomous-worker-completion, Property 10
# Validates: Requirements 17.1

@given(
    st.integers(1, 10),
    st.lists(browser_job_strategy(), min_size=0, max_size=20),
)
@settings(max_examples=100)
def test_concurrent_job_limit(max_concurrent_jobs: int, jobs: List[BrowserJob]):
    """
    Property 10: Active job count never exceeds configured maximum.

    For any sequence of job claim and completion events processed by WorkCycle,
    the number of concurrently active jobs SHALL never exceed max_concurrent_jobs
    at any point in the sequence.

    **Validates: Requirements 17.1**
    """
    # Assign unique IDs to avoid collisions from Hypothesis-generated text
    unique_jobs = []
    for i, job in enumerate(jobs):
        unique_jobs.append(BrowserJob(
            id=f"job-{i}",
            title=job.title,
            description=job.description,
            platform=job.platform,
            budget=job.budget,
            required_skills=job.required_skills,
            deadline=job.deadline,
            job_type=job.job_type,
        ))

    wc = _make_work_cycle(max_concurrent_jobs)

    # Track peak concurrency by instrumenting _execute_job
    peak_concurrent = [0]
    current_concurrent = [0]

    original_execute_job = wc._execute_job

    async def tracking_execute_job(job):
        current_concurrent[0] += 1
        if current_concurrent[0] > peak_concurrent[0]:
            peak_concurrent[0] = current_concurrent[0]
        # Yield control so other coroutines can start (simulates real concurrency)
        await asyncio.sleep(0)
        result = await original_execute_job(job)
        current_concurrent[0] -= 1
        return result

    # Patch _execute_job on the instance
    wc._execute_job = tracking_execute_job

    # Patch _discover_jobs to return our test jobs
    async def mock_discover():
        return unique_jobs

    wc._discover_jobs = mock_discover

    # Configure work_generator and qa to produce passing results
    from janus_worker_core import WorkResult, QAResult

    async def mock_generate(job):
        return WorkResult(
            content="x" * 600,
            job_type=job.job_type,
            quality_score=0.9,
            generation_time_seconds=0.01,
            attempts=1,
        )

    wc._work_generator.generate = mock_generate

    def mock_validate(work_result, job):
        return QAResult(
            passed=True,
            score=0.9,
            completeness=0.9,
            relevance=0.9,
            format_score=0.9,
            feedback="ok",
        )

    wc._qa.validate = mock_validate

    # Run the cycle
    asyncio.run(wc.run_one_cycle())

    # The batch passed to asyncio.gather is capped at max_concurrent_jobs
    # so peak concurrency must never exceed the configured limit
    assert peak_concurrent[0] <= max_concurrent_jobs, (
        f"Peak concurrent jobs {peak_concurrent[0]} exceeded "
        f"max_concurrent_jobs={max_concurrent_jobs}"
    )


# ── Property 11: Job priority queue is ordered by deadline ascending ──────────
# Feature: janus-autonomous-worker-completion, Property 11
# Validates: Requirements 17.4

@composite
def browser_job_strategy_with_distinct_deadlines(draw):
    """Generate a list of BrowserJob instances with distinct, non-None deadlines."""
    jobs = draw(st.lists(browser_job_strategy(), min_size=2, max_size=20))
    # Assign distinct deadlines using unique integer offsets (seconds from epoch)
    offsets = draw(
        st.lists(
            st.integers(min_value=1, max_value=10_000_000),
            min_size=len(jobs),
            max_size=len(jobs),
            unique=True,
        )
    )
    result = []
    for i, job in enumerate(jobs):
        result.append(BrowserJob(
            id=f"job-deadline-{i}",
            title=job.title,
            description=job.description,
            platform=job.platform,
            budget=job.budget,
            required_skills=job.required_skills,
            deadline=datetime.utcfromtimestamp(offsets[i]),
            job_type=job.job_type,
        ))
    return result


@given(browser_job_strategy_with_distinct_deadlines())
@settings(max_examples=100)
def test_deadline_ordered_priority_queue(jobs):
    """
    Property 11: Job priority queue is ordered by deadline ascending.

    For any set of active jobs with distinct deadlines, the priority order
    produced by WorkCycle SHALL place the job with the earliest deadline first
    (i.e., jobs are sorted by deadline ascending).

    **Validates: Requirements 17.4**
    """
    wc = _make_work_cycle(max_concurrent_jobs=len(jobs))

    execution_order = []

    original_execute_job = wc._execute_job

    async def tracking_execute_job(job):
        execution_order.append(job.id)
        return await original_execute_job(job)

    wc._execute_job = tracking_execute_job

    async def mock_discover():
        return list(jobs)

    wc._discover_jobs = mock_discover

    from janus_worker_core import WorkResult, QAResult

    async def mock_generate(job):
        return WorkResult(
            content="x" * 600,
            job_type=job.job_type,
            quality_score=0.9,
            generation_time_seconds=0.01,
            attempts=1,
        )

    wc._work_generator.generate = mock_generate

    def mock_validate(work_result, job):
        return QAResult(
            passed=True,
            score=0.9,
            completeness=0.9,
            relevance=0.9,
            format_score=0.9,
            feedback="ok",
        )

    wc._qa.validate = mock_validate

    asyncio.run(wc.run_one_cycle())

    # Build the expected order: jobs sorted by deadline ascending
    expected_order = [
        j.id for j in sorted(jobs, key=lambda j: j.deadline)
    ]

    assert execution_order == expected_order, (
        f"Jobs were not executed in deadline order.\n"
        f"Expected: {expected_order}\n"
        f"Got:      {execution_order}"
    )


# ── Property 3: Low-quality work is never submitted ───────────────────────────
# Feature: janus-autonomous-worker-completion, Property 3
# Validates: Requirements 1.4, 14.3



@given(st.floats(0.0, 0.699))
@settings(max_examples=100)
def test_low_quality_work_never_submitted(quality_score: float):
    """
    Property 3: Low-quality work is never submitted.

    For any WorkResult where quality_score < 0.7, WorkCycle SHALL NOT call
    PlatformBrowser.deliver() or UpworkBrowser.submit_work() for that result.

    **Validates: Requirements 1.4, 14.3**
    """
    # Filter out NaN and infinity — only finite scores are meaningful
    assume(math.isfinite(quality_score))

    from janus_worker_core import WorkResult, QAResult

    wc = _make_work_cycle(max_concurrent_jobs=5)

    # Mock WorkGenerator.generate() to return a WorkResult with the low quality score
    async def mock_generate(job):
        return WorkResult(
            content="short",
            job_type=job.job_type,
            quality_score=quality_score,
            generation_time_seconds=0.01,
            attempts=1,
        )

    wc._work_generator.generate = mock_generate

    # QA validates against the actual score — score < 0.7 means passed=False
    def mock_validate(work_result, job):
        passed = work_result.quality_score >= 0.7
        return QAResult(
            passed=passed,
            score=work_result.quality_score,
            completeness=work_result.quality_score,
            relevance=work_result.quality_score,
            format_score=work_result.quality_score,
            feedback="QA failed: low quality" if not passed else "ok",
        )

    wc._qa.validate = mock_validate

    # Provide one job to process
    test_job = BrowserJob(
        id="low-quality-job",
        title="Test Job",
        description="A test job",
        platform="upwork",
        budget=50.0,
        required_skills=["python"],
        deadline=None,
        job_type="general",
    )

    async def mock_discover():
        return [test_job]

    wc._discover_jobs = mock_discover

    # Track whether deliver/submit_work are called via mocks on the browser classes
    mock_upwork_browser = MagicMock()
    mock_upwork_browser.submit_work = MagicMock(return_value=False)
    mock_upwork_browser.submit_work_async = AsyncMock(return_value=False)

    mock_fiverr_browser = MagicMock()
    mock_fiverr_browser.deliver_order = MagicMock(return_value=False)
    mock_fiverr_browser.deliver_order_async = AsyncMock(return_value=False)

    with patch("janus_worker_core.UpworkBrowser", return_value=mock_upwork_browser) as mock_upwork_cls, \
         patch("janus_worker_core.FiverrBrowser", return_value=mock_fiverr_browser) as mock_fiverr_cls, \
         patch("janus_worker_core._HAS_PLATFORM_BROWSER", True):

        asyncio.run(wc.run_one_cycle())

        # Assert neither browser's submit/deliver was called
        mock_upwork_browser.submit_work.assert_not_called()
        mock_fiverr_browser.deliver_order.assert_not_called()
