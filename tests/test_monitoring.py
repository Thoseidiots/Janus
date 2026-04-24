"""
tests/test_monitoring.py
Property-based tests for MonitoringSystem.

# Feature: janus-autonomous-worker-completion, Property 15
"""

import io
import tempfile
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from janus_worker_core import (
    JobRecord,
    MonitoringSystem,
    PerformanceMetrics,
)


# ── Composite strategy for JobRecord ─────────────────────────────────────────

@st.composite
def job_record_strategy(draw) -> JobRecord:
    """Generate arbitrary JobRecord instances for property testing."""
    job_id = draw(st.text(min_size=1, max_size=64))
    title = draw(st.text(max_size=200))
    description = draw(st.text(max_size=500))
    platform = draw(st.sampled_from(["upwork", "fiverr", "freelancer", "toptal"]))
    budget = draw(
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)
    )
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
        metadata={},
    )


# ── Composite strategy for transaction dicts ──────────────────────────────────

@st.composite
def transaction_strategy(draw) -> Dict[str, Any]:
    """Generate arbitrary transaction dicts with amount, tx_type, description."""
    amount = draw(
        st.decimals(
            min_value=Decimal("0.01"),
            max_value=Decimal("100000.00"),
            allow_nan=False,
            allow_infinity=False,
            places=2,
        )
    )
    tx_type = draw(st.sampled_from(["income", "expense"]))
    description = draw(st.text(max_size=200))
    return {
        "amount": amount,
        "tx_type": tx_type,
        "description": description,
    }


# ── Property 15: Performance metrics contain all required fields ──────────────
# Feature: janus-autonomous-worker-completion, Property 15
# Validates: Requirements 8.8

@given(st.lists(job_record_strategy()), st.lists(transaction_strategy()))
@settings(max_examples=100)
def test_performance_metrics_required_fields(
    job_history: List[JobRecord],
    transactions: List[Dict[str, Any]],
) -> None:
    """
    Property 15: For any job and payment history passed to
    MonitoringSystem.get_metrics(), the returned PerformanceMetrics SHALL have
    non-None values for jobs_completed, total_earned, average_job_value,
    skill_levels, and error_rate.

    Validates: Requirements 8.8
    """
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        log_path = tmp.name

    monitor = MonitoringSystem(log_path=log_path)

    # Log events derived from the generated job and transaction history so
    # the MonitoringSystem's in-memory counters reflect the inputs.
    for job in job_history:
        if job.status == "completed":
            quality = job.quality_score if job.quality_score is not None else 0.8
            monitor.log_job_completed(job.id, quality)

    for tx in transactions:
        if tx["tx_type"] == "income":
            monitor.log_payment(tx["amount"], "test_platform")

    metrics = monitor.get_metrics()

    assert isinstance(metrics, PerformanceMetrics), (
        "get_metrics() must return a PerformanceMetrics instance"
    )

    assert metrics.jobs_completed is not None, "jobs_completed must not be None"
    assert metrics.total_earned is not None, "total_earned must not be None"
    assert metrics.average_job_value is not None, "average_job_value must not be None"
    assert metrics.skill_levels is not None, "skill_levels must not be None"
    assert metrics.error_rate is not None, "error_rate must not be None"

    assert isinstance(metrics.jobs_completed, int), "jobs_completed must be an int"
    assert isinstance(metrics.total_earned, Decimal), "total_earned must be a Decimal"
    assert isinstance(metrics.average_job_value, Decimal), "average_job_value must be a Decimal"
    assert isinstance(metrics.skill_levels, dict), "skill_levels must be a dict"
    assert isinstance(metrics.error_rate, float), "error_rate must be a float"
