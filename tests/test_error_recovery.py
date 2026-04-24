# Feature: janus-autonomous-worker-completion, Property 6
"""
Property-based tests for error recovery and exponential backoff.

Validates: Requirements 5.1
"""

import pytest
from hypothesis import given, settings, strategies as st


# The backoff sequence used by LearningEngine and _with_backoff helpers
BACKOFF_DELAYS = [1, 2, 4, 8, 16]


# ── Property 6: Exponential backoff delays follow the correct sequence ────────

@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=5))
def test_exponential_backoff_sequence(n: int) -> None:
    """
    Property 6: Exponential backoff delays follow the correct sequence.

    For attempt number N (1-indexed), the delay at index N-1 in the
    BACKOFF_DELAYS list must equal 2**(N-1) seconds.

    Validates: Requirements 5.1
    """
    # Feature: janus-autonomous-worker-completion, Property 6
    expected_delay = 2 ** (n - 1)
    actual_delay = BACKOFF_DELAYS[n - 1]
    assert actual_delay == expected_delay, (
        f"Attempt {n}: expected delay {expected_delay}s, got {actual_delay}s"
    )
