"""
test_action_planner.py — property tests for ActionPlanner.

Properties covered:
  - Property 15: Planner step history length equals steps taken
  - Property 16: Planner stops at max_steps
  - Property 17: Candidate actions contain all required fields

All OS calls are mocked so the suite runs without a physical display.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure workspace root is on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing the module under test.
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUBS = ["pyautogui", "pytesseract", "pygetwindow", "cv2", "imagehash"]
for _pkg in _STUBS:
    if _pkg not in sys.modules:
        _stub(_pkg)

# Remove any PIL stubs so the real Pillow is used
for _pil_pkg in ["PIL", "PIL.Image", "PIL.ImageGrab"]:
    if _pil_pkg in sys.modules and not hasattr(sys.modules[_pil_pkg], "__file__"):
        del sys.modules[_pil_pkg]

_pag = sys.modules["pyautogui"]
_pag.size = lambda: (1920, 1080)  # type: ignore[attr-defined]
_pag.FAILSAFE = True  # type: ignore[attr-defined]

import janus_computer_use as _jcu  # noqa: E402
_jcu._check_dependencies = lambda: None  # type: ignore[attr-defined]

from janus_computer_use import (  # noqa: E402
    ActionPlanner,
    ActionResult,
    ActionType,
    CandidateAction,
)

# ---------------------------------------------------------------------------
# Hypothesis imports
# ---------------------------------------------------------------------------
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_engine_and_planner():
    """
    Build a fully-mocked ComputerUseEngine + ActionPlanner pair.

    Mocking strategy
    ----------------
    - engine.screen.capture  → returns a real (tiny) PIL Image
    - engine.screen.ocr      → returns [] (no success keywords → goal never achieved)
    - engine.vision.find_element → returns []
    - engine.execute_action  → returns ActionResult(success=True, action_type=SCREENSHOT)
    - engine._logger._make_thumbnail → returns a base64 string
    - brain.ask              → returns a valid JSON array with one screenshot action
    """
    from PIL import Image

    dummy_image = Image.new("RGB", (10, 10), color=(0, 0, 0))
    dummy_thumbnail = "dGVzdA=="  # base64("test")

    # Valid brain response: one screenshot action, confidence 0.8
    brain_response = json.dumps([
        {
            "action_type": "screenshot",
            "params": {},
            "confidence": 0.8,
            "rationale": "Take a screenshot to observe the screen.",
        }
    ])

    # --- mock brain ---
    mock_brain = MagicMock()
    mock_brain.ask.return_value = brain_response

    # --- mock engine ---
    engine = MagicMock()
    engine._context = {}

    # screen.capture returns a PIL Image
    engine.screen.capture = AsyncMock(return_value=dummy_image)
    # screen.ocr returns empty list → no success keywords → goal never achieved
    engine.screen.ocr = AsyncMock(return_value=[])
    # vision.find_element returns empty list
    engine.vision.find_element = AsyncMock(return_value=[])
    # execute_action returns a successful ActionResult
    engine.execute_action = AsyncMock(
        return_value=ActionResult(
            success=True,
            action_type=ActionType.SCREENSHOT,
        )
    )
    # _logger._make_thumbnail returns a base64 string
    engine._logger._make_thumbnail = MagicMock(return_value=dummy_thumbnail)

    planner = ActionPlanner(engine=engine, brain=mock_brain)
    return engine, planner


# ===========================================================================
# Property 15: Planner step history length equals steps taken
# ===========================================================================

# Feature: janus-computer-use, Property 15: Planner step history length equals steps taken
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.integers(min_value=1, max_value=50))
def test_planner_step_history_length_equals_steps_taken(n):
    """
    **Validates: Requirements 8.7**

    After running the planner for N steps (with a goal that is never achieved),
    len(planner._history) must equal N.
    """
    _engine, planner = _make_engine_and_planner()

    goal = "do something that never completes"
    result = _run(planner.run(goal, max_steps=n))

    assert len(planner._history) == n, (
        f"Expected history length {n}, got {len(planner._history)}"
    )


# ===========================================================================
# Property 16: Planner stops at max_steps
# ===========================================================================

# Feature: janus-computer-use, Property 16: Planner stops at max_steps
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.integers(min_value=1, max_value=20))
def test_planner_stops_at_max_steps(m):
    """
    **Validates: Requirements 8.5**

    When the goal is never achieved, the planner must execute exactly M steps
    and return ActionResult(success=False).
    """
    _engine, planner = _make_engine_and_planner()

    goal = "do something that never completes"
    result = _run(planner.run(goal, max_steps=m))

    assert result.success is False, (
        f"Expected success=False when max_steps={m} is reached, got success={result.success}"
    )
    assert len(planner._history) == m, (
        f"Expected exactly {m} steps in history, got {len(planner._history)}"
    )


# ===========================================================================
# Property 17: Candidate actions contain all required fields
# ===========================================================================

# Hypothesis strategy: generate a list of JSON objects that look like planner
# responses.  Each item may have arbitrary string values for action_type and
# rationale, a float confidence, and a dict params.
_action_type_values = [at.value for at in ActionType]

_candidate_item_strategy = st.fixed_dictionaries({
    "action_type": st.sampled_from(_action_type_values),
    "params": st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
        max_size=5,
    ),
    "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "rationale": st.text(min_size=1, max_size=100),
})

# Feature: janus-computer-use, Property 17: Candidate actions contain all required fields
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.lists(_candidate_item_strategy, min_size=1, max_size=5))
def test_candidate_actions_contain_all_required_fields(items):
    """
    **Validates: Requirements 8.2**

    Every CandidateAction returned by _parse_brain_response() must have:
      - action.action_type  (not None)
      - action.params       (a dict)
      - confidence          in [0.0, 1.0]
      - rationale           (not None)
    """
    _engine, planner = _make_engine_and_planner()

    json_str = json.dumps(items)
    candidates = planner._parse_brain_response(json_str)

    # The parser must return at least one candidate for valid input
    assert len(candidates) == len(items), (
        f"Expected {len(items)} candidates, got {len(candidates)}"
    )

    for i, candidate in enumerate(candidates):
        assert candidate.action.action_type is not None, (
            f"Candidate {i}: action.action_type must not be None"
        )
        assert isinstance(candidate.action.params, dict), (
            f"Candidate {i}: action.params must be a dict, got {type(candidate.action.params)}"
        )
        assert 0.0 <= candidate.confidence <= 1.0, (
            f"Candidate {i}: confidence {candidate.confidence} is not in [0.0, 1.0]"
        )
        assert candidate.rationale is not None, (
            f"Candidate {i}: rationale must not be None"
        )
