"""
tests/test_browser_computer_use.py
===================================
Unit tests for BrowserComputerUse.

All engine calls are mocked — no real browser or display is required.

Requirements: 10.3
"""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure stubs are in place before importing the module under test
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stubs() -> None:
    stubs = {
        "pyautogui": {"size": lambda: (1920, 1080), "FAILSAFE": True},
        "pytesseract": {},
        "PIL": {},
        "PIL.Image": {},
        "PIL.ImageGrab": {},
        "pygetwindow": {},
        "cv2": {},
        "imagehash": {},
    }
    for pkg, attrs in stubs.items():
        if pkg not in sys.modules:
            mod = _make_stub(pkg)
            for attr, val in attrs.items():
                setattr(mod, attr, val)


_ensure_stubs()

# Patch _check_dependencies so the module loads without real packages
with patch("builtins.__import__", side_effect=lambda name, *a, **kw: sys.modules.get(name) or __import__(name, *a, **kw)):
    pass

import importlib
import unittest.mock as mock

# Patch _check_dependencies at module level before importing
with mock.patch.dict(sys.modules):
    # Import with _check_dependencies patched
    import janus_computer_use as _jcu_module
    # Patch the function so it doesn't raise
    _jcu_module._check_dependencies = lambda: None

from janus_computer_use import (
    ActionResult,
    ActionType,
    BrowserComputerUse,
    OCRWord,
    ScreenRegion,
    UIElement,
    WaitCondition,
    WaitConditionType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ui_element(label: str, cx: int = 100, cy: int = 100) -> UIElement:
    """Create a minimal UIElement for testing."""
    bb = ScreenRegion(x=cx - 10, y=cy - 5, width=20, height=10)
    return UIElement(
        element_type="input",
        label=label,
        bounding_box=bb,
        confidence=0.9,
        center=(cx, cy),
    )


def _make_ocr_word(text: str, x: int = 0, y: int = 0) -> OCRWord:
    bb = ScreenRegion(x=x, y=y, width=len(text) * 8, height=16)
    return OCRWord(text=text, bounding_box=bb, confidence=0.95)


def _make_engine(
    *,
    hotkey_ok: bool = True,
    type_ok: bool = True,
    press_ok: bool = True,
    click_ok: bool = True,
    ocr_words: list | None = None,
    find_element_results: dict | None = None,
) -> MagicMock:
    """Build a mock ComputerUseEngine with configurable sub-system behaviour."""
    engine = MagicMock()

    # keyboard
    engine.keyboard.hotkey = AsyncMock(
        return_value=ActionResult(success=hotkey_ok, action_type=ActionType.HOTKEY)
    )
    engine.keyboard.type_text = AsyncMock(
        return_value=ActionResult(
            success=type_ok,
            action_type=ActionType.TYPE,
            chars_delivered=10,
        )
    )
    engine.keyboard.press_key = AsyncMock(
        return_value=ActionResult(success=press_ok, action_type=ActionType.HOTKEY)
    )

    # mouse
    engine.mouse.click = AsyncMock(
        return_value=ActionResult(success=click_ok, action_type=ActionType.CLICK)
    )

    # screen
    fake_image = MagicMock()
    engine.screen.capture = AsyncMock(return_value=fake_image)
    engine.screen.ocr = AsyncMock(return_value=ocr_words or [])

    # vision
    _find_results = find_element_results or {}

    async def _find_element(label: str, screenshot):
        return _find_results.get(label, [])

    engine.vision.find_element = _find_element

    # wait_for
    engine.wait_for = AsyncMock(
        return_value=ActionResult(success=True, action_type=ActionType.WAIT)
    )

    return engine


# ---------------------------------------------------------------------------
# Tests for open()
# ---------------------------------------------------------------------------

class TestOpen:
    """Tests for BrowserComputerUse.open()."""

    def test_open_sets_expected_domain(self):
        """open() must store the netloc of the URL as _expected_domain."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("example.com")]
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/path"))
        assert bcu._expected_domain == "example.com"

    def test_open_sets_domain_with_subdomain(self):
        """open() extracts the full netloc including subdomains."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("www.jobs.example.com")]
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.open("https://www.jobs.example.com/search"))
        assert bcu._expected_domain == "www.jobs.example.com"

    def test_open_calls_ctrl_l_to_focus_address_bar(self):
        """open() must call keyboard.hotkey('ctrl', 'l') to focus the address bar."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("example.com")]
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        engine.keyboard.hotkey.assert_called_once_with("ctrl", "l")

    def test_open_types_url(self):
        """open() must type the full URL into the address bar."""
        url = "https://example.com/page"
        engine = _make_engine(
            ocr_words=[_make_ocr_word("example.com")]
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.open(url))
        engine.keyboard.type_text.assert_called_once_with(url)

    def test_open_presses_enter(self):
        """open() must press Enter after typing the URL."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("example.com")]
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        engine.keyboard.press_key.assert_called_with("enter")

    def test_open_returns_success_when_domain_matches(self):
        """open() returns a successful ActionResult when the domain is visible in OCR."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("example.com")]
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        assert result.success is True

    def test_open_returns_failure_on_hotkey_error(self):
        """open() propagates a failed ActionResult when hotkey fails."""
        engine = _make_engine(hotkey_ok=False)
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        assert result.success is False

    def test_open_returns_failure_on_type_error(self):
        """open() propagates a failed ActionResult when type_text fails."""
        engine = _make_engine(type_ok=False)
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        assert result.success is False


# ---------------------------------------------------------------------------
# Tests for domain mismatch detection
# ---------------------------------------------------------------------------

class TestDomainMismatch:
    """Tests for domain mismatch detection in BrowserComputerUse."""

    def test_domain_mismatch_returns_failed_action_result(self):
        """When OCR text does not contain the expected domain, open() returns failure."""
        # OCR returns text that does NOT contain the expected domain
        engine = _make_engine(
            ocr_words=[_make_ocr_word("phishing-site.com")]
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        assert result.success is False
        assert result.error_message is not None
        assert "Domain mismatch" in result.error_message

    def test_domain_mismatch_error_message_contains_expected_domain(self):
        """The error message must mention the expected domain."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("other-site.net")]
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://mysite.com/"))
        assert "mysite.com" in result.error_message

    def test_no_mismatch_when_domain_present_in_ocr(self):
        """No mismatch is reported when the expected domain appears in OCR text."""
        engine = _make_engine(
            ocr_words=[
                _make_ocr_word("mysite.com"),
                _make_ocr_word("Welcome"),
            ]
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://mysite.com/"))
        assert result.success is True

    def test_check_domain_skipped_when_no_expected_domain(self):
        """_check_domain() returns None when _expected_domain is not set."""
        engine = _make_engine()
        bcu = BrowserComputerUse(engine)
        # _expected_domain is None by default
        result = asyncio.get_event_loop().run_until_complete(bcu._check_domain())
        assert result is None

    def test_domain_check_is_case_insensitive(self):
        """Domain comparison is case-insensitive."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("EXAMPLE.COM")]
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.open("https://example.com/"))
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests for login()
# ---------------------------------------------------------------------------

class TestLogin:
    """Tests for BrowserComputerUse.login()."""

    def test_login_finds_username_field_and_types(self):
        """login() must click the username field and type the username."""
        username_el = _make_ui_element("username", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)
        submit_el = _make_ui_element("submit", cx=200, cy=400)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
                "password": [password_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(
            bcu.login("testuser", "testpass")
        )
        # Should have clicked the username field
        assert engine.mouse.click.call_count >= 1
        first_click_args = engine.mouse.click.call_args_list[0]
        assert first_click_args == mock.call(200, 300)

    def test_login_types_username(self):
        """login() must type the username string."""
        username_el = _make_ui_element("username", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)
        submit_el = _make_ui_element("submit", cx=200, cy=400)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
                "password": [password_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.login("myuser", "mypass"))

        typed_texts = [call.args[0] for call in engine.keyboard.type_text.call_args_list]
        assert "myuser" in typed_texts

    def test_login_types_password(self):
        """login() must type the password string."""
        username_el = _make_ui_element("username", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)
        submit_el = _make_ui_element("submit", cx=200, cy=400)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
                "password": [password_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.login("myuser", "s3cr3t"))

        typed_texts = [call.args[0] for call in engine.keyboard.type_text.call_args_list]
        assert "s3cr3t" in typed_texts

    def test_login_falls_back_to_email_label(self):
        """login() tries 'email' when 'username' is not found."""
        email_el = _make_ui_element("email", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)
        submit_el = _make_ui_element("submit", cx=200, cy=400)

        engine = _make_engine(
            find_element_results={
                "email": [email_el],
                "password": [password_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.login("user@example.com", "pass"))
        assert result.success is True

    def test_login_returns_failure_when_username_field_not_found(self):
        """login() returns a failed ActionResult when no username field is found."""
        password_el = _make_ui_element("password", cx=200, cy=350)

        engine = _make_engine(
            find_element_results={
                "password": [password_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.login("user", "pass"))
        assert result.success is False
        assert result.error_message is not None

    def test_login_returns_failure_when_password_field_not_found(self):
        """login() returns a failed ActionResult when no password field is found."""
        username_el = _make_ui_element("username", cx=200, cy=300)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(bcu.login("user", "pass"))
        assert result.success is False

    def test_login_presses_enter_when_no_submit_button(self):
        """login() falls back to pressing Enter when no submit button is found."""
        username_el = _make_ui_element("username", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
                "password": [password_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.login("user", "pass"))
        engine.keyboard.press_key.assert_called_with("enter")

    def test_login_clicks_submit_button_when_found(self):
        """login() clicks the submit button when it is found."""
        username_el = _make_ui_element("username", cx=200, cy=300)
        password_el = _make_ui_element("password", cx=200, cy=350)
        submit_el = _make_ui_element("submit", cx=200, cy=400)

        engine = _make_engine(
            find_element_results={
                "username": [username_el],
                "password": [password_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(bcu.login("user", "pass"))

        # Last click should be on the submit button
        last_click = engine.mouse.click.call_args_list[-1]
        assert last_click == mock.call(200, 400)


# ---------------------------------------------------------------------------
# Tests for _extract_domain()
# ---------------------------------------------------------------------------

class TestExtractDomain:
    """Tests for the static _extract_domain helper."""

    def test_extracts_simple_domain(self):
        assert BrowserComputerUse._extract_domain("https://example.com/") == "example.com"

    def test_extracts_subdomain(self):
        assert BrowserComputerUse._extract_domain("https://www.example.com/path") == "www.example.com"

    def test_extracts_domain_with_port(self):
        assert BrowserComputerUse._extract_domain("http://localhost:8080/app") == "localhost:8080"

    def test_empty_url_returns_empty_string(self):
        assert BrowserComputerUse._extract_domain("") == ""

    def test_extracts_domain_from_http_url(self):
        assert BrowserComputerUse._extract_domain("http://jobs.example.org/search?q=python") == "jobs.example.org"


# ---------------------------------------------------------------------------
# Tests for apply_to_job()
# ---------------------------------------------------------------------------

class TestApplyToJob:
    """Tests for BrowserComputerUse.apply_to_job()."""

    def test_apply_opens_url_first(self):
        """apply_to_job() must call open() with the job URL."""
        cover_el = _make_ui_element("cover letter", cx=300, cy=400)
        submit_el = _make_ui_element("submit", cx=300, cy=500)

        engine = _make_engine(
            ocr_words=[_make_ocr_word("jobs.example.com")],
            find_element_results={
                "cover letter": [cover_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(
            bcu.apply_to_job("https://jobs.example.com/job/123", "My cover letter")
        )
        # Ctrl+L should have been called (part of open())
        engine.keyboard.hotkey.assert_called_with("ctrl", "l")

    def test_apply_returns_failure_on_open_failure(self):
        """apply_to_job() returns failure when open() fails."""
        engine = _make_engine(hotkey_ok=False)
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(
            bcu.apply_to_job("https://jobs.example.com/job/123", "cover letter text")
        )
        assert result.success is False

    def test_apply_returns_failure_when_no_text_area(self):
        """apply_to_job() returns failure when no cover letter field is found."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("jobs.example.com")],
            find_element_results={},
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(
            bcu.apply_to_job("https://jobs.example.com/job/123", "cover letter text")
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# Tests for submit_work()
# ---------------------------------------------------------------------------

class TestSubmitWork:
    """Tests for BrowserComputerUse.submit_work()."""

    def test_submit_opens_url_first(self):
        """submit_work() must call open() with the submission URL."""
        submission_el = _make_ui_element("submission", cx=300, cy=400)
        submit_el = _make_ui_element("submit", cx=300, cy=500)

        engine = _make_engine(
            ocr_words=[_make_ocr_word("platform.example.com")],
            find_element_results={
                "submission": [submission_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(
            bcu.submit_work("https://platform.example.com/submit", "My work content")
        )
        engine.keyboard.hotkey.assert_called_with("ctrl", "l")

    def test_submit_types_content(self):
        """submit_work() must type the content into the submission field."""
        submission_el = _make_ui_element("submission", cx=300, cy=400)
        submit_el = _make_ui_element("submit", cx=300, cy=500)

        engine = _make_engine(
            ocr_words=[_make_ocr_word("platform.example.com")],
            find_element_results={
                "submission": [submission_el],
                "submit": [submit_el],
            }
        )
        bcu = BrowserComputerUse(engine)
        asyncio.get_event_loop().run_until_complete(
            bcu.submit_work("https://platform.example.com/submit", "My work content")
        )
        typed_texts = [call.args[0] for call in engine.keyboard.type_text.call_args_list]
        assert "My work content" in typed_texts

    def test_submit_returns_failure_on_open_failure(self):
        """submit_work() returns failure when open() fails."""
        engine = _make_engine(hotkey_ok=False)
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(
            bcu.submit_work("https://platform.example.com/submit", "content")
        )
        assert result.success is False

    def test_submit_returns_failure_when_no_text_area(self):
        """submit_work() returns failure when no submission field is found."""
        engine = _make_engine(
            ocr_words=[_make_ocr_word("platform.example.com")],
            find_element_results={},
        )
        bcu = BrowserComputerUse(engine)
        result = asyncio.get_event_loop().run_until_complete(
            bcu.submit_work("https://platform.example.com/submit", "content")
        )
        assert result.success is False
