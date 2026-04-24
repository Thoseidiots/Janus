"""
janus_computer_use.py
=====================
Janus Computer Use — capability layer that gives the Janus autonomous worker
full human-like desktop control on Windows.

Architecture overview
---------------------
The module exposes a single top-level async context manager, ``ComputerUseEngine``,
that coordinates eight sub-systems:

    MouseController     — cursor movement, clicks, scroll, drag
    KeyboardController  — text typing, key combinations, special keys
    ScreenReader        — screenshot capture and OCR (pytesseract + Pillow)
    VisualDetector      — template matching and UI-element detection (OpenCV)
    WindowManager       — list, focus, resize, move, minimise, maximise windows
    ActionPlanner       — perceive → reason → act loop backed by AvusBrain
    BrowserComputerUse  — high-level browser helper (login, search, apply, submit)
    ActionLogger        — persistent structured log with base64 thumbnails

All blocking OS calls are offloaded via ``asyncio.to_thread`` so the module is
fully async-compatible and integrates with the existing ``janus_autonomous_worker.py``
and ``janus_automation_platform.py`` without modifying their core logic.

Requirements: 10.1, 10.2
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import asyncio
import base64
import datetime
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Tesseract path — set before pytesseract is imported
# ---------------------------------------------------------------------------
import os as _os
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _tp in _TESSERACT_PATHS:
    if _os.path.exists(_tp):
        try:
            import pytesseract as _pt
            _pt.pytesseract.tesseract_cmd = _tp
        except ImportError:
            pass
        break

# ---------------------------------------------------------------------------
# Dependency check — must run before any third-party import
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    # (import_name, pip_name)
    ("pyautogui",    "pyautogui"),
    ("pytesseract",  "pytesseract"),
    ("PIL",          "Pillow"),
    ("pygetwindow",  "pygetwindow"),
    ("cv2",          "opencv-python"),
    ("imagehash",    "imagehash"),
]


def _check_dependencies() -> None:
    """Import each required package and raise a descriptive ``ImportError``
    listing *all* missing packages if any are absent.

    This function is called at module import time (see bottom of file) so that
    missing dependencies are surfaced immediately rather than at first use.

    Validates: Requirements 10.2
    """
    missing_import_names: List[str] = []
    missing_pip_names: List[str] = []

    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing_import_names.append(import_name)
            missing_pip_names.append(pip_name)

    if missing_import_names:
        raise ImportError(
            f"janus_computer_use requires the following packages which are not "
            f"installed: {', '.join(missing_import_names)}.\n"
            f"Install them with:\n"
            f"    pip install {' '.join(missing_pip_names)}\n"
            f"Also ensure Tesseract OCR is installed:\n"
            f"    winget install UB-Mannheim.TesseractOCR"
        )


# ---------------------------------------------------------------------------
# Third-party imports (only reached when all dependencies are present)
# ---------------------------------------------------------------------------
# These are imported lazily inside each class/function so that the module can
# still be partially imported for testing with mocked dependencies.  The
# _check_dependencies() call at the bottom of this file ensures the real
# packages are present in production.

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("janus")

# ---------------------------------------------------------------------------
# Data models — enums
# ---------------------------------------------------------------------------

class ActionType(Enum):
    MOVE         = "move"
    CLICK        = "click"
    RIGHT_CLICK  = "right_click"
    DOUBLE_CLICK = "double_click"
    TYPE         = "type"
    HOTKEY       = "hotkey"
    SCROLL       = "scroll"
    DRAG         = "drag"
    SCREENSHOT   = "screenshot"
    OCR          = "ocr"
    FIND_ELEMENT = "find_element"
    WAIT         = "wait"
    FOCUS_WINDOW = "focus_window"


class ScrollDirection(Enum):
    UP    = "up"
    DOWN  = "down"
    LEFT  = "left"
    RIGHT = "right"


class WaitConditionType(Enum):
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_GONE    = "element_gone"
    TEXT_PRESENT    = "text_present"
    TEXT_GONE       = "text_gone"
    IMAGE_PRESENT   = "image_present"


# ---------------------------------------------------------------------------
# Data models — dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class UIElement:
    element_type: str          # "button", "input", "checkbox", etc.
    label: str
    bounding_box: ScreenRegion
    confidence: float          # 0.0 – 1.0
    center: Tuple[int, int]    # (cx, cy) — recommended click target


@dataclass
class OCRWord:
    text: str
    bounding_box: ScreenRegion
    confidence: float          # 0.0 – 1.0


@dataclass
class WaitCondition:
    condition_type: WaitConditionType
    target: str                # text string, element label, or template path
    timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5


@dataclass
class Action:
    action_type: ActionType
    params: Dict[str, Any]     # type-specific parameters
    pre_approved: bool = False  # skip destructive-action safety check


@dataclass
class ActionResult:
    success: bool
    action_type: ActionType
    data: Optional[Any] = None
    error_message: Optional[str] = None
    chars_delivered: Optional[int] = None  # for TYPE actions
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class CandidateAction:
    action: Action
    confidence: float
    rationale: str


@dataclass
class StepRecord:
    step_number: int
    action: Action
    result: ActionResult
    screenshot_before: str     # base64-encoded thumbnail
    screenshot_after: str      # base64-encoded thumbnail


@dataclass
class WindowInfo:
    handle: int
    title: str
    process_name: str
    bounding_box: ScreenRegion


@dataclass
class ActionLogEntry:
    action_type: str
    target: str
    timestamp: str             # ISO 8601
    outcome: str               # "success" | "failure"
    error_message: Optional[str]
    screenshot_thumbnail: str  # base64-encoded JPEG thumbnail

# ---------------------------------------------------------------------------
# MouseController
# ---------------------------------------------------------------------------

class MouseController:
    """Controls mouse movement, clicks, scrolling, and drag operations.

    All blocking pyautogui calls are offloaded via ``asyncio.to_thread`` so
    this class is fully async-compatible.

    Requirements: 1.1–1.7, 6.1–6.6
    """

    def __init__(self) -> None:
        import pyautogui
        w, h = pyautogui.size()
        self._screen_width: int = w
        self._screen_height: int = h

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_coords(self, x: int, y: int) -> bool:
        """Return ``True`` if (x, y) is within display bounds, else log and return ``False``."""
        if x < 0 or y < 0 or x >= self._screen_width or y >= self._screen_height:
            logger.error(
                "MouseController: coordinate (%d, %d) is out of bounds "
                "(screen %dx%d).",
                x, y, self._screen_width, self._screen_height,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def move(self, x: int, y: int, human_like: bool = False) -> ActionResult:
        """Move the cursor to (x, y).

        When *human_like* is ``True`` the cursor follows a smooth easeInOutQuad
        path over 0.5 s instead of teleporting instantly.

        Requirements: 1.1, 1.7
        """
        if not self._validate_coords(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                error_message=f"Coordinate ({x}, {y}) is out of bounds.",
            )
        try:
            import pyautogui
            if human_like:
                await asyncio.to_thread(
                    pyautogui.moveTo,
                    x, y,
                    duration=0.5,
                    tween=pyautogui.easeInOutQuad,
                )
            else:
                await asyncio.to_thread(pyautogui.moveTo, x, y)
            return ActionResult(success=True, action_type=ActionType.MOVE)
        except Exception as exc:  # includes pyautogui.FailSafeException
            logger.error("MouseController.move failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.MOVE,
                error_message=str(exc),
            )

    async def click(self, x: int, y: int, button: str = "left") -> ActionResult:
        """Left-, right-, or middle-click at (x, y).

        Requirements: 1.2, 1.3, 1.5
        """
        if not self._validate_coords(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message=f"Coordinate ({x}, {y}) is out of bounds.",
            )
        try:
            import pyautogui
            await asyncio.to_thread(pyautogui.click, x, y, button=button)
            return ActionResult(success=True, action_type=ActionType.CLICK)
        except Exception as exc:
            logger.error("MouseController.click failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message=str(exc),
            )

    async def double_click(self, x: int, y: int) -> ActionResult:
        """Double-click at (x, y).

        Requirements: 1.4, 1.5
        """
        if not self._validate_coords(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.DOUBLE_CLICK,
                error_message=f"Coordinate ({x}, {y}) is out of bounds.",
            )
        try:
            import pyautogui
            await asyncio.to_thread(pyautogui.doubleClick, x, y)
            return ActionResult(success=True, action_type=ActionType.DOUBLE_CLICK)
        except Exception as exc:
            logger.error("MouseController.double_click failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.DOUBLE_CLICK,
                error_message=str(exc),
            )

    async def right_click(self, x: int, y: int) -> ActionResult:
        """Right-click at (x, y).

        Requirements: 1.3, 1.5
        """
        if not self._validate_coords(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.RIGHT_CLICK,
                error_message=f"Coordinate ({x}, {y}) is out of bounds.",
            )
        try:
            import pyautogui
            await asyncio.to_thread(pyautogui.rightClick, x, y)
            return ActionResult(success=True, action_type=ActionType.RIGHT_CLICK)
        except Exception as exc:
            logger.error("MouseController.right_click failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.RIGHT_CLICK,
                error_message=str(exc),
            )

    async def scroll(
        self,
        x: int,
        y: int,
        direction: "ScrollDirection",
        amount: int = 3,
    ) -> ActionResult:
        """Scroll at (x, y) in the given direction by *amount* clicks.

        Vertical directions use ``pyautogui.scroll``; horizontal directions use
        ``pyautogui.hscroll``.

        Requirements: 6.1, 6.2, 6.6
        """
        if not self._validate_coords(x, y):
            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                error_message=f"Coordinate ({x}, {y}) is out of bounds.",
            )
        try:
            import pyautogui
            # Move to the target position first so the scroll lands in the right widget.
            await asyncio.to_thread(pyautogui.moveTo, x, y)

            if direction == ScrollDirection.UP:
                await asyncio.to_thread(pyautogui.scroll, amount, x=x, y=y)
            elif direction == ScrollDirection.DOWN:
                await asyncio.to_thread(pyautogui.scroll, -amount, x=x, y=y)
            elif direction == ScrollDirection.LEFT:
                await asyncio.to_thread(pyautogui.hscroll, -amount, x=x, y=y)
            elif direction == ScrollDirection.RIGHT:
                await asyncio.to_thread(pyautogui.hscroll, amount, x=x, y=y)
            else:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SCROLL,
                    error_message=f"Unknown scroll direction: {direction!r}",
                )
            return ActionResult(success=True, action_type=ActionType.SCROLL)
        except Exception as exc:
            logger.error("MouseController.scroll failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                error_message=str(exc),
            )

    async def drag(
        self,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
    ) -> ActionResult:
        """Drag from (src_x, src_y) to (dst_x, dst_y).

        The source coordinate must be within display bounds.  The destination
        is clamped to display bounds if it falls outside (a warning is logged).

        Requirements: 6.3, 6.4, 6.5
        """
        if not self._validate_coords(src_x, src_y):
            return ActionResult(
                success=False,
                action_type=ActionType.DRAG,
                error_message=f"Source coordinate ({src_x}, {src_y}) is out of bounds.",
            )

        # Clamp destination to screen bounds.
        clamped_x = max(0, min(dst_x, self._screen_width - 1))
        clamped_y = max(0, min(dst_y, self._screen_height - 1))
        if clamped_x != dst_x or clamped_y != dst_y:
            logger.warning(
                "MouseController.drag: destination (%d, %d) clamped to (%d, %d).",
                dst_x, dst_y, clamped_x, clamped_y,
            )

        try:
            import pyautogui
            await asyncio.to_thread(
                pyautogui.dragTo,
                clamped_x, clamped_y,
                duration=0.5,
                button="left",
            )
            return ActionResult(success=True, action_type=ActionType.DRAG)
        except Exception as exc:
            logger.error("MouseController.drag failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.DRAG,
                error_message=str(exc),
            )

# ---------------------------------------------------------------------------
# KeyboardController
# ---------------------------------------------------------------------------

class KeyboardController:
    """Controls keyboard input: text typing, key combinations, and special keys.

    All blocking pyautogui calls are offloaded via ``asyncio.to_thread`` so
    this class is fully async-compatible.

    Requirements: 2.1–2.7
    """

    def __init__(self, typing_speed_cps: float = 30.0) -> None:
        self._interval: float = 1.0 / typing_speed_cps

    async def type_text(self, text: str) -> ActionResult:
        """Type *text* character by character.

        ASCII characters are typed via ``pyautogui.typewrite`` with the
        configured inter-key interval.  Non-ASCII characters are handled by
        copying them to the clipboard via ``pyperclip`` and pasting with
        Ctrl+V so that Unicode input works reliably.

        Returns an ``ActionResult`` with ``chars_delivered`` equal to
        ``len(text)``.

        Requirements: 2.1, 2.4, 2.7
        """
        try:
            import pyautogui
            for char in text:
                if ord(char) < 128:
                    # ASCII: use typewrite with interval (pass as single-element list)
                    await asyncio.to_thread(
                        pyautogui.typewrite, [char], interval=self._interval
                    )
                else:
                    # Non-ASCII: use clipboard paste
                    try:
                        import pyperclip
                        await asyncio.to_thread(pyperclip.copy, char)
                        await asyncio.to_thread(pyautogui.hotkey, "ctrl", "v")
                    except ImportError:
                        # Fallback: try pyautogui.write which may handle some Unicode
                        await asyncio.to_thread(
                            pyautogui.write, char, interval=self._interval
                        )
            return ActionResult(
                success=True,
                action_type=ActionType.TYPE,
                chars_delivered=len(text),
            )
        except Exception as exc:
            logger.error("KeyboardController.type_text failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.TYPE,
                error_message=str(exc),
                chars_delivered=0,
            )

    async def press_key(self, key: str) -> ActionResult:
        """Press a single key by name (e.g. ``"enter"``, ``"tab"``, ``"f5"``).

        Requirements: 2.3
        """
        try:
            import pyautogui
            await asyncio.to_thread(pyautogui.press, key)
            return ActionResult(success=True, action_type=ActionType.HOTKEY)
        except Exception as exc:
            logger.error("KeyboardController.press_key failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error_message=str(exc),
            )

    async def hotkey(self, *keys: str) -> ActionResult:
        """Press a hotkey combination (e.g. ``hotkey("ctrl", "c")``).

        Requirements: 2.2
        """
        try:
            import pyautogui
            await asyncio.to_thread(pyautogui.hotkey, *keys)
            return ActionResult(success=True, action_type=ActionType.HOTKEY)
        except Exception as exc:
            logger.error("KeyboardController.hotkey failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error_message=str(exc),
            )

    async def key_combination(self, modifiers: List[str], key: str) -> ActionResult:
        """Press a key combination with explicit modifier ordering.

        Modifiers are pressed in order, the primary key is pressed and
        released, then modifiers are released in reverse order.

        Requirements: 2.2
        """
        try:
            import pyautogui
            # Press all modifiers in order
            for mod in modifiers:
                await asyncio.to_thread(pyautogui.keyDown, mod)
            # Press and release the primary key
            await asyncio.to_thread(pyautogui.keyDown, key)
            await asyncio.to_thread(pyautogui.keyUp, key)
            # Release modifiers in reverse order
            for mod in reversed(modifiers):
                await asyncio.to_thread(pyautogui.keyUp, mod)
            return ActionResult(success=True, action_type=ActionType.HOTKEY)
        except Exception as exc:
            logger.error("KeyboardController.key_combination failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.HOTKEY,
                error_message=str(exc),
            )

# ---------------------------------------------------------------------------
# ScreenReader
# ---------------------------------------------------------------------------

class ScreenReader:
    """Captures screenshots and performs OCR.

    All blocking calls are offloaded via ``asyncio.to_thread``.

    Requirements: 3.1–3.7
    """

    async def capture(self, region: Optional["ScreenRegion"] = None) -> Any:
        """Capture a screenshot, optionally restricted to *region*.

        Retries once after 500 ms on failure.

        Returns a ``PIL.Image.Image``.

        Requirements: 3.1, 3.2
        """
        from PIL import ImageGrab

        def _grab():
            bbox = None
            if region is not None:
                bbox = (
                    region.x,
                    region.y,
                    region.x + region.width,
                    region.y + region.height,
                )
            return ImageGrab.grab(bbox)

        try:
            return await asyncio.to_thread(_grab)
        except Exception as exc:
            logger.warning("ScreenReader.capture failed (attempt 1): %s — retrying in 500 ms", exc)
            await asyncio.sleep(0.5)
            try:
                return await asyncio.to_thread(_grab)
            except Exception as exc2:
                logger.error("ScreenReader.capture failed (attempt 2): %s", exc2)
                raise

    async def ocr(self, image: Any) -> List["OCRWord"]:
        """Run OCR on *image* and return a list of ``OCRWord`` objects.

        Confidence values from pytesseract (0–100) are divided by 100 and
        clamped to ``[0.0, 1.0]``.  Words with empty text are filtered out.

        Requirements: 3.3, 3.5, 3.7
        """
        import pytesseract
        from pytesseract import Output

        def _run_ocr():
            return pytesseract.image_to_data(image, output_type=Output.DICT)

        data = await asyncio.to_thread(_run_ocr)

        words: List[OCRWord] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = data["text"][i]
            if not isinstance(text, str) or not text.strip():
                continue
            raw_conf = data["conf"][i]
            try:
                raw_conf = float(raw_conf)
            except (TypeError, ValueError):
                raw_conf = 0.0
            # Clamp to [0.0, 1.0]
            confidence = max(0.0, min(1.0, raw_conf / 100.0))
            bb = ScreenRegion(
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
            words.append(OCRWord(text=text.strip(), bounding_box=bb, confidence=confidence))
        return words

    async def capture_and_ocr(self, region: Optional["ScreenRegion"] = None) -> List["OCRWord"]:
        """Capture a screenshot and run OCR on it.

        Requirements: 3.1–3.7
        """
        image = await self.capture(region)
        return await self.ocr(image)

# ---------------------------------------------------------------------------
# VisualDetector
# ---------------------------------------------------------------------------

class VisualDetector:
    """Detects UI elements via template matching and OCR-based label search.

    Requirements: 4.1–4.7
    """

    def __init__(self) -> None:
        self._screen_reader = ScreenReader()

    async def find_template(
        self,
        template: Any,
        screenshot: Any,
    ) -> Optional["UIElement"]:
        """Find *template* inside *screenshot* using normalised cross-correlation.

        Returns a ``UIElement`` if the best match correlation is ≥ 0.7,
        otherwise returns ``None``.

        Requirements: 4.5
        """
        import cv2
        import numpy as np

        def _match():
            # Convert PIL images to numpy arrays (grayscale)
            if hasattr(screenshot, "convert"):
                ss_gray = np.array(screenshot.convert("L"))
            else:
                ss_gray = np.array(screenshot)
                if ss_gray.ndim == 3:
                    ss_gray = cv2.cvtColor(ss_gray, cv2.COLOR_BGR2GRAY)

            if hasattr(template, "convert"):
                tpl_gray = np.array(template.convert("L"))
            else:
                tpl_gray = np.array(template)
                if tpl_gray.ndim == 3:
                    tpl_gray = cv2.cvtColor(tpl_gray, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(ss_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            return max_val, max_loc, tpl_gray.shape

        max_val, max_loc, tpl_shape = await asyncio.to_thread(_match)

        if max_val < 0.7:
            return None

        h, w = tpl_shape[:2]
        bb = ScreenRegion(x=max_loc[0], y=max_loc[1], width=w, height=h)
        center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
        return UIElement(
            element_type="template",
            label="",
            bounding_box=bb,
            confidence=float(max_val),
            center=center,
        )

    @staticmethod
    def _looks_like_file_path(label: str) -> bool:
        """Return ``True`` if *label* looks like a file path to an image.

        Heuristic: the label contains a path separator or ends with a common
        image extension.
        """
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp")
        lower = label.lower()
        has_separator = os.sep in label or "/" in label or "\\" in label
        has_image_ext = any(lower.endswith(ext) for ext in image_extensions)
        return has_separator or has_image_ext

    async def find_element(
        self,
        label: str,
        screenshot: Any,
    ) -> List["UIElement"]:
        """Find UI elements whose OCR text matches *label* (case-insensitive).

        If *label* looks like a file path to an image, also attempts template
        matching by loading the image from disk and calling ``find_template``.

        Returns a list sorted by confidence descending.  Returns an empty list
        (never raises) if no match is found; logs a warning in that case.

        Requirements: 4.1, 4.2, 4.3
        """
        try:
            words = await self._screen_reader.ocr(screenshot)
        except Exception as exc:
            logger.warning("VisualDetector.find_element OCR failed: %s", exc)
            return []

        label_lower = label.lower()
        elements: List[UIElement] = []
        for word in words:
            if label_lower in word.text.lower():
                bb = word.bounding_box
                center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
                elements.append(
                    UIElement(
                        element_type="text",
                        label=word.text,
                        bounding_box=bb,
                        confidence=word.confidence,
                        center=center,
                    )
                )

        # If label looks like a file path, also attempt template matching
        if self._looks_like_file_path(label):
            try:
                from PIL import Image as _PILImage
                template_img = await asyncio.to_thread(_PILImage.open, label)
                match = await self.find_template(template_img, screenshot)
                if match is not None:
                    elements.append(match)
            except Exception as exc:
                logger.warning(
                    "VisualDetector.find_element: template matching for %r failed: %s",
                    label, exc,
                )

        if not elements:
            logger.warning("VisualDetector.find_element: no match for label %r", label)

        # Sort by confidence descending
        elements.sort(key=lambda e: e.confidence, reverse=True)
        return elements

    async def find_by_type(
        self,
        element_type: str,
        screenshot: Any,
    ) -> List["UIElement"]:
        """Find UI elements of a given type using OCR heuristics and contour detection.

        Returns a list sorted by confidence descending.

        Requirements: 4.4
        """
        import cv2
        import numpy as np

        elements: List[UIElement] = []

        # OCR-based heuristics: look for common labels associated with element types
        type_keywords: Dict[str, List[str]] = {
            "button": ["ok", "cancel", "submit", "apply", "close", "yes", "no", "save"],
            "input": ["enter", "type", "search", "username", "password", "email"],
            "checkbox": ["check", "enable", "disable", "agree"],
            "link": ["click", "here", "more", "read"],
        }
        keywords = type_keywords.get(element_type.lower(), [])

        try:
            words = await self._screen_reader.ocr(screenshot)
            for word in words:
                for kw in keywords:
                    if kw in word.text.lower():
                        bb = word.bounding_box
                        center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
                        elements.append(
                            UIElement(
                                element_type=element_type,
                                label=word.text,
                                bounding_box=bb,
                                confidence=word.confidence,
                                center=center,
                            )
                        )
                        break
        except Exception as exc:
            logger.warning("VisualDetector.find_by_type OCR failed: %s", exc)

        # Contour-based detection for buttons/inputs
        try:
            def _contour_detect():
                if hasattr(screenshot, "convert"):
                    gray = np.array(screenshot.convert("L"))
                else:
                    gray = np.array(screenshot)
                    if gray.ndim == 3:
                        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours

            contours = await asyncio.to_thread(_contour_detect)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Heuristic: buttons are roughly rectangular with reasonable aspect ratio
                if element_type.lower() == "button" and 20 <= w <= 300 and 15 <= h <= 60:
                    bb = ScreenRegion(x=x, y=y, width=w, height=h)
                    center = (bb.x + bb.width // 2, bb.y + bb.height // 2)
                    elements.append(
                        UIElement(
                            element_type=element_type,
                            label="",
                            bounding_box=bb,
                            confidence=0.5,
                            center=center,
                        )
                    )
        except Exception as exc:
            logger.warning("VisualDetector.find_by_type contour detection failed: %s", exc)

        # Sort by confidence descending
        elements.sort(key=lambda e: e.confidence, reverse=True)
        return elements

    def center_of(self, element: "UIElement") -> Tuple[int, int]:
        """Return the center pixel of *element*'s bounding box.

        Requirements: 4.7
        """
        bb = element.bounding_box
        return (bb.x + bb.width // 2, bb.y + bb.height // 2)

# ---------------------------------------------------------------------------
# WindowManager
# ---------------------------------------------------------------------------

class WindowManager:
    """Lists and controls desktop windows.

    All blocking pygetwindow calls are offloaded via ``asyncio.to_thread``.
    Invalid handles or unmatched titles return a failed ``ActionResult``
    rather than raising an exception.

    Requirements: 7.1–7.8
    """

    async def list_windows(self) -> List["WindowInfo"]:
        """Return a list of all open windows.

        Requirements: 7.1
        """
        import pygetwindow

        def _get_all():
            return pygetwindow.getAllWindows()

        try:
            windows = await asyncio.to_thread(_get_all)
        except Exception as exc:
            logger.error("WindowManager.list_windows failed: %s", exc)
            return []

        result: List[WindowInfo] = []
        for win in windows:
            try:
                # Attempt to get process name via psutil/win32process
                process_name = ""
                try:
                    import psutil
                    import win32process
                    _, pid = win32process.GetWindowThreadProcessId(win._hWnd)
                    process_name = psutil.Process(pid).name()
                except Exception:
                    pass

                handle = getattr(win, "_hWnd", 0) or 0
                title = getattr(win, "title", "") or ""
                left = getattr(win, "left", 0) or 0
                top = getattr(win, "top", 0) or 0
                width = getattr(win, "width", 0) or 0
                height = getattr(win, "height", 0) or 0

                bb = ScreenRegion(x=left, y=top, width=width, height=height)
                result.append(
                    WindowInfo(
                        handle=handle,
                        title=title,
                        process_name=process_name,
                        bounding_box=bb,
                    )
                )
            except Exception as exc:
                logger.warning("WindowManager.list_windows: skipping window due to error: %s", exc)

        return result

    def _resolve_window(self, handle_or_title: Union[int, str]) -> Optional[Any]:
        """Resolve a window by integer handle or case-insensitive substring title.

        Returns the pygetwindow window object, or ``None`` if not found.

        Requirements: 7.8
        """
        import pygetwindow

        try:
            all_windows = pygetwindow.getAllWindows()
        except Exception:
            return None

        if isinstance(handle_or_title, int):
            for win in all_windows:
                if getattr(win, "_hWnd", None) == handle_or_title:
                    return win
            return None
        else:
            pattern = handle_or_title.lower()
            for win in all_windows:
                title = getattr(win, "title", "") or ""
                if pattern in title.lower():
                    return win
            return None

    async def focus(self, handle_or_title: Union[int, str]) -> "ActionResult":
        """Bring the specified window to the foreground.

        Returns a failed ``ActionResult`` if the window is not found.

        Requirements: 7.2, 7.7
        """
        try:
            win = await asyncio.to_thread(self._resolve_window, handle_or_title)
            if win is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.FOCUS_WINDOW,
                    error_message=f"Window not found: {handle_or_title!r}",
                )
            await asyncio.to_thread(win.activate)
            # Restore if minimised
            try:
                if getattr(win, "isMinimized", False):
                    await asyncio.to_thread(win.restore)
            except Exception:
                pass
            return ActionResult(success=True, action_type=ActionType.FOCUS_WINDOW)
        except Exception as exc:
            logger.error("WindowManager.focus failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error_message=str(exc),
            )

    async def resize(
        self,
        handle_or_title: Union[int, str],
        width: int,
        height: int,
    ) -> "ActionResult":
        """Resize the specified window.

        Requirements: 7.3, 7.7
        """
        try:
            win = await asyncio.to_thread(self._resolve_window, handle_or_title)
            if win is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.FOCUS_WINDOW,
                    error_message=f"Window not found: {handle_or_title!r}",
                )
            await asyncio.to_thread(win.resizeTo, width, height)
            return ActionResult(success=True, action_type=ActionType.FOCUS_WINDOW)
        except Exception as exc:
            logger.error("WindowManager.resize failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error_message=str(exc),
            )

    async def move(
        self,
        handle_or_title: Union[int, str],
        x: int,
        y: int,
    ) -> "ActionResult":
        """Move the specified window to (x, y).

        Requirements: 7.4, 7.7
        """
        try:
            win = await asyncio.to_thread(self._resolve_window, handle_or_title)
            if win is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.FOCUS_WINDOW,
                    error_message=f"Window not found: {handle_or_title!r}",
                )
            await asyncio.to_thread(win.moveTo, x, y)
            return ActionResult(success=True, action_type=ActionType.FOCUS_WINDOW)
        except Exception as exc:
            logger.error("WindowManager.move failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error_message=str(exc),
            )

    async def minimise(self, handle_or_title: Union[int, str]) -> "ActionResult":
        """Minimise the specified window.

        Requirements: 7.5, 7.7
        """
        try:
            win = await asyncio.to_thread(self._resolve_window, handle_or_title)
            if win is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.FOCUS_WINDOW,
                    error_message=f"Window not found: {handle_or_title!r}",
                )
            await asyncio.to_thread(win.minimize)
            return ActionResult(success=True, action_type=ActionType.FOCUS_WINDOW)
        except Exception as exc:
            logger.error("WindowManager.minimise failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error_message=str(exc),
            )

    async def maximise(self, handle_or_title: Union[int, str]) -> "ActionResult":
        """Maximise the specified window.

        Requirements: 7.6, 7.7
        """
        try:
            win = await asyncio.to_thread(self._resolve_window, handle_or_title)
            if win is None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.FOCUS_WINDOW,
                    error_message=f"Window not found: {handle_or_title!r}",
                )
            await asyncio.to_thread(win.maximize)
            return ActionResult(success=True, action_type=ActionType.FOCUS_WINDOW)
        except Exception as exc:
            logger.error("WindowManager.maximise failed: %s", exc)
            return ActionResult(
                success=False,
                action_type=ActionType.FOCUS_WINDOW,
                error_message=str(exc),
            )

# ---------------------------------------------------------------------------
# ActionLogger
# ---------------------------------------------------------------------------

class ActionLogger:
    """Persistent structured logger for computer-use actions.

    Writes JSON-lines to a log file and emits structured events via the
    ``janus`` Python logger.

    Requirements: 9.5, 9.6, 10.6
    """

    def __init__(self, log_path: str = "janus_computer_use.log") -> None:
        self._log_path = log_path
        self._log_file = open(log_path, "a", encoding="utf-8")  # noqa: WPS515
        self._logger = logging.getLogger("janus")

    def log(self, entry: "ActionLogEntry") -> None:
        """Append *entry* to the JSON-lines log file and emit a structured log event.

        Requirements: 9.5, 9.6, 10.6
        """
        record = {
            "action_type": entry.action_type,
            "target": entry.target,
            "timestamp": entry.timestamp,
            "outcome": entry.outcome,
            "error_message": entry.error_message,
            "screenshot_thumbnail": entry.screenshot_thumbnail,
        }
        try:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        except Exception as exc:
            logger.error("ActionLogger.log: failed to write to file: %s", exc)

        self._logger.info(
            "computer_use_action",
            extra={
                "event_type": "computer_use_action",
                "action_type": entry.action_type,
                "target": entry.target,
                "outcome": entry.outcome,
                "timestamp": entry.timestamp,
            },
        )

    def _make_thumbnail(self, image: Any) -> str:
        """Resize *image* to 160×90, encode as JPEG, and return a base64 string.

        Requirements: 9.6
        """
        try:
            from PIL import Image
            from io import BytesIO

            if not isinstance(image, Image.Image):
                # Attempt to wrap if it's a numpy array or similar
                image = Image.fromarray(image)

            thumb = image.resize((160, 90), Image.LANCZOS)
            buf = BytesIO()
            thumb.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("ActionLogger._make_thumbnail failed: %s", exc)
            return ""

    def flush(self) -> None:
        """Flush and close the log file."""
        try:
            self._log_file.flush()
            self._log_file.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Screen Recorder — capture backend selection (Requirements: 9.4, 10.2)
# ---------------------------------------------------------------------------

try:
    import mss as _mss_module
    _CAPTURE_BACKEND = "mss"
except ImportError:
    _mss_module = None  # type: ignore[assignment]
    _CAPTURE_BACKEND = "pil"
    logging.warning(
        "janus_computer_use: 'mss' not installed; falling back to PIL.ImageGrab for screen capture."
    )


def _capture_frame_mss():
    """Capture the primary monitor using mss and return a PIL.Image."""
    from PIL import Image as _PILImage
    with _mss_module.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        raw = sct.grab(monitor)
        return _PILImage.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def _capture_frame_pil():
    """Capture the screen using PIL.ImageGrab and return a PIL.Image."""
    from PIL import ImageGrab
    return ImageGrab.grab()


def _capture_frame():
    """Dispatch to the selected capture backend."""
    if _CAPTURE_BACKEND == "mss":
        return _capture_frame_mss()
    return _capture_frame_pil()


# ---------------------------------------------------------------------------
# Screen Recorder — data models (Requirements: 8.1, 8.2, 2.5, 4.5)
# ---------------------------------------------------------------------------

def _validate_range(name: str, value, min_val, max_val) -> None:
    """Raise ValueError if *value* is outside [min_val, max_val].

    Requirements: 8.2
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"ScreenRecorder: '{name}' must be in [{min_val}, {max_val}], got {value!r}"
        )


@dataclass
class ScreenRecorderConfig:
    """Validated configuration for ScreenRecorder.

    All parameters are validated in ``__post_init__``.

    Requirements: 8.1, 8.2
    """

    capture_rate_fps: int = 5
    buffer_duration_seconds: int = 30
    motion_threshold: int = 5
    high_motion_threshold: int = 20
    stuck_duration_seconds: int = 10
    transition_settling_seconds: float = 1.5
    temporal_context_frames: int = 3
    gif_max_dimension: int = 640

    def __post_init__(self) -> None:
        _validate_range("capture_rate_fps", self.capture_rate_fps, 1, 30)
        _validate_range("buffer_duration_seconds", self.buffer_duration_seconds, 5, 300)
        _validate_range("motion_threshold", self.motion_threshold, 0, 64)
        _validate_range("high_motion_threshold", self.high_motion_threshold, 1, 64)
        _validate_range("stuck_duration_seconds", self.stuck_duration_seconds, 1, 3600)
        _validate_range("transition_settling_seconds", self.transition_settling_seconds, 0.1, 60.0)
        _validate_range("temporal_context_frames", self.temporal_context_frames, 1, 10)
        _validate_range("gif_max_dimension", self.gif_max_dimension, 64, 4096)

    @property
    def buffer_capacity(self) -> int:
        """Total number of frames the ring buffer can hold."""
        return self.capture_rate_fps * self.buffer_duration_seconds


@dataclass
class RecordedFrame:
    """A single captured screen frame.

    Requirements: 2.5
    """

    image: Any          # PIL.Image.Image
    timestamp: float    # time.monotonic() at capture
    phash: Any          # imagehash.ImageHash


@dataclass
class ScreenClip:
    """A contiguous sequence of frames extracted from the ring buffer.

    Requirements: 2.5, 4.5
    """

    frames: List[RecordedFrame]
    start_time: float
    end_time: float
    warning: Optional[str] = None

    @property
    def frame_count(self) -> int:
        """Total number of frames in this clip."""
        return len(self.frames)


@dataclass
class EncodeResult:
    """Outcome of a video encode operation.

    Requirements: 4.5
    """

    success: bool
    output_path: Optional[str] = None
    file_size_bytes: int = 0
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# ScreenRecorder (Requirements: 1.x, 2.x, 3.x, 5.x, 6.x, 7.x, 8.x)
# ---------------------------------------------------------------------------

class ScreenRecorder:
    """Continuous background screen capture with ring-buffer storage, motion
    detection, clip extraction, and event callbacks.

    Usage (standalone)::

        async with ScreenRecorder(capture_rate_fps=5) as recorder:
            clip = await recorder.get_clip(start, end)

    Requirements: 1.1–1.7, 2.1–2.5, 3.1–3.7, 5.1–5.6, 6.1, 7.1–7.6, 8.1–8.4
    """

    def __init__(
        self,
        capture_rate_fps: int = 5,
        buffer_duration_seconds: int = 30,
        motion_threshold: int = 5,
        high_motion_threshold: int = 20,
        stuck_duration_seconds: int = 10,
        transition_settling_seconds: float = 1.5,
        temporal_context_frames: int = 3,
        gif_max_dimension: int = 640,
    ) -> None:
        """Initialise the recorder.  Validation happens inside ScreenRecorderConfig.

        Requirements: 8.1, 8.3
        """
        import dataclasses as _dc
        self._config = ScreenRecorderConfig(
            capture_rate_fps=capture_rate_fps,
            buffer_duration_seconds=buffer_duration_seconds,
            motion_threshold=motion_threshold,
            high_motion_threshold=high_motion_threshold,
            stuck_duration_seconds=stuck_duration_seconds,
            transition_settling_seconds=transition_settling_seconds,
            temporal_context_frames=temporal_context_frames,
            gif_max_dimension=gif_max_dimension,
        )
        self._dc = _dc

        # Ring buffer — deque with fixed maxlen handles eviction automatically
        self._buffer: deque = deque(maxlen=self._config.buffer_capacity)

        # Motion tracking
        self._last_frame_diff: int = 0
        self._diff_history: deque = deque()          # (timestamp, diff) pairs
        self._last_retained_frame: Optional[RecordedFrame] = None
        self._last_motion_time: float = 0.0

        # Transition settling state
        self._in_transition: bool = False
        self._transition_settled_at: Optional[float] = None

        # Lifecycle
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Event callbacks
        self._transition_callbacks: List[Any] = []
        self._stuck_callbacks: List[Any] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Dict[str, Any]:
        """All configuration parameters as a plain dict.

        Requirements: 8.3
        """
        return self._dc.asdict(self._config)

    @property
    def is_running(self) -> bool:
        """True while the background capture loop is active."""
        return self._running

    @property
    def last_frame_diff(self) -> int:
        """Most recent perceptual-hash distance between consecutive frames.

        Returns 0 if no frames have been captured yet.

        Requirements: 3.6
        """
        return self._last_frame_diff

    @property
    def motion_score(self) -> float:
        """Sum of frame-diff values recorded in the last 5 seconds.

        Requirements: 5.6
        """
        import time as _time
        now = _time.monotonic()
        cutoff = now - 5.0
        return float(sum(d for ts, d in self._diff_history if ts >= cutoff))

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def on_ui_transition(self, callback) -> None:
        """Register an async callback for UI_Transition_Complete events.

        Requirements: 5.4
        """
        self._transition_callbacks.append(callback)

    def on_stuck_state(self, callback) -> None:
        """Register an async callback for Stuck_State events.

        Requirements: 5.4
        """
        self._stuck_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal — event firing
    # ------------------------------------------------------------------

    async def _fire_transition_callbacks(self) -> None:
        """Invoke all registered UI-transition callbacks.

        Requirements: 5.4, 5.5
        """
        for cb in self._transition_callbacks:
            try:
                await cb()
            except Exception as exc:
                logger.error("ScreenRecorder._fire_transition_callbacks: callback error: %s", exc)

    async def _fire_stuck_callbacks(self, duration: float, last_frame: Optional[RecordedFrame]) -> None:
        """Invoke all registered stuck-state callbacks.

        Requirements: 5.4, 5.5
        """
        for cb in self._stuck_callbacks:
            try:
                await cb(duration=duration, last_frame=last_frame)
            except Exception as exc:
                logger.error("ScreenRecorder._fire_stuck_callbacks: callback error: %s", exc)

    # ------------------------------------------------------------------
    # Internal — frame capture
    # ------------------------------------------------------------------

    async def _capture_one_frame(self) -> RecordedFrame:
        """Capture a single frame and compute its perceptual hash.

        Requirements: 1.1, 3.1
        """
        import time as _time
        import imagehash as _imagehash

        image = await asyncio.to_thread(_capture_frame)
        phash = await asyncio.to_thread(_imagehash.phash, image)
        return RecordedFrame(image=image, timestamp=_time.monotonic(), phash=phash)

    # ------------------------------------------------------------------
    # Internal — motion detection and frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: RecordedFrame) -> None:
        """Apply motion detection and append the frame to the ring buffer if retained.

        Requirements: 3.1–3.5, 3.7
        """
        import time as _time
        import asyncio as _asyncio

        if self._last_retained_frame is None:
            # Always retain the very first frame
            self._buffer.append(frame)
            self._last_retained_frame = frame
            self._last_motion_time = frame.timestamp
            return

        diff = self._last_retained_frame.phash - frame.phash
        self._last_frame_diff = diff

        # Update rolling diff history; prune entries older than 5 seconds
        self._diff_history.append((frame.timestamp, diff))
        cutoff = frame.timestamp - 5.0
        while self._diff_history and self._diff_history[0][0] < cutoff:
            self._diff_history.popleft()

        # Retain frame if motion threshold is disabled (0) or diff meets threshold
        if self._config.motion_threshold == 0 or diff >= self._config.motion_threshold:
            self._buffer.append(frame)
            self._last_retained_frame = frame
            self._last_motion_time = frame.timestamp

            # High-motion → UI transition detection
            if diff > self._config.high_motion_threshold:
                self._in_transition = True
                self._transition_settled_at = None
                # Schedule callback fire without blocking
                try:
                    loop = _asyncio.get_event_loop()
                    loop.create_task(self._fire_transition_callbacks())
                except RuntimeError:
                    pass  # no running loop in tests

        # UI transition settling logic
        if self._in_transition:
            if diff < self._config.motion_threshold:
                now = _time.monotonic()
                if self._transition_settled_at is None:
                    self._transition_settled_at = now
                elif now - self._transition_settled_at >= self._config.transition_settling_seconds:
                    self._in_transition = False
                    self._transition_settled_at = None
                    try:
                        loop = _asyncio.get_event_loop()
                        loop.create_task(self._fire_transition_callbacks())
                    except RuntimeError:
                        pass

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background capture loop.

        If already running, returns immediately.

        Requirements: 1.4, 1.5
        """
        if self._running:
            return
        self._running = True
        import time as _time
        self._last_motion_time = _time.monotonic()
        self._task = asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        """Stop the background capture loop and release frame memory.

        Requirements: 1.6, 7.2
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._buffer.clear()

    async def __aenter__(self) -> "ScreenRecorder":
        """Start the recorder and return self.

        Requirements: 7.6
        """
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        """Stop the recorder.

        Requirements: 7.6
        """
        await self.stop()

    async def _capture_loop(self) -> None:
        """Background capture loop — runs until ``_running`` is False.

        Requirements: 1.4, 1.5, 1.7
        """
        import time as _time

        async def _run_once() -> None:
            while self._running:
                t0 = _time.monotonic()
                try:
                    frame = await self._capture_one_frame()
                    self._process_frame(frame)

                    # Stuck-state check
                    now = _time.monotonic()
                    idle = now - self._last_motion_time
                    if idle >= self._config.stuck_duration_seconds:
                        await self._fire_stuck_callbacks(
                            duration=idle,
                            last_frame=self._last_retained_frame,
                        )
                        # Reset to avoid repeated firing every frame
                        self._last_motion_time = now

                except Exception as exc:
                    logger.warning("ScreenRecorder._capture_loop: frame error: %s", exc)

                elapsed = _time.monotonic() - t0
                sleep_time = max(0.0, 1.0 / self._config.capture_rate_fps - elapsed)
                await asyncio.sleep(sleep_time)

        try:
            await _run_once()
        except Exception as exc:
            logger.error("ScreenRecorder._capture_loop: unhandled exception: %s", exc)
            # Attempt one restart
            try:
                self._running = True
                await _run_once()
            except Exception as exc2:
                logger.error("ScreenRecorder._capture_loop: restart failed: %s", exc2)
                self._running = False
                await self._fire_stuck_callbacks(
                    duration=0.0,
                    last_frame=self._last_retained_frame,
                )

    # ------------------------------------------------------------------
    # Clip extraction
    # ------------------------------------------------------------------

    async def get_clip(self, start_time: float, end_time: float) -> ScreenClip:
        """Extract frames from the ring buffer within [start_time, end_time].

        Requirements: 2.1, 2.3, 2.5
        """
        snapshot = list(self._buffer)
        filtered = [f for f in snapshot if start_time <= f.timestamp <= end_time]
        filtered.sort(key=lambda f: f.timestamp)

        warning: Optional[str] = None
        if snapshot and snapshot[0].timestamp > start_time:
            warning = "Ring buffer does not cover the full requested interval"
        elif not snapshot:
            warning = "Ring buffer does not cover the full requested interval"

        return ScreenClip(
            frames=filtered,
            start_time=start_time,
            end_time=end_time,
            warning=warning,
        )

    async def get_recent_frames(self, n: int) -> List[RecordedFrame]:
        """Return the last *n* frames from the ring buffer (most recent last).

        Requirements: 6.1
        """
        snapshot = list(self._buffer)
        count = min(n, len(snapshot))
        return snapshot[-count:] if count > 0 else []


# ---------------------------------------------------------------------------
# VideoEncoder (Requirements: 4.1–4.8)
# ---------------------------------------------------------------------------

class VideoEncoder:
    """Stateless async video encoder.  Encodes a ``ScreenClip`` to MP4 or GIF.

    All encoding work is offloaded via ``asyncio.to_thread`` so the event loop
    is never blocked.

    Requirements: 4.1–4.8
    """

    async def encode_mp4(self, clip: ScreenClip, output_path: str) -> EncodeResult:
        """Encode *clip* to an MP4 file at *output_path*.

        Requirements: 4.1, 4.3, 4.6, 4.7
        """
        if not clip.frames:
            return EncodeResult(
                success=False,
                error_message="Cannot encode empty clip to MP4",
            )

        def _encode() -> None:
            import cv2
            import numpy as np

            frames = clip.frames
            if len(frames) >= 2:
                duration = frames[-1].timestamp - frames[0].timestamp
                fps = len(frames) / duration if duration > 0 else 5.0
            else:
                fps = 5.0

            w, h = frames[0].image.size
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            try:
                for frame in frames:
                    arr = cv2.cvtColor(np.array(frame.image), cv2.COLOR_RGB2BGR)
                    writer.write(arr)
            finally:
                writer.release()

        try:
            await asyncio.to_thread(_encode)
            size = os.path.getsize(output_path)
            return EncodeResult(success=True, output_path=output_path, file_size_bytes=size)
        except Exception as exc:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return EncodeResult(success=False, error_message=str(exc))

    async def encode_gif(
        self,
        clip: ScreenClip,
        output_path: str,
        max_dimension: int = 640,
    ) -> EncodeResult:
        """Encode *clip* to an animated GIF at *output_path*.

        Requirements: 4.2, 4.4, 4.6, 4.7
        """
        if not clip.frames:
            return EncodeResult(
                success=False,
                error_message="Cannot encode empty clip to GIF",
            )

        def _encode() -> None:
            from PIL import Image as _PILImage

            images = []
            for frame in clip.frames:
                img = frame.image.copy()
                w, h = img.size
                if max(w, h) > max_dimension:
                    scale = max_dimension / max(w, h)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    img = img.resize((new_w, new_h), _PILImage.LANCZOS)
                images.append(img.convert("P", palette=_PILImage.ADAPTIVE))

            durations = []
            for i in range(len(clip.frames)):
                if i + 1 < len(clip.frames):
                    d = int(
                        (clip.frames[i + 1].timestamp - clip.frames[i].timestamp) * 1000
                    )
                else:
                    d = 200
                durations.append(max(20, d))

            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                loop=0,
                duration=durations,
            )

        try:
            await asyncio.to_thread(_encode)
            size = os.path.getsize(output_path)
            return EncodeResult(success=True, output_path=output_path, file_size_bytes=size)
        except Exception as exc:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return EncodeResult(success=False, error_message=str(exc))


# ---------------------------------------------------------------------------
# ComputerUseEngine
# ---------------------------------------------------------------------------

class ComputerUseEngine:
    """Top-level async context manager that coordinates all computer-use sub-systems.

    Usage::

        async with ComputerUseEngine(context={"goal": "..."}) as engine:
            result = await engine.execute_action(action)

    Requirements: 9.1, 9.2, 9.4, 9.5, 9.6, 10.4, 10.5
    """

    # Keywords that trigger the destructive-action safety pause
    _DESTRUCTIVE_KEYWORDS: List[str] = [
        "delete", "remove", "uninstall", "format", "erase",
        "permanently", "cannot be undone",
    ]

    # Keywords that indicate an error dialog is present
    _ERROR_DIALOG_KEYWORDS: List[str] = [
        "Error", "Warning", "Exception", "Failed", "Access Denied",
    ]

    def __init__(self, context: Optional[Dict[str, Any]] = None, enable_temporal_context: bool = False) -> None:
        """Initialise the engine and all sub-systems.

        Parameters
        ----------
        context:
            Optional session context dict (e.g. job_id, goal, platform).
            Accessible to ``ActionPlanner`` via ``engine._context``.
        enable_temporal_context:
            When ``True``, a ``ScreenRecorder`` is started in ``__aenter__``
            and stopped in ``__aexit__``.  Defaults to ``False``.

        Requirements: 10.4, 6.6, 7.1
        """
        self._context: Dict[str, Any] = context or {}
        self._enable_temporal_context: bool = enable_temporal_context

        # Sub-systems
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._screen = ScreenReader()
        self._vision = VisualDetector()
        self._windows = WindowManager()
        self._logger = ActionLogger()

        # Stuck-state detection: rolling buffer of the last 3 perceptual hashes
        self._hash_buffer: deque = deque(maxlen=3)

        # ScreenRecorder — instantiated in __aenter__ when enable_temporal_context=True
        self._screen_recorder: Optional[ScreenRecorder] = None

        # ActionPlanner is initialised in __aenter__ (requires engine to be ready)
        self._planner: Optional[Any] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ComputerUseEngine":
        """Enter the context: check dependencies and initialise ActionPlanner.

        Requirements: 10.5, 7.1
        """
        _check_dependencies()

        # Start ScreenRecorder if temporal context is enabled
        if self._enable_temporal_context:
            self._screen_recorder = ScreenRecorder()
            await self._screen_recorder.start()

        # Attempt to initialise ActionPlanner if it is available.
        # ActionPlanner is implemented in Task 11; until then we set it to None.
        try:
            # ActionPlanner is defined later in this module (Task 11).
            # Use a forward reference so this works once it is implemented.
            planner_cls = globals().get("ActionPlanner")
            if planner_cls is not None:
                # Try to get AvusBrain if available
                try:
                    from avus_brain import AvusBrain
                    brain = AvusBrain()
                except Exception:
                    brain = None
                self._planner = planner_cls(engine=self, brain=brain)
            else:
                self._planner = None
        except Exception as exc:
            logger.warning(
                "ComputerUseEngine.__aenter__: ActionPlanner not available: %s", exc
            )
            self._planner = None

        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the context: flush the action log and release resources.

        Requirements: 10.5, 7.2
        """
        # Stop ScreenRecorder if it was started
        if self._screen_recorder is not None:
            try:
                await self._screen_recorder.stop()
            except Exception as exc:
                logger.warning("ComputerUseEngine.__aexit__: recorder stop failed: %s", exc)

        try:
            self._logger.flush()
        except Exception as exc:
            logger.warning("ComputerUseEngine.__aexit__: logger flush failed: %s", exc)

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------

    @property
    def mouse(self) -> MouseController:
        """The ``MouseController`` sub-system."""
        return self._mouse

    @property
    def keyboard(self) -> KeyboardController:
        """The ``KeyboardController`` sub-system."""
        return self._keyboard

    @property
    def screen(self) -> ScreenReader:
        """The ``ScreenReader`` sub-system."""
        return self._screen

    @property
    def vision(self) -> VisualDetector:
        """The ``VisualDetector`` sub-system."""
        return self._vision

    @property
    def windows(self) -> WindowManager:
        """The ``WindowManager`` sub-system."""
        return self._windows

    @property
    def planner(self) -> Optional[Any]:
        """The ``ActionPlanner`` sub-system (``None`` until Task 11 is implemented)."""
        return self._planner

    @property
    def recorder(self) -> Optional["ScreenRecorder"]:
        """The ``ScreenRecorder`` sub-system (``None`` when temporal context is disabled).

        Requirements: 7.1
        """
        return self._screen_recorder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ocr_words_contain(self, words: List[OCRWord], keywords: List[str]) -> Optional[str]:
        """Return the first keyword found in *words*, or ``None``."""
        full_text = " ".join(w.text for w in words).lower()
        for kw in keywords:
            if kw.lower() in full_text:
                return kw
        return None

    async def _capture_thumbnail(self) -> str:
        """Capture a screenshot and return a base64-encoded JPEG thumbnail."""
        try:
            screenshot = await self._screen.capture()
            return self._logger._make_thumbnail(screenshot)
        except Exception as exc:
            logger.warning("ComputerUseEngine._capture_thumbnail failed: %s", exc)
            return ""

    async def _check_stuck_state(self) -> bool:
        """Capture a screenshot hash and check for stuck state.

        Appends the current perceptual hash to ``_hash_buffer``.  Returns
        ``True`` if all 3 hashes in the buffer have pairwise distance ≤ 5
        (i.e. the screen has not changed for 3 consecutive actions).

        Requirements: 9.4
        """
        try:
            import imagehash
            screenshot = await self._screen.capture()
            current_hash = imagehash.phash(screenshot)
            self._hash_buffer.append(current_hash)

            if len(self._hash_buffer) < 3:
                return False

            hashes = list(self._hash_buffer)
            # Check all pairwise distances
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    if (hashes[i] - hashes[j]) > 5:
                        return False
            return True
        except Exception as exc:
            logger.warning("ComputerUseEngine._check_stuck_state failed: %s", exc)
            return False

    async def _try_dismiss_error_dialog(self) -> None:
        """Attempt to dismiss an error dialog by pressing Escape or clicking OK/Close."""
        try:
            # Try Escape first
            await self._keyboard.press_key("escape")
            await asyncio.sleep(0.3)

            # Also try clicking the first "OK" or "Close" button found
            screenshot = await self._screen.capture()
            for label in ("OK", "Close", "ok", "close"):
                elements = await self._vision.find_element(label, screenshot)
                if elements:
                    cx, cy = elements[0].center
                    await self._mouse.click(cx, cy)
                    await asyncio.sleep(0.3)
                    break
        except Exception as exc:
            logger.warning("ComputerUseEngine._try_dismiss_error_dialog failed: %s", exc)

    async def _dispatch_action(self, action: "Action") -> "ActionResult":
        """Dispatch *action* to the appropriate sub-system.

        Requirements: 9.1
        """
        p = action.params
        at = action.action_type

        if at == ActionType.MOVE:
            return await self._mouse.move(
                p.get("x", 0), p.get("y", 0),
                human_like=p.get("human_like", False),
            )
        elif at == ActionType.CLICK:
            return await self._mouse.click(
                p.get("x", 0), p.get("y", 0),
                button=p.get("button", "left"),
            )
        elif at == ActionType.RIGHT_CLICK:
            return await self._mouse.right_click(p.get("x", 0), p.get("y", 0))
        elif at == ActionType.DOUBLE_CLICK:
            return await self._mouse.double_click(p.get("x", 0), p.get("y", 0))
        elif at == ActionType.TYPE:
            return await self._keyboard.type_text(p.get("text", ""))
        elif at == ActionType.HOTKEY:
            keys = p.get("keys", [])
            if isinstance(keys, list):
                return await self._keyboard.hotkey(*keys)
            return await self._keyboard.press_key(str(keys))
        elif at == ActionType.SCROLL:
            direction_val = p.get("direction", "down")
            try:
                direction = ScrollDirection(direction_val)
            except ValueError:
                direction = ScrollDirection.DOWN
            return await self._mouse.scroll(
                p.get("x", 0), p.get("y", 0),
                direction=direction,
                amount=p.get("amount", 3),
            )
        elif at == ActionType.DRAG:
            return await self._mouse.drag(
                p.get("src_x", 0), p.get("src_y", 0),
                p.get("dst_x", 0), p.get("dst_y", 0),
            )
        elif at == ActionType.SCREENSHOT:
            region = p.get("region")
            screenshot = await self._screen.capture(region)
            return ActionResult(success=True, action_type=at, data=screenshot)
        elif at == ActionType.OCR:
            region = p.get("region")
            words = await self._screen.capture_and_ocr(region)
            return ActionResult(success=True, action_type=at, data=words)
        elif at == ActionType.FIND_ELEMENT:
            screenshot = await self._screen.capture()
            label = p.get("label", "")
            elements = await self._vision.find_element(label, screenshot)
            return ActionResult(success=True, action_type=at, data=elements)
        elif at == ActionType.FOCUS_WINDOW:
            return await self._windows.focus(p.get("handle_or_title", ""))
        elif at == ActionType.WAIT:
            condition = p.get("condition")
            if condition is not None:
                return await self.wait_for(condition)
            return ActionResult(
                success=False,
                action_type=at,
                error_message="WAIT action requires a 'condition' parameter.",
            )
        else:
            return ActionResult(
                success=False,
                action_type=at,
                error_message=f"Unknown action type: {at!r}",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_action(self, action: "Action") -> "ActionResult":
        """Execute *action* with safety guards, error-dialog recovery, and logging.

        The entire execution is wrapped in a 30-second timeout.

        Safety pipeline
        ---------------
        1. Capture before-screenshot thumbnail for the log.
        2. Run destructive-action OCR scan; pause if keywords found and
           ``action.pre_approved`` is ``False``.
        3. Dispatch to the appropriate sub-system.
        4. Run error-dialog OCR scan; attempt dismissal and retry once if
           a dialog is detected.
        5. Check for stuck state (3 consecutive no-change screens).
        6. Log the action result.

        Requirements: 9.1, 9.2, 9.5, 9.6
        """
        async def _inner() -> ActionResult:
            # Step 1: capture before-screenshot thumbnail
            thumbnail = await self._capture_thumbnail()

            # Step 2: destructive-action check
            if not action.pre_approved:
                try:
                    screen_words = await self._screen.capture_and_ocr()
                    found_kw = self._ocr_words_contain(
                        screen_words, self._DESTRUCTIVE_KEYWORDS
                    )
                    if found_kw:
                        dialog_text = " ".join(w.text for w in screen_words)
                        result = ActionResult(
                            success=False,
                            action_type=action.action_type,
                            error_message=(
                                f"Destructive action paused: keyword '{found_kw}' "
                                f"detected on screen. Dialog text: {dialog_text[:200]}"
                            ),
                        )
                        self._logger.log(
                            ActionLogEntry(
                                action_type=action.action_type.value,
                                target=str(action.params),
                                timestamp=datetime.datetime.utcnow().isoformat(),
                                outcome="failure",
                                error_message=result.error_message,
                                screenshot_thumbnail=thumbnail,
                            )
                        )
                        return result
                except Exception as exc:
                    logger.warning(
                        "ComputerUseEngine.execute_action: destructive check failed: %s", exc
                    )

            # Step 3: dispatch to sub-system
            result = await self._dispatch_action(action)

            # Step 4: error-dialog recovery
            try:
                post_words = await self._screen.capture_and_ocr()
                if self._ocr_words_contain(post_words, self._ERROR_DIALOG_KEYWORDS):
                    # Attempt to dismiss the dialog
                    await self._try_dismiss_error_dialog()
                    await asyncio.sleep(0.5)

                    # Retry the action once
                    result = await self._dispatch_action(action)

                    # Check if dialog reappeared
                    retry_words = await self._screen.capture_and_ocr()
                    if self._ocr_words_contain(retry_words, self._ERROR_DIALOG_KEYWORDS):
                        dialog_text = " ".join(w.text for w in retry_words)
                        result = ActionResult(
                            success=False,
                            action_type=action.action_type,
                            error_message=(
                                f"Error dialog persists after retry: {dialog_text[:200]}"
                            ),
                        )
            except Exception as exc:
                logger.warning(
                    "ComputerUseEngine.execute_action: error-dialog check failed: %s", exc
                )

            # Step 5: stuck-state detection
            if result.success:
                stuck = await self._check_stuck_state()
                if stuck:
                    result = ActionResult(
                        success=False,
                        action_type=action.action_type,
                        error_message=(
                            "Stuck state detected after 3 consecutive no-change actions"
                        ),
                    )

            # Step 6: log the action
            outcome = "success" if result.success else "failure"
            self._logger.log(
                ActionLogEntry(
                    action_type=action.action_type.value,
                    target=str(action.params),
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    outcome=outcome,
                    error_message=result.error_message,
                    screenshot_thumbnail=thumbnail,
                )
            )

            return result

        try:
            return await asyncio.wait_for(_inner(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(
                "ComputerUseEngine.execute_action: timed out after 30 s for action %s",
                action.action_type,
            )
            return ActionResult(
                success=False,
                action_type=action.action_type,
                error_message="Action timed out after 30 seconds.",
            )
        except Exception as exc:
            logger.error("ComputerUseEngine.execute_action: unexpected error: %s", exc)
            return ActionResult(
                success=False,
                action_type=action.action_type,
                error_message=str(exc),
            )

    async def wait_for(self, condition: "WaitCondition") -> "ActionResult":
        """Poll until *condition* is satisfied or the timeout expires.

        The timeout is clamped to a maximum of 300 seconds.

        Supported condition types
        -------------------------
        - ``TEXT_PRESENT``: OCR scan until ``condition.target`` text appears.
        - ``TEXT_GONE``: OCR scan until ``condition.target`` text disappears.
        - ``ELEMENT_VISIBLE``: ``find_element`` until a match is found.
        - ``ELEMENT_GONE``: ``find_element`` until no match is found.
        - ``IMAGE_PRESENT``: template matching until the template image is found.

        Requirements: 5.1–5.6
        """
        # Clamp timeout to 300 seconds
        timeout = min(condition.timeout_seconds, 300.0)
        poll_interval = condition.poll_interval_seconds
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            try:
                screenshot = await self._screen.capture()
                satisfied = False

                ct = condition.condition_type

                if ct == WaitConditionType.TEXT_PRESENT:
                    words = await self._screen.ocr(screenshot)
                    text_lower = condition.target.lower()
                    satisfied = any(text_lower in w.text.lower() for w in words)

                elif ct == WaitConditionType.TEXT_GONE:
                    words = await self._screen.ocr(screenshot)
                    text_lower = condition.target.lower()
                    satisfied = not any(text_lower in w.text.lower() for w in words)

                elif ct == WaitConditionType.ELEMENT_VISIBLE:
                    elements = await self._vision.find_element(condition.target, screenshot)
                    satisfied = len(elements) > 0

                elif ct == WaitConditionType.ELEMENT_GONE:
                    elements = await self._vision.find_element(condition.target, screenshot)
                    satisfied = len(elements) == 0

                elif ct == WaitConditionType.IMAGE_PRESENT:
                    try:
                        from PIL import Image as _PILImage
                        template = await asyncio.to_thread(
                            _PILImage.open, condition.target
                        )
                        match = await self._vision.find_template(template, screenshot)
                        satisfied = match is not None
                    except Exception as exc:
                        logger.warning(
                            "ComputerUseEngine.wait_for: IMAGE_PRESENT template load failed: %s",
                            exc,
                        )
                        satisfied = False

                else:
                    return ActionResult(
                        success=False,
                        action_type=ActionType.WAIT,
                        error_message=f"Unknown WaitConditionType: {ct!r}",
                    )

                if satisfied:
                    return ActionResult(success=True, action_type=ActionType.WAIT)

            except Exception as exc:
                logger.warning("ComputerUseEngine.wait_for: poll error: %s", exc)

            # Check timeout
            if asyncio.get_event_loop().time() >= deadline:
                return ActionResult(
                    success=False,
                    action_type=ActionType.WAIT,
                    error_message=(
                        f"wait_for timed out after {timeout:.1f} s waiting for "
                        f"{condition.condition_type.value!r}: {condition.target!r}"
                    ),
                )

            await asyncio.sleep(poll_interval)

    async def run_goal(self, goal: str, max_steps: int = 50) -> "ActionResult":
        """Run an autonomous goal-completion loop via ``ActionPlanner``.

        Delegates to ``self.planner.run(goal, max_steps)``.

        Requirements: 8.1–8.7
        """
        if self._planner is None:
            return ActionResult(
                success=False,
                action_type=ActionType.SCREENSHOT,
                error_message=(
                    "ActionPlanner is not initialised. "
                    "Ensure ComputerUseEngine is used as an async context manager."
                ),
            )
        return await self._planner.run(goal, max_steps=max_steps)

# ---------------------------------------------------------------------------
# ActionPlanner
# ---------------------------------------------------------------------------

class ActionPlanner:
    """Perceive → reason → act loop backed by an AI brain (e.g. AvusBrain).

    The planner captures the current screen, runs OCR and visual element
    detection, builds a structured prompt, asks the brain for candidate
    actions, executes the top-ranked action, and records the step.  This
    cycle repeats until the goal is achieved, the maximum number of steps is
    reached, or a stuck state is detected.

    Requirements: 8.1–8.7
    """

    # Keywords in OCR text that indicate the goal has been achieved
    _SUCCESS_KEYWORDS: List[str] = [
        "success", "complete", "completed", "done", "finished",
        "submitted", "confirmed", "saved",
    ]

    def __init__(self, engine: "ComputerUseEngine", brain: Any) -> None:
        """Initialise the planner.

        Parameters
        ----------
        engine:
            The ``ComputerUseEngine`` instance that owns this planner.
        brain:
            An object with an ``ask(prompt: str) -> str`` method (e.g.
            ``AvusBrain``).  May be ``None``; in that case ``plan_next``
            returns a fallback screenshot action.

        Requirements: 8.1, 10.4
        """
        self._engine = engine
        self._brain = brain
        self._context: Dict[str, Any] = engine._context
        self._history: List[StepRecord] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        goal: str,
        ocr_words: List[OCRWord],
        elements: List[UIElement],
        history: List[StepRecord],
    ) -> str:
        """Build the structured prompt sent to the brain.

        Requirements: 8.2
        """
        ocr_text = " ".join(w.text for w in ocr_words) if ocr_words else "(no text detected)"

        if elements:
            element_lines = []
            for el in elements[:20]:  # cap at 20 to keep prompt manageable
                element_lines.append(
                    f"  - type={el.element_type!r} label={el.label!r} "
                    f"center={el.center} confidence={el.confidence:.2f}"
                )
            element_list = "\n".join(element_lines)
        else:
            element_list = "(no elements detected)"

        # Last 5 history entries
        recent = history[-5:] if len(history) >= 5 else history
        if recent:
            history_lines = []
            for rec in recent:
                outcome = "success" if rec.result.success else "failure"
                history_lines.append(
                    f"  Step {rec.step_number}: {rec.action.action_type.value} "
                    f"params={rec.action.params} → {outcome}"
                )
            history_summary = "\n".join(history_lines)
        else:
            history_summary = "(no previous steps)"

        return (
            f"GOAL: {goal}\n\n"
            f"CURRENT SCREEN (OCR text):\n{ocr_text}\n\n"
            f"DETECTED ELEMENTS:\n{element_list}\n\n"
            f"RECENT HISTORY (last 5 steps):\n{history_summary}\n\n"
            "Based on the above, list the top 3 actions to take next to achieve the goal.\n"
            "For each action, provide: action_type, target, parameters, confidence (0-1), rationale.\n"
            "Format as JSON array."
        )

    def _parse_brain_response(self, response: str) -> List[CandidateAction]:
        """Parse the brain's JSON response into a list of ``CandidateAction`` objects.

        Returns an empty list if parsing fails (caller should re-prompt).

        Requirements: 8.2, 8.3
        """
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            text = "\n".join(inner)

        try:
            raw_list = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(raw_list, list):
            return []

        candidates: List[CandidateAction] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            try:
                # Resolve action_type — accept string or enum value
                raw_at = item.get("action_type", "screenshot")
                try:
                    action_type = ActionType(str(raw_at).lower())
                except ValueError:
                    action_type = ActionType.SCREENSHOT

                params = item.get("params", item.get("parameters", {}))
                if not isinstance(params, dict):
                    params = {}

                raw_conf = item.get("confidence", 0.5)
                try:
                    confidence = float(raw_conf)
                except (TypeError, ValueError):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))

                rationale = str(item.get("rationale", ""))

                action = Action(action_type=action_type, params=params)
                candidates.append(
                    CandidateAction(
                        action=action,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )
            except Exception as exc:
                logger.warning("ActionPlanner._parse_brain_response: skipping item: %s", exc)
                continue

        return candidates

    def _fallback_screenshot_action(self) -> List[CandidateAction]:
        """Return a single fallback screenshot action when the brain is unavailable."""
        return [
            CandidateAction(
                action=Action(action_type=ActionType.SCREENSHOT, params={}),
                confidence=1.0,
                rationale="Brain unavailable — taking screenshot as fallback.",
            )
        ]

    def _image_to_thumbnail_b64(self, image: Any) -> str:
        """Convert a PIL image to a base64-encoded JPEG thumbnail string."""
        try:
            return self._engine._logger._make_thumbnail(image)
        except Exception:
            return ""

    def _build_temporal_context_section(
        self,
        frames: List[RecordedFrame],
        recorder: "ScreenRecorder",
        now: float,
    ) -> str:
        """Build the temporal context text block for the prompt.

        Each frame is encoded as a base64 JPEG thumbnail (max 320×240 px).

        Requirements: 6.2, 6.4
        """
        from io import BytesIO as _BytesIO

        lines = [f"TEMPORAL CONTEXT (last {len(frames)} frames):"]
        for frame in frames:
            try:
                from PIL import Image as _PILImage
                img = frame.image.copy()
                img.thumbnail((320, 240), _PILImage.LANCZOS)
                buf = _BytesIO()
                img.save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                relative_ts = now - frame.timestamp
                lines.append(f"  Frame -{relative_ts:.1f}s: {encoded}")
            except Exception as exc:
                logger.warning("_build_temporal_context_section: frame encode failed: %s", exc)

        lines.append(
            f"\nMOTION SUMMARY:\n"
            f"  Current frame diff: {recorder.last_frame_diff}\n"
            f"  Motion score (5s): {recorder.motion_score:.1f}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def plan_next(
        self,
        goal: str,
        screenshot: Any,
        history: List[StepRecord],
    ) -> List[CandidateAction]:
        """Perceive the current screen and ask the brain for candidate actions.

        Steps
        -----
        1. Run OCR on *screenshot*.
        2. Run ``find_element`` on *screenshot* (broad search for any visible text).
        3. Build a structured prompt.
        3a. Optionally inject temporal context from ScreenRecorder.
        4. Call ``brain.ask(prompt)`` in a thread.
        5. Parse the JSON response into ``List[CandidateAction]``.
        6. On parse failure, re-prompt with a simpler format.
        7. Return the list sorted by confidence descending.

        If ``brain`` is ``None``, returns a single fallback screenshot action.

        Requirements: 8.1, 8.2, 8.3, 6.1–6.5
        """
        if self._brain is None:
            return self._fallback_screenshot_action()

        # Step 1: OCR
        try:
            ocr_words = await self._engine.screen.ocr(screenshot)
        except Exception as exc:
            logger.warning("ActionPlanner.plan_next: OCR failed: %s", exc)
            ocr_words = []

        # Step 2: element detection (use OCR words as proxy elements)
        try:
            elements = await self._engine.vision.find_element("", screenshot)
        except Exception as exc:
            logger.warning("ActionPlanner.plan_next: find_element failed: %s", exc)
            elements = []

        # Step 3: build prompt
        prompt = self._build_prompt(goal, ocr_words, elements, history)

        # Step 3a: inject temporal context if recorder is available and running
        try:
            recorder = getattr(self._engine, "_screen_recorder", None)
            if recorder is not None and recorder.is_running:
                import time as _time
                n = self._engine._screen_recorder._config.temporal_context_frames
                recent_frames = await recorder.get_recent_frames(n)
                if recent_frames:
                    temporal_section = self._build_temporal_context_section(
                        recent_frames, recorder, _time.monotonic()
                    )
                    prompt = prompt + "\n\n" + temporal_section
        except Exception as exc:
            logger.warning("ActionPlanner.plan_next: temporal context injection failed: %s", exc)

        # Step 4: ask brain
        try:
            response = await asyncio.to_thread(self._brain.ask, prompt)
        except Exception as exc:
            logger.warning("ActionPlanner.plan_next: brain.ask failed: %s", exc)
            return self._fallback_screenshot_action()

        # Step 5: parse response
        candidates = self._parse_brain_response(response)

        # Step 6: re-prompt with simpler format on parse failure
        if not candidates:
            simple_prompt = (
                "List one action as JSON: "
                "{\"action_type\": \"...\", \"params\": {}, \"confidence\": 0.5, \"rationale\": \"...\"}"
            )
            try:
                simple_response = await asyncio.to_thread(self._brain.ask, simple_prompt)
                # Wrap in array if needed
                simple_text = simple_response.strip()
                if simple_text.startswith("{"):
                    simple_text = f"[{simple_text}]"
                candidates = self._parse_brain_response(simple_text)
            except Exception as exc:
                logger.warning("ActionPlanner.plan_next: re-prompt failed: %s", exc)

        if not candidates:
            return self._fallback_screenshot_action()

        # Step 7: sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    async def run(self, goal: str, max_steps: int = 50) -> ActionResult:
        """Run the perceive → reason → act loop until the goal is achieved or
        the step limit is reached.

        Loop body
        ---------
        1. Capture screenshot (before).
        2. Call ``plan_next`` to get candidate actions.
        3. Select the top-ranked action.
        4. Execute via ``engine.execute_action``.
        5. Capture screenshot (after).
        6. Record ``StepRecord`` in ``_history``.
        7. Check for stuck state (error message from execute_action).
        8. Check for goal achievement (success keywords in OCR).
        9. Check ``len(_history) >= max_steps``.

        Returns
        -------
        ``ActionResult(success=True, ...)``  when goal is achieved.
        ``ActionResult(success=False, error_message="Max steps reached")``
            when the step limit is exceeded.
        ``ActionResult(success=False, error_message="Stuck state ...")``
            when stuck state is detected.

        Requirements: 8.1–8.7
        """
        step_number = 0

        while True:
            step_number += 1

            # Step 1: capture screenshot before action
            try:
                screenshot_before = await self._engine.screen.capture()
                screenshot_before_b64 = self._image_to_thumbnail_b64(screenshot_before)
            except Exception as exc:
                logger.warning("ActionPlanner.run: screenshot capture failed: %s", exc)
                screenshot_before = None
                screenshot_before_b64 = ""

            # Step 2: plan next actions
            try:
                candidates = await self.plan_next(goal, screenshot_before, self._history)
            except Exception as exc:
                logger.warning("ActionPlanner.run: plan_next failed: %s", exc)
                candidates = self._fallback_screenshot_action()

            # Step 3: select top action
            top_candidate = candidates[0] if candidates else CandidateAction(
                action=Action(action_type=ActionType.SCREENSHOT, params={}),
                confidence=0.0,
                rationale="No candidates returned.",
            )
            action = top_candidate.action

            # Step 4: execute action
            result = await self._engine.execute_action(action)

            # Step 5: capture screenshot after action
            try:
                screenshot_after = await self._engine.screen.capture()
                screenshot_after_b64 = self._image_to_thumbnail_b64(screenshot_after)
            except Exception as exc:
                logger.warning("ActionPlanner.run: post-action screenshot failed: %s", exc)
                screenshot_after = None
                screenshot_after_b64 = ""

            # Step 6: record StepRecord
            record = StepRecord(
                step_number=step_number,
                action=action,
                result=result,
                screenshot_before=screenshot_before_b64,
                screenshot_after=screenshot_after_b64,
            )
            self._history.append(record)

            # Step 7: check for stuck state
            # execute_action already calls _check_stuck_state internally; we
            # detect it by inspecting the error message on the result.
            if (
                not result.success
                and result.error_message is not None
                and "stuck state" in result.error_message.lower()
            ):
                return ActionResult(
                    success=False,
                    action_type=ActionType.SCREENSHOT,
                    error_message=result.error_message,
                    data={"steps": len(self._history)},
                )

            # Step 8: check for goal achievement via OCR on the after-screenshot
            if screenshot_after is not None:
                try:
                    ocr_words = await self._engine.screen.ocr(screenshot_after)
                    ocr_text_lower = " ".join(w.text for w in ocr_words).lower()
                    if any(kw in ocr_text_lower for kw in self._SUCCESS_KEYWORDS):
                        summary = ocr_text_lower[:200]
                        return ActionResult(
                            success=True,
                            action_type=ActionType.SCREENSHOT,
                            data={
                                "summary": summary,
                                "steps": len(self._history),
                            },
                        )
                except Exception as exc:
                    logger.warning(
                        "ActionPlanner.run: goal-check OCR failed: %s", exc
                    )

            # Step 9: check step limit
            if len(self._history) >= max_steps:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SCREENSHOT,
                    error_message="Max steps reached",
                    data={"steps": len(self._history)},
                )

# ---------------------------------------------------------------------------
# BrowserComputerUse
# ---------------------------------------------------------------------------

class BrowserComputerUse:
    """High-level browser helper that drives a browser via the ComputerUseEngine.

    Provides human-like browser interactions: opening URLs, logging in,
    searching for jobs, applying to jobs, and submitting work.

    After each navigation action the address bar is read via OCR and the
    current domain is compared against the expected domain.  A mismatch
    returns a failed ``ActionResult`` immediately.

    Requirements: 10.3
    """

    def __init__(self, engine: "ComputerUseEngine", browser: str = "chrome") -> None:
        self._engine = engine
        self._browser = browser
        self._expected_domain: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Return the netloc (domain) portion of *url*.

        Examples::

            >>> BrowserComputerUse._extract_domain("https://www.example.com/path")
            'www.example.com'
        """
        from urllib.parse import urlparse
        return urlparse(url).netloc

    async def _check_domain(self) -> Optional["ActionResult"]:
        """Capture a screenshot, run OCR, and verify the expected domain is visible.

        Returns a failed ``ActionResult`` if the expected domain is not found
        in the OCR text, or ``None`` if the check passes (or no expected domain
        is set).
        """
        if not self._expected_domain:
            return None

        try:
            screenshot = await self._engine.screen.capture()
            words = await self._engine.screen.ocr(screenshot)
            ocr_text = " ".join(w.text for w in words).lower()
            if self._expected_domain.lower() not in ocr_text:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SCREENSHOT,
                    error_message=(
                        f"Domain mismatch: expected '{self._expected_domain}' "
                        f"but it was not found in the address bar OCR text."
                    ),
                )
        except Exception as exc:
            logger.warning("BrowserComputerUse._check_domain failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def open(self, url: str) -> "ActionResult":
        """Navigate the browser to *url*.

        Steps:
        1. Extract and store the expected domain from *url*.
        2. Focus the address bar with Ctrl+L.
        3. Type the URL and press Enter.
        4. Wait for the page to load (TEXT_PRESENT with short timeout).
        5. Verify the domain via OCR; return failed ``ActionResult`` on mismatch.

        Requirements: 10.3
        """
        self._expected_domain = self._extract_domain(url)

        # Focus address bar
        hotkey_result = await self._engine.keyboard.hotkey("ctrl", "l")
        if not hotkey_result.success:
            return hotkey_result

        # Type URL
        type_result = await self._engine.keyboard.type_text(url)
        if not type_result.success:
            return type_result

        # Press Enter to navigate
        enter_result = await self._engine.keyboard.press_key("enter")
        if not enter_result.success:
            return enter_result

        # Wait for page load — use a short 5-second timeout
        wait_condition = WaitCondition(
            condition_type=WaitConditionType.TEXT_PRESENT,
            target="",
            timeout_seconds=5.0,
        )
        try:
            await self._engine.wait_for(wait_condition)
        except Exception as exc:
            logger.warning("BrowserComputerUse.open: wait_for failed: %s", exc)

        # Domain mismatch check — log warning but don't fail (OCR may miss address bar)
        mismatch = await self._check_domain()
        if mismatch is not None:
            logger.warning("BrowserComputerUse.open: %s (continuing anyway)", mismatch.error_message)
            return mismatch

        return ActionResult(success=True, action_type=ActionType.HOTKEY)

    async def login(self, username: str, password: str) -> "ActionResult":
        """Fill in and submit a login form.

        Tries multiple common field labels for the username field
        (``"username"``, ``"email"``, ``"user"``, ``"login"``) and for the
        password field (``"password"``, ``"pass"``).  After filling both
        fields the submit button is located and clicked.

        Requirements: 10.3
        """
        try:
            screenshot = await self._engine.screen.capture()
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message=f"Failed to capture screenshot: {exc}",
            )

        # --- Username field ---
        username_element = None
        for label in ("username", "email", "user", "login"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                username_element = elements[0]
                break

        if username_element is None:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message="Could not find username/email field on the page.",
            )

        click_result = await self._engine.mouse.click(*username_element.center)
        if not click_result.success:
            return click_result
        type_result = await self._engine.keyboard.type_text(username)
        if not type_result.success:
            return type_result

        # --- Password field ---
        password_element = None
        for label in ("password", "pass"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                password_element = elements[0]
                break

        if password_element is None:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message="Could not find password field on the page.",
            )

        click_result = await self._engine.mouse.click(*password_element.center)
        if not click_result.success:
            return click_result
        type_result = await self._engine.keyboard.type_text(password)
        if not type_result.success:
            return type_result

        # --- Submit button ---
        submit_element = None
        for label in ("submit", "login", "sign in", "log in"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                submit_element = elements[0]
                break

        if submit_element is None:
            # Fallback: press Enter
            return await self._engine.keyboard.press_key("enter")

        return await self._engine.mouse.click(*submit_element.center)

    async def search_jobs(self, query: str) -> List[Dict[str, Any]]:
        """Search for jobs matching *query* and return a list of result dicts.

        Each dict has keys ``"title"``, ``"company"``, and ``"url"`` parsed
        from the OCR text of the search results page.

        Requirements: 10.3
        """
        # Type the query into the current search field and submit
        type_result = await self._engine.keyboard.type_text(query)
        if not type_result.success:
            logger.warning("BrowserComputerUse.search_jobs: failed to type query: %s", type_result.error_message)
            return []

        await self._engine.keyboard.press_key("enter")

        # Wait briefly for results to load
        wait_condition = WaitCondition(
            condition_type=WaitConditionType.TEXT_PRESENT,
            target="",
            timeout_seconds=5.0,
        )
        try:
            await self._engine.wait_for(wait_condition)
        except Exception as exc:
            logger.warning("BrowserComputerUse.search_jobs: wait_for failed: %s", exc)

        # Capture and OCR the results page
        try:
            screenshot = await self._engine.screen.capture()
            words = await self._engine.screen.ocr(screenshot)
        except Exception as exc:
            logger.warning("BrowserComputerUse.search_jobs: OCR failed: %s", exc)
            return []

        # Parse OCR text into job result dicts heuristically
        # Group words into lines by their vertical position
        lines: Dict[int, List[str]] = {}
        for word in words:
            row_key = word.bounding_box.y // 20  # bucket by ~20px rows
            lines.setdefault(row_key, []).append(word.text)

        results: List[Dict[str, Any]] = []
        line_texts = [" ".join(tokens) for tokens in lines.values()]

        for i, line in enumerate(line_texts):
            # Heuristic: lines that look like job titles (non-trivial length)
            if len(line.strip()) > 5:
                title = line.strip()
                company = line_texts[i + 1].strip() if i + 1 < len(line_texts) else ""
                results.append({
                    "title": title,
                    "company": company,
                    "url": "",
                })
            if len(results) >= 10:
                break

        return results

    async def apply_to_job(self, job_url: str, cover_letter: str) -> "ActionResult":
        """Open *job_url* and submit an application with *cover_letter*.

        Requirements: 10.3
        """
        open_result = await self.open(job_url)
        if not open_result.success:
            return open_result

        try:
            screenshot = await self._engine.screen.capture()
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message=f"Failed to capture screenshot: {exc}",
            )

        # Find cover letter / application text area
        cover_letter_element = None
        for label in ("cover letter", "cover_letter", "message", "application", "textarea"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                cover_letter_element = elements[0]
                break

        if cover_letter_element is None:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message="Could not find cover letter / application text area.",
            )

        click_result = await self._engine.mouse.click(*cover_letter_element.center)
        if not click_result.success:
            return click_result

        type_result = await self._engine.keyboard.type_text(cover_letter)
        if not type_result.success:
            return type_result

        # Find and click submit button
        submit_element = None
        for label in ("submit", "apply", "send application"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                submit_element = elements[0]
                break

        if submit_element is None:
            return await self._engine.keyboard.press_key("enter")

        return await self._engine.mouse.click(*submit_element.center)

    async def submit_work(self, submission_url: str, content: str) -> "ActionResult":
        """Open *submission_url* and submit *content* via the submission form.

        Requirements: 10.3
        """
        open_result = await self.open(submission_url)
        if not open_result.success:
            return open_result

        try:
            screenshot = await self._engine.screen.capture()
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message=f"Failed to capture screenshot: {exc}",
            )

        # Find submission text area
        submission_element = None
        for label in ("submission", "content", "work", "textarea", "message", "description"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                submission_element = elements[0]
                break

        if submission_element is None:
            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                error_message="Could not find submission text area.",
            )

        click_result = await self._engine.mouse.click(*submission_element.center)
        if not click_result.success:
            return click_result

        type_result = await self._engine.keyboard.type_text(content)
        if not type_result.success:
            return type_result

        # Find and click submit button
        submit_element = None
        for label in ("submit", "send", "upload", "deliver"):
            elements = await self._engine.vision.find_element(label, screenshot)
            if elements:
                submit_element = elements[0]
                break

        if submit_element is None:
            return await self._engine.keyboard.press_key("enter")

        return await self._engine.mouse.click(*submit_element.center)

# ---------------------------------------------------------------------------
# Module-level dependency check — runs on import
# ---------------------------------------------------------------------------
_check_dependencies()
