"""
screen_interpreter.py
======================
Converts raw screen pixels from JanusOS.capture_screen() into
text descriptions that AvusInference can understand and act on.

This is the vision layer of Janus. It bridges the gap between
what the OS sees (raw pixels) and what Avus needs (text descriptions).

Pipeline:
    raw pixels (bytes)
        ↓
    numpy array (BGRA → RGB)
        ↓
    region analysis (buttons, text, layout)
        ↓
    text description
        ↓
    AvusInference.generate_action()

Usage:
    from screen_interpreter import ScreenInterpreter
    from avus_inference import AvusInference

    interp = ScreenInterpreter()
    avus   = AvusInference()
    avus.load()

    # From raw JanusOS pixel bytes
    raw_bytes, w, h = janus_os.capture_screen()
    description     = interp.interpret(raw_bytes, w, h)
    action          = avus.generate_action(description)

    # Or interpret a saved screenshot file
    description = interp.interpret_file("screenshot.png")
    action      = avus.generate_action(description)
"""

import os
import platform
from typing import Optional, Tuple, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ScreenRegion
# ─────────────────────────────────────────────────────────────────────────────

class ScreenRegion:
    def __init__(self, x, y, w, h, region_type, label="", confidence=1.0):
        self.x            = x
        self.y            = y
        self.w            = w
        self.h            = h
        self.region_type  = region_type
        self.label        = label
        self.confidence   = confidence

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def to_description(self):
        cx, cy = self.center
        if self.label:
            return (f"A {self.region_type} labeled '{self.label}' "
                    f"at approximately ({cx}, {cy})")
        return f"A {self.region_type} at approximately ({cx}, {cy})"


# ─────────────────────────────────────────────────────────────────────────────
# ScreenInterpreter
# ─────────────────────────────────────────────────────────────────────────────

class ScreenInterpreter:
    """
    Converts screen pixels to text descriptions for Avus.

    On Windows with JanusOS: full pixel analysis pipeline.
    On Linux/Mac or without JanusOS: file-based or mock mode.

    The description format matches the ScreenActionDataset training format
    so Avus recognises it and can generate correct actions.
    """

    TITLE_BAR_HEIGHT = 32
    MIN_REGION_W     = 20
    MIN_REGION_H     = 12

    def __init__(self):
        self._ocr_available = self._check_ocr()
        self._cv2_available = self._check_cv2()

    # ── public API ────────────────────────────────────────────────────────────

    def interpret(self, raw_bytes: bytes, width: int, height: int,
                  goal: Optional[str] = None) -> str:
        """
        Convert raw BGRA pixel bytes from JanusOS to a text description.

        Parameters
        ----------
        raw_bytes : bytes
            Raw BGRA pixel data from JanusOS.capture_screen()
        width, height : int
            Screen dimensions in pixels
        goal : str, optional
            Current task goal — helps focus the description

        Returns
        -------
        str  — description in ScreenActionDataset format
        """
        try:
            arr = self._bytes_to_array(raw_bytes, width, height)
            return self._analyse(arr, width, height, goal)
        except Exception as e:
            return self._fallback_description(width, height, str(e))

    def interpret_file(self, path: str,
                       goal: Optional[str] = None) -> str:
        """Interpret a saved screenshot file (.png, .jpg, .bmp)."""
        try:
            arr    = self._load_image(path)
            h, w   = arr.shape[:2]
            return self._analyse(arr, w, h, goal)
        except Exception as e:
            return f"Screen: image file could not be analysed ({e})."

    def interpret_array(self, arr: np.ndarray,
                        goal: Optional[str] = None) -> str:
        """Interpret a numpy RGB array directly."""
        h, w = arr.shape[:2]
        return self._analyse(arr, w, h, goal)

    def mock_description(self, app: str, elements: List[str],
                         goal: Optional[str] = None) -> str:
        """
        Generate a mock screen description for testing without a real screen.
        Useful on Linux/Kaggle where JanusOS is not available.
        """
        desc = f"{app} is open. " + " ".join(elements)
        if goal:
            desc += f" Current task: {goal}."
        return desc

    def describe_with_ocr(self, raw_bytes: bytes, width: int,
                          height: int, goal: Optional[str] = None) -> str:
        """
        Full description including OCR text extraction.
        Requires pytesseract. Falls back to normal interpret() if unavailable.
        """
        base = self.interpret(raw_bytes, width, height, goal)
        arr  = self._bytes_to_array(raw_bytes, width, height)
        text = self.extract_text(arr)
        if text:
            snippet = text[:200].replace("\n", " ").strip()
            return f"{base} Visible text: \"{snippet}\"."
        return base

    def extract_text(self, arr: np.ndarray) -> str:
        """Extract visible text via OCR. Returns '' if unavailable."""
        if not self._ocr_available:
            return ""
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.fromarray(arr)).strip()
        except Exception:
            return ""

    # ── pixel analysis ────────────────────────────────────────────────────────

    def _bytes_to_array(self, raw: bytes, w: int, h: int) -> np.ndarray:
        """JanusOS BGRA bytes → RGB numpy array."""
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, [2, 1, 0]].copy()   # BGRA → RGB

    def _load_image(self, path: str) -> np.ndarray:
        try:
            import imageio
        except ImportError:
            os.system("pip install imageio -q")
            import imageio
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _analyse(self, arr: np.ndarray, w: int, h: int,
                 goal: Optional[str]) -> str:
        parts = []
        parts.append(f"{self._detect_app(arr, w, h)} is open.")
        parts.append(f"Screen resolution: {w}x{h}.")

        regions = self._detect_regions(arr, w, h)
        for region in regions[:5]:
            parts.append(region.to_description() + ".")

        zone = self._dominant_zone(arr)
        if zone:
            parts.append(zone)

        if goal:
            parts.append(f"Current task: {goal}.")

        return " ".join(parts)

    def _detect_app(self, arr: np.ndarray, w: int, h: int) -> str:
        if h < self.TITLE_BAR_HEIGHT:
            return "An application"
        bar   = arr[:self.TITLE_BAR_HEIGHT, :, :]
        r, g, b = float(np.mean(bar[:,:,0])), float(np.mean(bar[:,:,1])), float(np.mean(bar[:,:,2]))
        if b > r + 30 and b > g + 20:      return "A browser or blue-themed application"
        if r > 200 and g < 80 and b < 80:  return "An application with a red title bar"
        if r > 180 and g > 180 and b > 180: return "A light-themed application"
        if r < 60 and g < 60 and b < 60:   return "A dark-themed application or terminal"
        if g > r + 20:                      return "A green-themed application"
        return "An application"

    def _detect_regions(self, arr: np.ndarray, w: int,
                        h: int) -> List[ScreenRegion]:
        if self._cv2_available:
            return self._detect_regions_cv2(arr, w, h)
        return self._detect_regions_simple(arr, w, h)

    def _detect_regions_simple(self, arr: np.ndarray, w: int,
                                h: int) -> List[ScreenRegion]:
        regions = []
        gx, gy  = 8, 6
        pw, ph  = max(w // gx, 1), max(h // gy, 1)
        seen    = set()

        for row in range(gy):
            for col in range(gx):
                x0, y0 = col * pw, row * ph
                x1, y1 = min(x0 + pw, w), min(y0 + ph, h)
                patch  = arr[y0:y1, x0:x1, :]
                if patch.size == 0:
                    continue
                r = float(np.mean(patch[:,:,0]))
                g = float(np.mean(patch[:,:,1]))
                b = float(np.mean(patch[:,:,2]))
                rtype, label = self._classify_patch(r, g, b, x0, y0, w, h)
                if rtype and (col, row) not in seen:
                    seen.add((col, row))
                    regions.append(ScreenRegion(x0, y0, pw, ph, rtype, label))
        return regions[:8]

    def _classify_patch(self, r, g, b, x, y, w, h):
        if r > 210 and g > 210 and b > 210:
            return ("toolbar area", "") if y < h * 0.1 else ("light UI element", "")
        if b > r + 40 and b > 150:
            return "button", "blue"
        if r < 60 and g < 60 and b < 60 and y > h * 0.1:
            return "dark panel", ""
        return None, ""

    def _detect_regions_cv2(self, arr: np.ndarray, w: int,
                             h: int) -> List[ScreenRegion]:
        try:
            import cv2
            grey  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grey, 50, 150)
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for cnt in cnts:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw < self.MIN_REGION_W or bh < self.MIN_REGION_H:
                    continue
                aspect = bw / max(bh, 1)
                rtype  = "button" if (2 < aspect < 8 and bh < 60) else "UI element"
                regions.append(ScreenRegion(bx, by, bw, bh, rtype))
            regions.sort(key=lambda r: r.w * r.h, reverse=True)
            return regions[:8]
        except Exception:
            return self._detect_regions_simple(arr, w, h)

    def _dominant_zone(self, arr: np.ndarray) -> str:
        r = float(np.mean(arr[:,:,0]))
        g = float(np.mean(arr[:,:,1]))
        b = float(np.mean(arr[:,:,2]))
        brightness = (r + g + b) / 3
        if brightness > 200:    return "The screen is predominantly light."
        if brightness < 50:     return "The screen is predominantly dark."
        if b > r + 30:          return "The screen has a blue colour scheme."
        if g > r + 20:          return "The screen has a green colour scheme."
        return ""

    def _fallback_description(self, w, h, error):
        return (f"A screen of resolution {w}x{h} is visible. "
                f"Unable to fully analyse pixel content ({error}).")

    def _check_ocr(self):
        try:
            import pytesseract
            return True
        except ImportError:
            return False

    def _check_cv2(self):
        try:
            import cv2
            return True
        except ImportError:
            return False

    @property
    def capabilities(self):
        return {"ocr": self._ocr_available, "cv2": self._cv2_available,
                "platform": platform.system()}


# ─────────────────────────────────────────────────────────────────────────────
# Shared instance
# ─────────────────────────────────────────────────────────────────────────────

_interp = None

def get_interpreter() -> ScreenInterpreter:
    global _interp
    if _interp is None:
        _interp = ScreenInterpreter()
    return _interp

def describe_screen(raw_bytes: bytes, width: int, height: int,
                    goal: Optional[str] = None) -> str:
    return get_interpreter().interpret(raw_bytes, width, height, goal)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    interp = ScreenInterpreter()
    print(f"Capabilities: {interp.capabilities}\n")

    print("── Mock description ────────────────────────────────────")
    print(interp.mock_description(
        "Chrome",
        ["A 'Login' button is at (640, 400).",
         "A username field is at (640, 300)."],
        goal="Log into the website"
    ))

    print("\n── Synthetic screen ────────────────────────────────────")
    fake = np.ones((768, 1366, 3), dtype=np.uint8) * 240
    fake[:32, :, :]          = [66, 133, 244]   # blue title bar
    fake[300:340, 580:780, :] = [66, 133, 244]  # blue button
    print(interp.interpret_array(fake, goal="Click the login button"))

    print("\n── Raw bytes test ──────────────────────────────────────")
    w, h = 400, 300
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[:, :, :3] = 200
    bgra[:32, :, 0] = 50
    bgra[:32, :, 1] = 100
    bgra[:32, :, 2] = 200
    bgra[:, :, 3]  = 255
    print(interp.interpret(bgra.tobytes(), w, h, goal="Find the close button"))

    print("\nScreenInterpreter ready.")
    print("Connect: avus.generate_action(interp.interpret(raw, w, h))")
