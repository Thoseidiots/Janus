"""
Synthetic Image Generator — produces procedural images from text prompts.
No external HTTP calls are made. Uses NumPy + Pillow for all pixel operations.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ImageRequest:
    prompt: str
    aspect_ratio: Literal['1:1', '16:9', '9:16'] = '1:1'
    seed: int = 0
    reference_palette: Optional[np.ndarray] = None  # shape (N, 3) HSL


@dataclass
class ImageResult:
    data_uri: str          # "data:image/png;base64,..."
    width: int
    height: int
    seed: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ASPECT_RATIO_DIMS: dict[str, tuple[int, int]] = {
    '1:1':  (512, 512),
    '16:9': (512, 288),
    '9:16': (288, 512),
}

# Visual keyword → (hue_deg, saturation, value) hints for HSV-based generation
_KEYWORD_PALETTE: dict[str, tuple[float, float, float]] = {
    # colours
    'red':     (0.0,   0.85, 0.85),
    'orange':  (30.0,  0.85, 0.90),
    'yellow':  (55.0,  0.85, 0.95),
    'green':   (120.0, 0.70, 0.60),
    'blue':    (210.0, 0.80, 0.75),
    'purple':  (270.0, 0.70, 0.65),
    'pink':    (330.0, 0.60, 0.90),
    'white':   (0.0,   0.00, 1.00),
    'black':   (0.0,   0.00, 0.05),
    'grey':    (0.0,   0.00, 0.50),
    'gray':    (0.0,   0.00, 0.50),
    'brown':   (25.0,  0.60, 0.40),
    'cyan':    (185.0, 0.80, 0.80),
    'teal':    (175.0, 0.75, 0.55),
    # scenes / subjects
    'sunset':  (20.0,  0.90, 0.85),
    'sunrise': (35.0,  0.85, 0.90),
    'sky':     (205.0, 0.65, 0.85),
    'ocean':   (200.0, 0.75, 0.60),
    'sea':     (195.0, 0.70, 0.55),
    'forest':  (115.0, 0.65, 0.35),
    'tree':    (110.0, 0.60, 0.40),
    'fire':    (15.0,  0.95, 0.90),
    'snow':    (210.0, 0.10, 0.95),
    'night':   (240.0, 0.40, 0.15),
    'desert':  (35.0,  0.55, 0.80),
    'mountain':(200.0, 0.30, 0.55),
    'city':    (220.0, 0.25, 0.45),
    'space':   (240.0, 0.50, 0.10),
    'galaxy':  (260.0, 0.60, 0.20),
    'abstract':(270.0, 0.50, 0.50),
    'person':  (25.0,  0.35, 0.70),
    'portrait':(20.0,  0.30, 0.72),
    'flower':  (300.0, 0.70, 0.80),
    'water':   (200.0, 0.65, 0.70),
    'gold':    (45.0,  0.90, 0.85),
    'silver':  (210.0, 0.10, 0.75),
    'dark':    (240.0, 0.30, 0.20),
    'bright':  (55.0,  0.50, 0.95),
    'neon':    (150.0, 1.00, 1.00),
    'pastel':  (180.0, 0.30, 0.90),
    'vintage': (35.0,  0.40, 0.65),
    'retro':   (30.0,  0.50, 0.70),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_keywords(prompt: str) -> list[str]:
    """Return visual keywords found in *prompt* (lower-cased, order-preserved)."""
    words = prompt.lower().split()
    found: list[str] = []
    seen: set[str] = set()
    for word in words:
        # strip punctuation
        clean = word.strip('.,!?;:\'"()[]{}')
        if clean in _KEYWORD_PALETTE and clean not in seen:
            found.append(clean)
            seen.add(clean)
    return found


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (h in [0,360], s/v in [0,1]) to RGB in [0,1]."""
    h = h % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


def _array_to_png_data_uri(arr: np.ndarray) -> str:
    """Convert a uint8 (H, W, 3) array to a PNG base64 data URI."""
    img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _gradient_field(
    rng: np.random.Generator,
    h: int,
    w: int,
    color1: tuple[float, float, float],
    color2: tuple[float, float, float],
    angle_deg: float,
) -> np.ndarray:
    """Linear gradient from color1 to color2 at *angle_deg*."""
    angle_rad = math.radians(angle_deg)
    xs = np.linspace(0.0, 1.0, w)
    ys = np.linspace(0.0, 1.0, h)
    xx, yy = np.meshgrid(xs, ys)
    t = (xx * math.cos(angle_rad) + yy * math.sin(angle_rad))
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    t = t[:, :, np.newaxis]
    c1 = np.array(color1, dtype=np.float32)
    c2 = np.array(color2, dtype=np.float32)
    canvas = (1.0 - t) * c1 + t * c2
    return np.clip(canvas, 0.0, 1.0)


def _noise_layer(rng: np.random.Generator, h: int, w: int, scale: float = 0.15) -> np.ndarray:
    """Simple value-noise layer in [0, scale]."""
    # Low-res noise upsampled for a smooth look
    small_h = max(4, h // 8)
    small_w = max(4, w // 8)
    noise_small = rng.random((small_h, small_w)).astype(np.float32)
    img_small = Image.fromarray((noise_small * 255).astype(np.uint8), mode='L')
    img_large = img_small.resize((w, h), Image.BILINEAR)
    noise = np.array(img_large, dtype=np.float32) / 255.0 * scale
    return noise[:, :, np.newaxis]


def _draw_shapes(
    rng: np.random.Generator,
    canvas: np.ndarray,
    color: tuple[float, float, float],
    n_shapes: int = 5,
) -> np.ndarray:
    """Overlay semi-transparent geometric primitives onto *canvas*."""
    h, w = canvas.shape[:2]
    result = canvas.copy()
    for _ in range(n_shapes):
        shape_type = rng.integers(0, 3)  # 0=circle, 1=rect, 2=triangle
        alpha = float(rng.uniform(0.10, 0.35))
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        size = int(rng.integers(max(1, min(w, h) // 10), max(2, min(w, h) // 3)))

        # Build a mask
        mask = np.zeros((h, w), dtype=np.float32)
        if shape_type == 0:
            # Circle
            ys, xs = np.ogrid[:h, :w]
            dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            mask[dist <= size] = 1.0
        elif shape_type == 1:
            # Rectangle
            x0, x1 = max(0, cx - size), min(w, cx + size)
            y0, y1 = max(0, cy - size // 2), min(h, cy + size // 2)
            mask[y0:y1, x0:x1] = 1.0
        else:
            # Triangle (rasterised via PIL)
            tmp = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(tmp)
            pts = [
                (cx, cy - size),
                (cx - size, cy + size),
                (cx + size, cy + size),
            ]
            draw.polygon(pts, fill=255)
            mask = np.array(tmp, dtype=np.float32) / 255.0

        c = np.array(color, dtype=np.float32)
        result += mask[:, :, np.newaxis] * alpha * (c - result)

    return np.clip(result, 0.0, 1.0)


def _placeholder_image(prompt: str, w: int, h: int) -> np.ndarray:
    """Grey gradient with prompt text overlay — used when no keywords found."""
    # Grey gradient top-to-bottom
    top = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    bot = np.array([0.40, 0.40, 0.40], dtype=np.float32)
    t = np.linspace(0.0, 1.0, h)[:, np.newaxis, np.newaxis]
    canvas = ((1.0 - t) * top + t * bot)
    canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas, mode='RGB')
    draw = ImageDraw.Draw(img)

    # Wrap text to fit width
    max_chars = max(1, w // 8)
    words = prompt.split()
    lines: list[str] = []
    current = ''
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + ' ' + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_h = max(12, h // (len(lines) + 2))
    y_start = (h - line_h * len(lines)) // 2
    for i, line in enumerate(lines):
        # Estimate text width (approx 7px per char at default font)
        text_w = len(line) * 7
        x = max(4, (w - text_w) // 2)
        y = y_start + i * line_h
        draw.text((x + 1, y + 1), line, fill=(30, 30, 30))
        draw.text((x, y), line, fill=(220, 220, 220))

    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SyntheticImageGenerator:
    """Generates procedural images from text prompts without any HTTP calls."""

    _VALID_RATIOS = frozenset(_ASPECT_RATIO_DIMS.keys())

    def generate(self, request: ImageRequest) -> ImageResult:
        """Generate a synthetic image for *request*.

        Raises:
            ValueError: if *request.aspect_ratio* is not one of the valid values.
        """
        if request.aspect_ratio not in self._VALID_RATIOS:
            valid = ', '.join(sorted(self._VALID_RATIOS))
            raise ValueError(
                f"Invalid aspect_ratio {request.aspect_ratio!r}. "
                f"Valid values are: {valid}"
            )

        w, h = _ASPECT_RATIO_DIMS[request.aspect_ratio]
        rng = np.random.default_rng(request.seed)

        keywords = _extract_keywords(request.prompt)

        if not keywords:
            canvas_u8 = _placeholder_image(request.prompt, w, h)
        else:
            canvas_u8 = self._generate_from_keywords(rng, keywords, w, h)

        data_uri = _array_to_png_data_uri(canvas_u8)
        return ImageResult(data_uri=data_uri, width=w, height=h, seed=request.seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_from_keywords(
        self,
        rng: np.random.Generator,
        keywords: list[str],
        w: int,
        h: int,
    ) -> np.ndarray:
        """Build a procedural image driven by *keywords*."""
        # Blend palette hints from all matched keywords
        palettes = [_KEYWORD_PALETTE[kw] for kw in keywords]
        avg_h = sum(p[0] for p in palettes) / len(palettes)
        avg_s = sum(p[1] for p in palettes) / len(palettes)
        avg_v = sum(p[2] for p in palettes) / len(palettes)

        # Primary and secondary colours
        color1 = _hsv_to_rgb(avg_h, avg_s, avg_v)
        color2 = _hsv_to_rgb(
            (avg_h + float(rng.uniform(30, 90))) % 360.0,
            max(0.0, avg_s - 0.2),
            min(1.0, avg_v + 0.15),
        )

        angle = float(rng.uniform(0, 180))
        canvas = _gradient_field(rng, h, w, color1, color2, angle)

        # Add noise
        noise = _noise_layer(rng, h, w, scale=0.12)
        canvas = np.clip(canvas + noise, 0.0, 1.0)

        # Add shape primitives
        accent_h = (avg_h + float(rng.uniform(120, 240))) % 360.0
        accent_color = _hsv_to_rgb(accent_h, min(1.0, avg_s + 0.1), min(1.0, avg_v + 0.05))
        n_shapes = int(rng.integers(3, 9))
        canvas = _draw_shapes(rng, canvas, accent_color, n_shapes)

        return np.clip(canvas * 255, 0, 255).astype(np.uint8)
