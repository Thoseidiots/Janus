# updated_enhanced_vision.py

“””
Full replacement for enhanced_vision.py.
Adds temporal video analysis methods to EnhancedVisionProcessor while
keeping all original single-frame functionality intact.
All processing is pure Python + ctypes — no third-party imports.
“””

import ctypes
import ctypes.wintypes as wt
import math
import time
import hashlib
import collections
from typing import List, Tuple, Optional, Dict, Any

from video_observer import VideoFrame, _compute_motion, _scan_subtitle_region

# ── Colour / pixel utilities ──────────────────────────────────────────────────

def _lum(b: int, g: int, r: int) -> int:
return (11*b + 59*g + 30*r) // 100

def _region_avg_lum(frame: VideoFrame, x: int, y: int, w: int, h: int) -> float:
total = 0
count = 0
row_stride = frame.width * 4
for row in range(y, min(y + h, frame.height)):
off_base = row * row_stride + x * 4
for col in range(0, w, 4):
off = off_base + col * 4
if off + 2 >= len(frame.data):
continue
total += _lum(frame.data[off], frame.data[off+1], frame.data[off+2])
count += 1
return total / count if count else 0.0

def _histogram(frame: VideoFrame, bins: int = 16) -> List[float]:
“”“Compute normalised luminance histogram over the whole frame.”””
counts = [0] * bins
bin_w  = 256 // bins
data   = frame.data
i      = 0
n      = 0
while i < len(data) - 3:
lum = _lum(data[i], data[i+1], data[i+2])
counts[min(lum // bin_w, bins - 1)] += 1
n  += 1
i  += 16   # sample every 4th pixel (stride 16 bytes in BGRA)
if n == 0:
return counts
return [c / n for c in counts]

# ── EnhancedVisionProcessor ───────────────────────────────────────────────────

class EnhancedVisionProcessor:
“””
Processes VideoFrame objects (and sequences) to extract perceptual
features useful for Janus learning.

```
Original single-frame methods:
    analyse_frame(frame) → Dict
    detect_dominant_region(frame) → str

New temporal / video methods:
    process_video_frame_sequence(frames) → Dict
    detect_motion(prev, curr) → float
    detect_text_changes(prev, curr) → bool
    summarize_scene_delta(frames) → str
"""

def __init__(self):
    self._scene_history: collections.deque = collections.deque(maxlen=200)

# ── Single-frame analysis ─────────────────────────────────────────────────

def analyse_frame(self, frame: VideoFrame) -> Dict[str, Any]:
    """Return perceptual features for a single frame."""
    hist          = _histogram(frame)
    avg_lum       = sum(h * i * (256 // len(hist)) for i, h in enumerate(hist))
    dark_ratio    = sum(hist[:4])
    bright_ratio  = sum(hist[-4:])
    mid_ratio     = 1.0 - dark_ratio - bright_ratio

    top_lum    = _region_avg_lum(frame, 0, 0,             frame.width, frame.height // 3)
    bottom_lum = _region_avg_lum(frame, 0, frame.height * 2 // 3, frame.width, frame.height // 3)
    sub_signal = _scan_subtitle_region(frame)

    return {
        "avg_luminance":  avg_lum,
        "dark_ratio":     dark_ratio,
        "bright_ratio":   bright_ratio,
        "mid_ratio":      mid_ratio,
        "top_lum":        top_lum,
        "bottom_lum":     bottom_lum,
        "subtitle_signal": bool(sub_signal),
        "scene_hash":     frame.scene_hash,
        "timestamp":      frame.timestamp,
        "resolution":     (frame.width, frame.height),
    }

def detect_dominant_region(self, frame: VideoFrame) -> str:
    """Classify which vertical region of frame has the most activity."""
    h = frame.height
    top    = _region_avg_lum(frame, 0, 0,         frame.width, h // 3)
    middle = _region_avg_lum(frame, 0, h // 3,    frame.width, h // 3)
    bottom = _region_avg_lum(frame, 0, h * 2 // 3, frame.width, h // 3)
    regions = {"top": top, "middle": middle, "bottom": bottom}
    return max(regions, key=lambda k: regions[k])

# ── Temporal / sequence methods ───────────────────────────────────────────

def detect_motion(self, prev: VideoFrame, curr: VideoFrame) -> float:
    """
    Wrapper around video_observer motion diff.
    Returns 0.0–1.0 motion score.
    """
    return _compute_motion(prev, curr)

def detect_text_changes(self, prev: VideoFrame, curr: VideoFrame) -> bool:
    """
    Returns True if the subtitle region changed significantly between frames.
    Uses luminance diff in bottom 15% only.
    """
    w, h    = prev.width, prev.height
    sub_y   = int(h * 0.85)
    stride  = frame_stride = w * 4
    p_data  = memoryview(prev.data)
    c_data  = memoryview(curr.data)
    total   = 0
    n       = 0
    for row in range(sub_y, h):
        off_base = row * frame_stride
        for col in range(0, w, 8):
            off = off_base + col * 4
            if off + 2 >= len(prev.data):
                continue
            pl = _lum(p_data[off], p_data[off+1], p_data[off+2])
            cl = _lum(c_data[off], c_data[off+1], c_data[off+2])
            diff = pl - cl
            total += diff if diff >= 0 else -diff
            n += 1
    if n == 0:
        return False
    return (total / n) > 15   # threshold: average subtitle pixel change > 15 lum units

def process_video_frame_sequence(self, frames: List[VideoFrame]) -> Dict[str, Any]:
    """
    Analyse a list of frames (e.g. last 60 from buffer).
    Returns rich temporal feature dict for use by goal_planner / world_model.
    """
    if not frames:
        return {"error": "no_frames"}

    analyses    = [self.analyse_frame(f) for f in frames]
    motion_scores = []
    text_changes  = 0
    for i in range(1, len(frames)):
        ms = self.detect_motion(frames[i-1], frames[i])
        motion_scores.append(ms)
        if self.detect_text_changes(frames[i-1], frames[i]):
            text_changes += 1

    duration      = frames[-1].timestamp - frames[0].timestamp
    avg_motion    = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    peak_motion   = max(motion_scores) if motion_scores else 0.0

    scene_hashes  = [a["scene_hash"] for a in analyses]
    unique_scenes = len(set(scene_hashes))
    avg_lum       = sum(a["avg_luminance"] for a in analyses) / len(analyses)
    sub_presence  = sum(1 for a in analyses if a["subtitle_signal"]) / len(analyses)

    # Classify content type heuristically
    content_type = "unknown"
    if sub_presence > 0.5 and avg_motion < 0.01:
        content_type = "slide_or_talking_head"
    elif avg_motion > 0.05 and unique_scenes > len(frames) * 0.3:
        content_type = "fast_paced_demo"
    elif sub_presence > 0.3 and avg_motion < 0.05:
        content_type = "tutorial_with_captions"
    elif avg_motion < 0.005:
        content_type = "paused_or_static"

    result = {
        "duration_s":      duration,
        "frame_count":     len(frames),
        "avg_motion":      avg_motion,
        "peak_motion":     peak_motion,
        "unique_scenes":   unique_scenes,
        "text_changes":    text_changes,
        "avg_luminance":   avg_lum,
        "subtitle_ratio":  sub_presence,
        "content_type":    content_type,
        "fps_observed":    len(frames) / duration if duration > 0 else 0,
    }

    # Store in scene history for longitudinal tracking
    self._scene_history.append({
        "timestamp": frames[-1].timestamp,
        **result
    })

    return result

def summarize_scene_delta(self, frames: List[VideoFrame]) -> str:
    """
    Produce a human-readable learning summary from a frame sequence.
    Intended for logging into Janus memory / goal planner.
    """
    features = self.process_video_frame_sequence(frames)
    if "error" in features:
        return "No frames available to summarise."

    ct     = features["content_type"]
    dur    = features["duration_s"]
    motion = features["avg_motion"]
    subs   = features["subtitle_ratio"]
    scenes = features["unique_scenes"]
    tc     = features["text_changes"]

    lines = [
        f"Content type: {ct.replace('_', ' ')}.",
        f"Observed {dur:.1f}s of video with {scenes} distinct scenes.",
    ]

    if ct == "tutorial_with_captions":
        lines.append("Likely instructional content with captions — high learning value.")
    elif ct == "slide_or_talking_head":
        lines.append("Slide/lecture format — dense information delivery likely.")
    elif ct == "fast_paced_demo":
        lines.append("Fast demo — visual coding or hardware walkthrough possible.")
    elif ct == "paused_or_static":
        lines.append("Video appears paused or at a static frame.")

    if tc > 5:
        lines.append(f"Text/subtitle region changed {tc} times — active captioning detected.")
    if motion > 0.05:
        lines.append("High visual activity — frequent screen changes or animations.")

    return " ".join(lines)

def get_scene_history(self) -> List[Dict[str, Any]]:
    return list(self._scene_history)
```