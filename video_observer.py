# video_observer.py

“””
Core video observation module for Janus.
Captures a target window (browser playing YouTube/tutorial), runs motion
detection, subtitle region scanning, scene summarisation, and feeds
everything into Janus memory + goal planner.

Windows-only: ctypes user32/gdi32/kernel32 only — no third-party deps.
“””

import ctypes
import ctypes.wintypes as wt
import threading
import time
import collections
import math
import subprocess
import queue
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Deque, Dict, Any

# ── ctypes setup ──────────────────────────────────────────────────────────────

user32  = ctypes.windll.user32
gdi32   = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# Win32 constants

DIB_RGB_COLORS  = 0
BI_RGB          = 0
SRCCOPY         = 0x00CC0020
SM_CXSCREEN     = 0
SM_CYSCREEN     = 1

class BITMAPINFOHEADER(ctypes.Structure):
*fields* = [
(“biSize”,          ctypes.c_uint32),
(“biWidth”,         ctypes.c_int32),
(“biHeight”,        ctypes.c_int32),
(“biPlanes”,        ctypes.c_uint16),
(“biBitCount”,      ctypes.c_uint16),
(“biCompression”,   ctypes.c_uint32),
(“biSizeImage”,     ctypes.c_uint32),
(“biXPelsPerMeter”, ctypes.c_int32),
(“biYPelsPerMeter”, ctypes.c_int32),
(“biClrUsed”,       ctypes.c_uint32),
(“biClrImportant”,  ctypes.c_uint32),
]

class BITMAPINFO(ctypes.Structure):
*fields* = [(“bmiHeader”, BITMAPINFOHEADER), (“bmiColors”, ctypes.c_uint32 * 3)]

class RECT(ctypes.Structure):
*fields* = [(“left”, ctypes.c_long), (“top”, ctypes.c_long),
(“right”, ctypes.c_long), (“bottom”, ctypes.c_long)]

# ── Frame dataclass ───────────────────────────────────────────────────────────

@dataclass
class VideoFrame:
timestamp:   float
width:       int
height:      int
data:        bytearray          # raw BGRA bytes
motion_score: float = 0.0
subtitle_text: str  = “”
scene_hash:  str    = “”

# ── Window finder ─────────────────────────────────────────────────────────────

def _find_window_by_title(partial_title: str) -> Optional[int]:
“”“Enumerate top-level windows, return hwnd whose title contains partial_title.”””
found = ctypes.c_void_p(None)

```
@ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
def callback(hwnd, lParam):
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return True
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    if partial_title.lower() in buf.value.lower():
        found.value = hwnd
        return False
    return True

user32.EnumWindows(callback, 0)
return found.value
```

def _get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
r = RECT()
if user32.GetWindowRect(hwnd, ctypes.byref(r)):
return (r.left, r.top, r.right - r.left, r.bottom - r.top)
return None

# ── Raw window capture (scoped to target hwnd) ────────────────────────────────

def _capture_window(hwnd: int) -> Optional[VideoFrame]:
rect = _get_window_rect(hwnd)
if rect is None:
return None
x, y, w, h = rect
if w <= 0 or h <= 0:
return None

```
hwnd_dc   = user32.GetDC(hwnd)
mem_dc    = gdi32.CreateCompatibleDC(hwnd_dc)
bitmap    = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
old_bmp   = gdi32.SelectObject(mem_dc, bitmap)

# PrintWindow captures the window even if partially occluded
user32.PrintWindow(hwnd, mem_dc, 2)  # PW_RENDERFULLCONTENT = 2

bmi = BITMAPINFO()
bmi.bmiHeader.biSize        = ctypes.sizeof(BITMAPINFOHEADER)
bmi.bmiHeader.biWidth       = w
bmi.bmiHeader.biHeight      = -h   # top-down
bmi.bmiHeader.biPlanes      = 1
bmi.bmiHeader.biBitCount    = 32
bmi.bmiHeader.biCompression = BI_RGB

buf = bytearray(w * h * 4)
raw = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
gdi32.GetDIBits(mem_dc, bitmap, 0, h, raw, ctypes.byref(bmi), DIB_RGB_COLORS)

gdi32.SelectObject(mem_dc, old_bmp)
gdi32.DeleteObject(bitmap)
gdi32.DeleteDC(mem_dc)
user32.ReleaseDC(hwnd, hwnd_dc)

return VideoFrame(
    timestamp=time.time(),
    width=w,
    height=h,
    data=buf,
    scene_hash=hashlib.md5(bytes(buf[::64])).hexdigest()  # fast sparse hash
)
```

# ── Motion detection (byte-level diff, no numpy) ─────────────────────────────

def _compute_motion(prev: VideoFrame, curr: VideoFrame) -> float:
“”“Returns 0.0–1.0 motion score via MAD on luminance-approximated samples.”””
if prev.width != curr.width or prev.height != curr.height:
return 1.0   # size changed = treat as scene cut

```
# Sample every 16th pixel (stride=64 bytes in BGRA) for speed
stride   = 64
p_data   = memoryview(prev.data)
c_data   = memoryview(curr.data)
total    = 0
n        = 0
length   = len(prev.data) - 4
i        = 0
while i < length:
    # Approximate luminance: 0.11B + 0.59G + 0.30R (integer weights *100)
    pb = p_data[i]; pg = p_data[i+1]; pr = p_data[i+2]
    cb = c_data[i]; cg = c_data[i+1]; cr = c_data[i+2]
    p_lum = (11*pb + 59*pg + 30*pr)
    c_lum = (11*cb + 59*cg + 30*cr)
    diff  = p_lum - c_lum
    total += diff if diff >= 0 else -diff
    n     += 1
    i     += stride

if n == 0:
    return 0.0
return min(1.0, (total / n) / 10000.0)
```

# ── Subtitle region scanner (bottom 15% of frame) ────────────────────────────

def _scan_subtitle_region(frame: VideoFrame) -> str:
“””
Heuristic: scan bottom 15% of frame for high-contrast horizontal edges.
Returns a simple description string — not OCR, but detects *presence* and
rough position of text-like edges for change tracking.
“””
w, h    = frame.width, frame.height
sub_y   = int(h * 0.85)
sub_h   = h - sub_y
data    = memoryview(frame.data)

```
edge_count = 0
bright_px  = 0
dark_px    = 0

row_stride = w * 4
for row in range(sub_y, h - 1):
    row_off = row * row_stride
    for col in range(4, w - 4, 4):   # sample every 4th column
        off = row_off + col * 4
        if off + 4 >= len(frame.data):
            continue
        b = data[off]; g = data[off+1]; r = data[off+2]
        lum = (11*b + 59*g + 30*r) // 100

        # Horizontal edge: compare to neighbour 4px right
        off2 = off + 16
        if off2 + 2 < len(frame.data):
            b2 = data[off2]; g2 = data[off2+1]; r2 = data[off2+2]
            lum2 = (11*b2 + 59*g2 + 30*r2) // 100
            if abs(lum - lum2) > 60:
                edge_count += 1

        if lum > 200:
            bright_px += 1
        elif lum < 50:
            dark_px += 1

total_sampled = (sub_h * w) // 4
if total_sampled == 0:
    return ""

edge_ratio  = edge_count / total_sampled
bright_ratio = bright_px / total_sampled

if edge_ratio > 0.03 and bright_ratio > 0.05:
    return f"subtitle_detected(edges={edge_count},bright={bright_px})"
return ""
```

# ── Scene delta summariser ────────────────────────────────────────────────────

def _summarise_delta(frames: List[VideoFrame]) -> str:
“””
Produce a plain-text learning summary from a sequence of frames.
Uses motion scores, subtitle detections, and scene hash change rate.
“””
if not frames:
return “No frames captured.”

```
duration      = frames[-1].timestamp - frames[0].timestamp
avg_motion    = sum(f.motion_score for f in frames) / len(frames)
sub_frames    = [f for f in frames if f.subtitle_text]
scene_changes = len(set(f.scene_hash for f in frames))
playing       = avg_motion > 0.005

lines = [
    f"Observed {len(frames)} frames over {duration:.1f}s.",
    f"Playback state: {'playing' if playing else 'paused/static'}.",
    f"Average motion: {avg_motion:.4f} ({'high' if avg_motion > 0.05 else 'low'}).",
    f"Scene changes detected: {scene_changes}.",
    f"Subtitle-like regions detected in {len(sub_frames)}/{len(frames)} frames.",
]
if sub_frames:
    lines.append("Subtitle activity suggests spoken/captioned content.")
if avg_motion > 0.08:
    lines.append("High motion — likely fast-paced visual tutorial or demo.")
elif avg_motion < 0.002 and playing:
    lines.append("Very low motion — likely slide-based or talking-head content.")

return " ".join(lines)
```

# ── VideoObserver ─────────────────────────────────────────────────────────────

class VideoObserver:
“””
Main video observation engine.  Runs a background capture thread that
fills a ring buffer of VideoFrame objects.  Designed to be driven by
the cognitive loop or called directly.
“””

```
BUFFER_SIZE  = 60       # frames kept in memory
TARGET_FPS   = 10       # capture target
FRAME_DELAY  = 1.0 / TARGET_FPS

def __init__(self, memory=None, goal_planner=None):
    self.memory       = memory        # janus-brain ReflectionMemory (optional)
    self.goal_planner = goal_planner  # GoalPlanner (optional)

    self._hwnd:    Optional[int]  = None
    self._title:   str            = ""
    self._running: bool           = False
    self._thread:  Optional[threading.Thread] = None
    self._lock     = threading.Lock()

    self._buffer:  Deque[VideoFrame] = collections.deque(maxlen=self.BUFFER_SIZE)
    self._summary_queue: queue.Queue = queue.Queue(maxsize=32)

    # Playback control key codes (VK codes)
    self._VK_SPACE = 0x20
    self._VK_LEFT  = 0x25
    self._VK_RIGHT = 0x27

# ── Public API ────────────────────────────────────────────────────────────

def watch_video(self, url_or_title: str, browser_path: Optional[str] = None) -> bool:
    """
    Launch a browser to url_or_title (if it looks like a URL) or
    find an existing window matching the title, then start observing.
    Returns True if observation started.
    """
    if self._running:
        self.stop()

    # If it's a URL, open in default browser
    if url_or_title.startswith("http"):
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", url_or_title],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            time.sleep(3.0)   # give browser time to open
            self._title = "YouTube"   # default search target
        except Exception:
            self._title = url_or_title
    else:
        self._title = url_or_title

    # Find the window
    hwnd = self._locate_window(self._title, retries=8, delay=1.5)
    if hwnd is None:
        return False

    self._hwnd = hwnd
    self._running = True
    self._thread = threading.Thread(target=self._capture_loop, daemon=True)
    self._thread.start()
    return True

def observe_window(self, hwnd: int) -> bool:
    """Attach observer to an already-known hwnd."""
    if self._running:
        self.stop()
    self._hwnd    = hwnd
    self._running = True
    self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
    self._thread.start()
    return True

def stop(self):
    self._running = False
    if self._thread and self._thread.is_alive():
        self._thread.join(timeout=3.0)
    self._thread = None
    self._hwnd   = None

def get_current_summary(self) -> str:
    with self._lock:
        frames = list(self._buffer)
    return _summarise_delta(frames)

def get_last_n_seconds(self, seconds: float) -> List[VideoFrame]:
    cutoff = time.time() - seconds
    with self._lock:
        return [f for f in self._buffer if f.timestamp >= cutoff]

def is_playing(self) -> bool:
    frames = self.get_last_n_seconds(2.0)
    if len(frames) < 2:
        return False
    motions = [f.motion_score for f in frames]
    return (sum(motions) / len(motions)) > 0.003

def control_playback(self, action: str):
    """action: 'pause', 'play', 'seek_forward', 'seek_back'"""
    if self._hwnd is None:
        return
    user32.SetForegroundWindow(self._hwnd)
    time.sleep(0.1)

    key_map = {
        "pause":        self._VK_SPACE,
        "play":         self._VK_SPACE,
        "seek_forward": self._VK_RIGHT,
        "seek_back":    self._VK_LEFT,
    }
    vk = key_map.get(action)
    if vk:
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP = 2

def drain_summaries(self) -> List[str]:
    """Pull accumulated summaries produced during observation."""
    summaries = []
    while not self._summary_queue.empty():
        try:
            summaries.append(self._summary_queue.get_nowait())
        except queue.Empty:
            break
    return summaries

# ── Internal capture loop ─────────────────────────────────────────────────

def _capture_loop(self):
    prev_frame: Optional[VideoFrame] = None
    frames_since_summary = 0
    SUMMARY_EVERY = self.TARGET_FPS * 30   # summarise every ~30s

    while self._running:
        t0 = time.perf_counter()

        # Re-find window if it moved or was minimised
        if self._hwnd and not user32.IsWindow(self._hwnd):
            self._hwnd = self._locate_window(self._title, retries=3, delay=1.0)
            if self._hwnd is None:
                time.sleep(2.0)
                continue

        frame = None
        if self._hwnd:
            try:
                frame = _capture_window(self._hwnd)
            except Exception:
                frame = None

        if frame is not None:
            if prev_frame is not None:
                frame.motion_score = _compute_motion(prev_frame, frame)
            frame.subtitle_text = _scan_subtitle_region(frame)

            with self._lock:
                self._buffer.append(frame)

            prev_frame = frame
            frames_since_summary += 1

            # Periodic summary into queue / memory
            if frames_since_summary >= SUMMARY_EVERY:
                summary = self.get_current_summary()
                self._push_summary(summary)
                frames_since_summary = 0

        elapsed = time.perf_counter() - t0
        sleep   = self.FRAME_DELAY - elapsed
        if sleep > 0:
            time.sleep(sleep)

def _push_summary(self, summary: str):
    """Feed summary to memory and the summary queue."""
    try:
        self._summary_queue.put_nowait(summary)
    except queue.Full:
        pass

    # If janus-brain memory is attached, store it
    if self.memory is not None:
        try:
            import torch
            from janus_brain.homeostasis import ValenceVector
            # Neutral valence for knowledge ingestion
            neutral = ValenceVector(
                pleasure=torch.tensor(0.5), arousal=torch.tensor(0.4),
                curiosity=torch.tensor(0.8), autonomy=torch.tensor(0.6),
                connection=torch.tensor(0.4), competence=torch.tensor(0.5),
            )
            self.memory.add(neutral, f"[VIDEO_OBSERVATION] {summary}")
        except Exception:
            pass

def _locate_window(self, title: str, retries: int = 5, delay: float = 1.0) -> Optional[int]:
    for _ in range(retries):
        hwnd = _find_window_by_title(title)
        if hwnd:
            return hwnd
        time.sleep(delay)
    return None
```

# ── Module-level singleton (importable by other modules) ──────────────────────

_observer: Optional[VideoObserver] = None

def get_observer() -> VideoObserver:
global _observer
if _observer is None:
_observer = VideoObserver()
return _observer