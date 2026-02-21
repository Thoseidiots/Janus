# video_capability.py

“””
Registers all VideoObserver tools into JanusCapabilityHub.
Mirrors the style of autonomy_capability.py — each tool is a dict
with name, description, parameters, risk_level, and an execute callable.
“””

from typing import Any, Dict, Optional
from video_observer import get_observer, VideoObserver
from updated_enhanced_vision import EnhancedVisionProcessor

_vision = EnhancedVisionProcessor()

def _obs() -> VideoObserver:
return get_observer()

# ── Tool definitions ──────────────────────────────────────────────────────────

def _tool_watch_video(params: Dict[str, Any]) -> Dict[str, Any]:
“””
Launch + observe a video by URL or window title.
params: {“url_or_title”: str}
“””
target = params.get(“url_or_title”, “”)
if not target:
return {“status”: “error”, “message”: “url_or_title required”}
success = _obs().watch_video(target)
return {
“status”: “ok” if success else “error”,
“message”: f”Observer started for ‘{target}’” if success
else f”Could not find/open window for ‘{target}’”,
}

def _tool_observe_current_video(params: Dict[str, Any]) -> Dict[str, Any]:
“””
Attach to an already-open video window by partial title.
params: {“window_title”: str}
“””
from video_observer import _find_window_by_title
title = params.get(“window_title”, “YouTube”)
hwnd  = _find_window_by_title(title)
if hwnd is None:
return {“status”: “error”, “message”: f”Window ‘{title}’ not found”}
success = _obs().observe_window(hwnd)
return {“status”: “ok” if success else “error”,
“hwnd”: hwnd, “title”: title}

def _tool_get_current_summary(params: Dict[str, Any]) -> Dict[str, Any]:
“”“Return a plain-text summary of the currently buffered frames.”””
summary = _obs().get_current_summary()
return {“status”: “ok”, “summary”: summary}

def _tool_summarize_last_n_seconds(params: Dict[str, Any]) -> Dict[str, Any]:
“””
Run EnhancedVisionProcessor over the last N seconds of frames.
params: {“seconds”: float}  default 60
“””
seconds = float(params.get(“seconds”, 60.0))
frames  = _obs().get_last_n_seconds(seconds)
if not frames:
return {“status”: “ok”, “summary”: “No frames in buffer for that window.”,
“features”: {}}
features = _vision.process_video_frame_sequence(frames)
summary  = _vision.summarize_scene_delta(frames)
return {“status”: “ok”, “summary”: summary, “features”: features}

def _tool_pause_video(params: Dict[str, Any]) -> Dict[str, Any]:
_obs().control_playback(“pause”)
return {“status”: “ok”, “action”: “pause”}

def _tool_play_video(params: Dict[str, Any]) -> Dict[str, Any]:
_obs().control_playback(“play”)
return {“status”: “ok”, “action”: “play”}

def _tool_seek_forward(params: Dict[str, Any]) -> Dict[str, Any]:
_obs().control_playback(“seek_forward”)
return {“status”: “ok”, “action”: “seek_forward”}

def _tool_seek_back(params: Dict[str, Any]) -> Dict[str, Any]:
_obs().control_playback(“seek_back”)
return {“status”: “ok”, “action”: “seek_back”}

def _tool_stop_observing(params: Dict[str, Any]) -> Dict[str, Any]:
_obs().stop()
return {“status”: “ok”, “message”: “Video observation stopped”}

def _tool_is_playing(params: Dict[str, Any]) -> Dict[str, Any]:
playing = _obs().is_playing()
return {“status”: “ok”, “playing”: playing}

def _tool_drain_summaries(params: Dict[str, Any]) -> Dict[str, Any]:
“”“Pull all accumulated periodic summaries from the observer queue.”””
summaries = _obs().drain_summaries()
return {“status”: “ok”, “summaries”: summaries, “count”: len(summaries)}

# ── Registry ──────────────────────────────────────────────────────────────────

VIDEO_TOOLS = [
{
“name”:        “watch_video”,
“description”: “Open a YouTube URL or find a window by title and start observing it for learning.”,
“parameters”:  {“url_or_title”: “str — YouTube URL or partial window title”},
“risk_level”:  “medium”,
“execute”:     _tool_watch_video,
“valence_affinity”: {“curiosity”: 0.6, “competence”: 0.4, “pleasure”: 0.2},
},
{
“name”:        “observe_current_video”,
“description”: “Attach observer to an already-open video window.”,
“parameters”:  {“window_title”: “str”},
“risk_level”:  “low”,
“execute”:     _tool_observe_current_video,
“valence_affinity”: {“curiosity”: 0.5, “autonomy”: 0.3},
},
{
“name”:        “get_current_summary”,
“description”: “Get a plain-text summary of what Janus has observed so far in the current video.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_get_current_summary,
“valence_affinity”: {“competence”: 0.3, “curiosity”: 0.2},
},
{
“name”:        “summarize_last_60s”,
“description”: “Run full vision analysis on last 60 seconds of buffered video frames.”,
“parameters”:  {“seconds”: “float (default 60)”},
“risk_level”:  “low”,
“execute”:     _tool_summarize_last_n_seconds,
“valence_affinity”: {“competence”: 0.5, “curiosity”: 0.4},
},
{
“name”:        “pause_video”,
“description”: “Send pause keypress to the observed video window.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_pause_video,
“valence_affinity”: {“autonomy”: 0.3},
},
{
“name”:        “play_video”,
“description”: “Send play keypress to resume the observed video.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_play_video,
“valence_affinity”: {“autonomy”: 0.3, “curiosity”: 0.2},
},
{
“name”:        “seek_forward”,
“description”: “Skip forward in the current video.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_seek_forward,
“valence_affinity”: {“autonomy”: 0.2},
},
{
“name”:        “seek_back”,
“description”: “Seek backward in the current video.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_seek_back,
“valence_affinity”: {“autonomy”: 0.2},
},
{
“name”:        “stop_observing”,
“description”: “Stop video observation and free resources.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_stop_observing,
“valence_affinity”: {},
},
{
“name”:        “is_video_playing”,
“description”: “Check whether the observed video is currently playing (via motion detection).”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_is_playing,
“valence_affinity”: {},
},
{
“name”:        “drain_video_summaries”,
“description”: “Pull all learning summaries accumulated by the background observer.”,
“parameters”:  {},
“risk_level”:  “low”,
“execute”:     _tool_drain_summaries,
“valence_affinity”: {“competence”: 0.4, “curiosity”: 0.3, “pleasure”: 0.1},
},
]

def register_video_capability(hub: Any) -> None:
“””
Register all video tools into a JanusCapabilityHub instance.
Also injects observer memory reference if hub has a core.
“””
# Wire observer to janus-brain memory if available
obs = get_observer()
if hasattr(hub, “core”) and hasattr(hub.core, “memory”):
obs.memory = hub.core.memory
if hasattr(hub, “core”) and hasattr(hub.core, “goal_planner”):
obs.goal_planner = hub.core.goal_planner

```
for tool in VIDEO_TOOLS:
    hub.register_tool(tool)

print(f"[video_capability] Registered {len(VIDEO_TOOLS)} video tools.")
```