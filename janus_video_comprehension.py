"""
janus_video_comprehension.py
=============================
Janus actually understands what's happening in videos.

Builds on the existing video_observer.py (frame capture) and
screen_interpreter.py (pixel → text) to add real comprehension:

  - Watch a video and understand what's being taught
  - Extract key concepts, steps, and instructions
  - Answer questions about video content
  - Learn skills from tutorial videos
  - Summarize long videos into actionable notes

No API keys. Uses:
  - video_observer.py for frame capture (already built)
  - screen_interpreter.py for pixel analysis (already built)
  - JanusBrain for comprehension
  - pyttsx3 for audio transcription fallback

Usage:
    from janus_video_comprehension import JanusVideoComprehension
    vc = JanusVideoComprehension()

    # Watch and understand a video file
    notes = vc.watch_and_learn("tutorial.mp4")

    # Watch a live window (YouTube, etc.)
    notes = vc.watch_window("YouTube", duration_seconds=300)

    # Ask about what was watched
    answer = vc.ask("What were the main steps shown?")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Video note ────────────────────────────────────────────────────────────────

@dataclass
class VideoNote:
    timestamp:   float
    description: str
    key_points:  List[str] = field(default_factory=list)
    action_type: str       = "observation"  # observation|instruction|demo|transition

    def to_dict(self) -> dict:
        return {
            "timestamp":   f"{self.timestamp:.1f}s",
            "description": self.description,
            "key_points":  self.key_points,
            "type":        self.action_type,
        }


@dataclass
class VideoSummary:
    title:       str
    duration:    float
    notes:       List[VideoNote]
    key_concepts: List[str]
    action_steps: List[str]
    summary:     str
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            f"Duration: {self.duration:.0f}s | Notes: {len(self.notes)}",
            "",
            "## Summary",
            self.summary,
            "",
            "## Key Concepts",
        ]
        for concept in self.key_concepts:
            lines.append(f"- {concept}")
        lines += ["", "## Action Steps"]
        for i, step in enumerate(self.action_steps, 1):
            lines.append(f"{i}. {step}")
        lines += ["", "## Detailed Notes"]
        for note in self.notes[:20]:
            lines.append(f"**[{note.timestamp:.0f}s]** {note.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title":        self.title,
            "duration":     self.duration,
            "summary":      self.summary,
            "key_concepts": self.key_concepts,
            "action_steps": self.action_steps,
            "note_count":   len(self.notes),
            "created_at":   self.created_at,
        }


# ── Frame analyzer ────────────────────────────────────────────────────────────

class FrameAnalyzer:
    """
    Converts video frames into text descriptions using screen_interpreter.
    Groups similar frames to avoid redundant analysis.
    """

    def __init__(self):
        self._interpreter = None

    def _get_interpreter(self):
        if self._interpreter is None:
            from screen_interpreter import ScreenInterpreter
            self._interpreter = ScreenInterpreter()
        return self._interpreter

    def describe_frame(self, frame_data: bytes, width: int, height: int,
                       context: str = "") -> str:
        """Convert a frame to a text description."""
        try:
            interp = self._get_interpreter()
            return interp.interpret(frame_data, width, height, goal=context)
        except Exception as e:
            return f"Frame at {width}x{height} (analysis failed: {e})"

    def frames_are_similar(self, desc1: str, desc2: str) -> bool:
        """Check if two frame descriptions are similar enough to skip."""
        if not desc1 or not desc2:
            return False
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.85

    def extract_text_from_frame(self, frame_data: bytes, width: int, height: int) -> str:
        """Try to extract visible text from a frame using OCR if available."""
        try:
            import numpy as np
            arr = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 4)
            rgb = arr[:, :, [2, 1, 0]]
            interp = self._get_interpreter()
            return interp.extract_text(rgb)
        except Exception:
            return ""


# ── Video comprehension engine ────────────────────────────────────────────────

class JanusVideoComprehension:
    """
    Watches videos and understands their content.
    """

    SAMPLE_INTERVAL = 5.0   # analyze one frame every 5 seconds
    MAX_NOTES       = 100   # cap notes per video

    def __init__(self):
        self._analyzer  = FrameAnalyzer()
        self._notes:    List[VideoNote] = []
        self._summaries: List[VideoSummary] = []
        self._current_title = ""

    # ── Watch a video file ────────────────────────────────────────────────────

    def watch_file(self, video_path: str, title: str = "") -> VideoSummary:
        """
        Watch a video file and extract understanding.
        Requires opencv-python: pip install opencv-python
        """
        try:
            import cv2
        except ImportError:
            return self._stub_summary(title or Path(video_path).stem,
                                      "opencv-python not installed")

        path = Path(video_path)
        if not path.exists():
            return self._stub_summary(title or path.stem, f"File not found: {video_path}")

        self._notes = []
        self._current_title = title or path.stem
        cap = cv2.VideoCapture(str(path))

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps
        sample_every = int(fps * self.SAMPLE_INTERVAL)

        print(f"[VideoComprehension] Watching: {self._current_title}")
        print(f"  Duration: {duration:.0f}s | Sampling every {self.SAMPLE_INTERVAL}s")

        prev_desc = ""
        frame_idx = 0

        while cap.isOpened() and len(self._notes) < self.MAX_NOTES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                timestamp = frame_idx / fps
                h, w      = frame.shape[:2]

                # Convert BGR to BGRA bytes
                import numpy as np
                bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                raw  = bgra.tobytes()

                desc = self._analyzer.describe_frame(raw, w, h, context=self._current_title)

                # Skip if too similar to previous
                if not self._analyzer.frames_are_similar(desc, prev_desc):
                    note = self._build_note(timestamp, desc, frame, w, h)
                    self._notes.append(note)
                    prev_desc = desc

            frame_idx += 1

        cap.release()
        return self._synthesize(self._current_title, duration)

    # ── Watch a live window ───────────────────────────────────────────────────

    def watch_window(self, window_title: str,
                     duration_seconds: float = 120) -> VideoSummary:
        """
        Watch a live window (browser, video player) for N seconds.
        Uses video_observer.py for frame capture.
        """
        self._notes = []
        self._current_title = window_title

        try:
            from video_observer import get_observer
            obs = get_observer()

            print(f"[VideoComprehension] Watching window: {window_title}")
            ok = obs.watch_video(window_title)
            if not ok:
                return self._stub_summary(window_title, "Window not found")

            start = time.time()
            prev_desc = ""

            while time.time() - start < duration_seconds:
                time.sleep(self.SAMPLE_INTERVAL)
                elapsed = time.time() - start

                frames = obs.get_last_n_seconds(self.SAMPLE_INTERVAL + 1)
                if not frames:
                    continue

                frame = frames[-1]
                desc  = self._analyzer.describe_frame(
                    bytes(frame.data), frame.width, frame.height,
                    context=window_title
                )

                if not self._analyzer.frames_are_similar(desc, prev_desc):
                    note = VideoNote(
                        timestamp   = elapsed,
                        description = desc,
                        action_type = self._classify_action(desc),
                    )
                    self._notes.append(note)
                    prev_desc = desc

                if len(self._notes) >= self.MAX_NOTES:
                    break

            obs.stop()

        except Exception as e:
            return self._stub_summary(window_title, str(e))

        return self._synthesize(window_title, duration_seconds)

    # ── Ask about watched content ─────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Ask a question about the most recently watched video."""
        if not self._notes:
            return "No video has been watched yet."

        context = "\n".join(
            f"[{n.timestamp:.0f}s] {n.description}"
            for n in self._notes[:30]
        )

        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.ask(
                f"Video: {self._current_title}\n\n"
                f"Observations:\n{context}\n\n"
                f"Question: {question}",
                max_tokens=400,
            )
        except Exception:
            # Keyword search fallback
            words = question.lower().split()
            relevant = [n for n in self._notes
                        if any(w in n.description.lower() for w in words)]
            if relevant:
                return "\n".join(n.description for n in relevant[:5])
            return "Could not find relevant content in the video."

    def get_action_steps(self) -> List[str]:
        """Extract ordered action steps from the video."""
        instruction_notes = [n for n in self._notes
                             if n.action_type in ("instruction", "demo")]
        if not instruction_notes:
            instruction_notes = self._notes

        try:
            from avus_brain import get_brain
            brain = get_brain()
            content = "\n".join(n.description for n in instruction_notes[:20])
            response = brain.ask(
                f"Extract the step-by-step instructions from these video observations:\n{content}",
                max_tokens=300,
            )
            # Parse numbered list
            steps = re.findall(r'\d+\.\s+(.+)', response)
            return steps if steps else [response]
        except Exception:
            return [n.description for n in instruction_notes[:10]]

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, title: str, duration: float) -> VideoSummary:
        """Synthesize notes into a structured summary."""
        if not self._notes:
            return self._stub_summary(title, "No frames captured")

        all_descriptions = "\n".join(
            f"[{n.timestamp:.0f}s] {n.description}"
            for n in self._notes[:40]
        )

        # Generate summary
        summary_text = ""
        key_concepts = []
        action_steps = []

        try:
            from avus_brain import get_brain
            brain = get_brain()

            summary_text = brain.summarize(
                f"Video: {title}\nDuration: {duration:.0f}s\n\n"
                f"Observations:\n{all_descriptions}"
            )

            concepts_response = brain.ask(
                f"List the 5 key concepts from this video:\n{all_descriptions}",
                max_tokens=150,
            )
            key_concepts = re.findall(r'[-•*]\s*(.+)', concepts_response)
            if not key_concepts:
                key_concepts = [concepts_response[:200]]

            action_steps = self.get_action_steps()

        except Exception:
            summary_text = f"Watched {len(self._notes)} frames over {duration:.0f}s"
            key_concepts = list(set(
                word for n in self._notes
                for word in n.description.split()
                if len(word) > 6
            ))[:5]

        vs = VideoSummary(
            title        = title,
            duration     = duration,
            notes        = self._notes,
            key_concepts = key_concepts[:8],
            action_steps = action_steps[:10],
            summary      = summary_text,
        )
        self._summaries.append(vs)

        # Save to skill library
        self._save_to_skill_library(vs)

        print(f"[VideoComprehension] Summary: {summary_text[:100]}")
        return vs

    def _build_note(self, timestamp: float, desc: str,
                    frame: Any, w: int, h: int) -> VideoNote:
        """Build a VideoNote from a frame description."""
        # Try OCR for text content
        try:
            import numpy as np
            bgra = frame.tobytes() if hasattr(frame, "tobytes") else bytes(frame)
            text = self._analyzer.extract_text_from_frame(bgra, w, h)
            if text and len(text) > 10:
                desc = f"{desc} Visible text: '{text[:100]}'"
        except Exception:
            pass

        return VideoNote(
            timestamp   = timestamp,
            description = desc,
            action_type = self._classify_action(desc),
        )

    def _classify_action(self, desc: str) -> str:
        """Classify what type of content a frame shows using Avus brain."""
        try:
            from avus_brain import get_brain
            return get_brain().classify_content(desc)
        except Exception:
            pass
        # Heuristic fallback
        desc_lower = desc.lower()
        if any(w in desc_lower for w in ("click", "type", "select", "press", "drag")):
            return "demo"
        if any(w in desc_lower for w in ("step", "first", "next", "then", "finally")):
            return "instruction"
        if any(w in desc_lower for w in ("loading", "transition", "blank", "dark")):
            return "transition"
        return "observation"

    def _save_to_skill_library(self, vs: VideoSummary):
        """Save video summary to the skill library."""
        try:
            skill_lib = Path("skill_library.json")
            data = json.loads(skill_lib.read_text()) if skill_lib.exists() else {"skills": []}
            data["skills"].append({
                "name":        vs.title,
                "type":        "video_learned",
                "summary":     vs.summary,
                "key_concepts": vs.key_concepts,
                "action_steps": vs.action_steps,
                "learned_at":  vs.created_at,
                "confidence":  min(1.0, len(vs.notes) / 20),
            })
            skill_lib.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _stub_summary(self, title: str, error: str) -> VideoSummary:
        return VideoSummary(
            title        = title,
            duration     = 0,
            notes        = [],
            key_concepts = [],
            action_steps = [],
            summary      = f"Could not process video: {error}",
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_vc: Optional[JanusVideoComprehension] = None

def get_video_comprehension() -> JanusVideoComprehension:
    global _vc
    if _vc is None:
        _vc = JanusVideoComprehension()
    return _vc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Janus Video Comprehension")
    parser.add_argument("--watch",   type=str, metavar="FILE_OR_WINDOW")
    parser.add_argument("--window",  action="store_true", help="Watch a window by title")
    parser.add_argument("--duration",type=int, default=120)
    parser.add_argument("--ask",     type=str, metavar="QUESTION")
    parser.add_argument("--steps",   action="store_true", help="Extract action steps")
    args = parser.parse_args()

    vc = JanusVideoComprehension()

    if args.watch:
        if args.window:
            summary = vc.watch_window(args.watch, args.duration)
        else:
            summary = vc.watch_file(args.watch)
        print(summary.to_markdown())

    elif args.ask:
        print(vc.ask(args.ask))

    elif args.steps:
        steps = vc.get_action_steps()
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")

    else:
        parser.print_help()
