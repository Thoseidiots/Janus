"""
janus_brain_adapter.py
======================
Unified brain adapter for ActionPlanner.

Tries backends in priority order:

  1. Remote Avus server (Kaggle GPU — fast)
     Set JANUS_INFERENCE_URL in Kaggle.env to the ngrok URL from
     janus_kaggle_inference_server.py

  2. Local AvusBrain (your trained Avus model on CPU — slow but works)

  3. Stub (returns a screenshot action so ActionPlanner doesn't crash)

Only your own trained models are used. No third-party LLMs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("janus.brain_adapter")


# ---------------------------------------------------------------------------
# 1. Remote Avus server (Kaggle GPU)
# ---------------------------------------------------------------------------

def _load_remote_avus() -> Optional[Any]:
    """Connect to the remote Avus inference server if JANUS_INFERENCE_URL is set."""
    url = os.environ.get("JANUS_INFERENCE_URL", "").rstrip("/")

    if not url:
        for env_file in ["Kaggle.env", "Credintials.env", "Credentials.env", ".env"]:
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("JANUS_INFERENCE_URL="):
                            url = line.split("=", 1)[1].strip().rstrip("/")
                            break
            except Exception:
                pass
            if url:
                break

    if not url or "your-ngrok-url" in url:
        logger.debug("JANUS_INFERENCE_URL not configured — skipping remote Avus")
        return None

    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                logger.info("JanusBrain: remote Avus server at %s (device=%s)", url, data.get("device"))
                return _RemoteAvusBrain(url)
    except Exception as exc:
        logger.debug("Remote Avus server not reachable: %s", exc)

    return None


class _RemoteAvusBrain:
    """Sends prompts to the Kaggle Avus GPU server."""

    def __init__(self, base_url: str) -> None:
        self._url = base_url

    def ask(self, prompt: str) -> str:
        import urllib.request
        payload = json.dumps({
            "prompt": prompt[-800:],
            "max_new_tokens": 300,
            "temperature": 0.5,
        }).encode()
        req = urllib.request.Request(
            f"{self._url}/generate",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read()).get("text", "")
        except Exception as exc:
            logger.warning("Remote Avus inference failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# 2. Local AvusBrain (your trained model)
# ---------------------------------------------------------------------------

def _load_local_avus() -> Optional[Any]:
    """Load AvusBrain backed by your locally trained Avus weights."""
    try:
        from avus_brain import AvusBrain  # type: ignore
        brain = AvusBrain()
        brain.ensure_loaded()
        logger.info("JanusBrain: using local AvusBrain")
        return _LocalAvusBrain(brain)
    except Exception as exc:
        logger.debug("Local AvusBrain not available: %s", exc)
        return None


class _LocalAvusBrain:
    """Wraps AvusBrain with raw generation for structured JSON prompts."""

    def __init__(self, brain: Any) -> None:
        self._brain = brain

    def ask(self, prompt: str) -> str:
        try:
            if "json" in prompt.lower() or "format as" in prompt.lower():
                # Use raw generation so the model outputs JSON, not a Q&A answer
                return self._brain._avus.generate(
                    prompt[-600:],
                    max_new_tokens=300,
                    temperature=0.5,
                )
            return self._brain.ask(prompt, max_tokens=300)
        except Exception as exc:
            logger.warning("Local AvusBrain.ask failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# 3. Stub (always available — keeps ActionPlanner from crashing)
# ---------------------------------------------------------------------------

class _StubBrain:
    """Returns a screenshot action when no real model is available."""

    def ask(self, prompt: str) -> str:
        goal_match = re.search(r"GOAL:\s*(.+)", prompt)
        goal = goal_match.group(1).strip()[:80] if goal_match else "complete task"
        return json.dumps([{
            "action_type": "screenshot",
            "params": {},
            "confidence": 0.5,
            "rationale": f"No brain available — observing screen for: {goal}",
        }])


# ---------------------------------------------------------------------------
# JanusBrain — main adapter
# ---------------------------------------------------------------------------

class JanusBrain:
    """
    Unified brain adapter. Uses your own trained Avus model only.
    Priority: remote Kaggle GPU → local CPU → stub.
    """

    def __init__(self) -> None:
        self._backend: Any = (
            _load_remote_avus()
            or _load_local_avus()
            or _StubBrain()
        )
        logger.info("JanusBrain: backend = %s", type(self._backend).__name__)

    def ask(self, prompt: str) -> str:
        try:
            response = self._backend.ask(prompt)
        except Exception as exc:
            logger.warning("JanusBrain.ask error: %s", exc)
            response = ""

        if not self._is_valid_json_list(response):
            return json.dumps([{
                "action_type": "screenshot",
                "params": {},
                "confidence": 0.5,
                "rationale": f"Response not parseable: {response[:80]}",
            }])

        return response

    @staticmethod
    def _is_valid_json_list(text: str) -> bool:
        try:
            parsed = json.loads(text.strip())
            return isinstance(parsed, list) and len(parsed) > 0
        except Exception:
            return False


def get_brain() -> JanusBrain:
    return JanusBrain()
