"""
ReliabilityManager — checkpoints, loop detection, and self-healing.

All external integrations are optional; degrades gracefully when modules
are unavailable.

Requirements: REQ-12.3, REQ-9.1
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from janus_checkpoint import JanusCheckpoint  # type: ignore
    _CHECKPOINT_AVAILABLE = True
except ImportError:
    _CHECKPOINT_AVAILABLE = False
    logger.debug("janus_checkpoint not available — using JSON file checkpoints")

try:
    from janus_selfheal import JanusSelfHeal  # type: ignore
    _SELFHEAL_AVAILABLE = True
except ImportError:
    _SELFHEAL_AVAILABLE = False
    logger.debug("janus_selfheal not available — using built-in self-heal stub")

try:
    from janus_loop_detector import LoopDetector  # type: ignore
    _LOOP_DETECTOR_AVAILABLE = True
except ImportError:
    _LOOP_DETECTOR_AVAILABLE = False
    logger.debug("janus_loop_detector not available — using built-in loop detection")


# ---------------------------------------------------------------------------
# ReliabilityManager
# ---------------------------------------------------------------------------

class ReliabilityManager:
    """
    Provides checkpoint persistence, loop detection, and self-healing.
    """

    _CHECKPOINT_DIR = "janus_checkpoints"

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        self._checkpoint_dir = checkpoint_dir or self._CHECKPOINT_DIR
        self._checkpoint_backend: Optional[object] = None
        self._selfheal_backend: Optional[object] = None
        self._loop_detector_backend: Optional[object] = None

        os.makedirs(self._checkpoint_dir, exist_ok=True)

        if _CHECKPOINT_AVAILABLE:
            try:
                self._checkpoint_backend = JanusCheckpoint()
                logger.info("JanusCheckpoint backend initialised")
            except Exception as exc:
                logger.warning("JanusCheckpoint init failed: %s", exc)

        if _SELFHEAL_AVAILABLE:
            try:
                self._selfheal_backend = JanusSelfHeal()
                logger.info("JanusSelfHeal backend initialised")
            except Exception as exc:
                logger.warning("JanusSelfHeal init failed: %s", exc)

        if _LOOP_DETECTOR_AVAILABLE:
            try:
                self._loop_detector_backend = LoopDetector()
                logger.info("LoopDetector backend initialised")
            except Exception as exc:
                logger.warning("LoopDetector init failed: %s", exc)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, state_dict: Dict) -> str:
        """
        Persist *state_dict* to a JSON file and return a checkpoint_id.
        """
        checkpoint_id = str(uuid.uuid4())

        if self._checkpoint_backend is not None:
            try:
                result = self._checkpoint_backend.save(  # type: ignore[union-attr]
                    state=state_dict,
                    checkpoint_id=checkpoint_id,
                )
                checkpoint_id = str(getattr(result, "id", checkpoint_id) or checkpoint_id)
                logger.info("Checkpoint saved via backend: %s", checkpoint_id)
                return checkpoint_id
            except Exception as exc:
                logger.warning("checkpoint backend save failed: %s", exc)

        # Fallback: write to JSON file
        path = os.path.join(self._checkpoint_dir, f"{checkpoint_id}.json")
        payload = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "state": state_dict,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.debug("Checkpoint saved to file: %s", path)
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Dict:
        """
        Load and return the state dict for *checkpoint_id*.

        Returns an empty dict if the checkpoint cannot be found.
        """
        if self._checkpoint_backend is not None:
            try:
                raw = self._checkpoint_backend.load(checkpoint_id=checkpoint_id)  # type: ignore[union-attr]
                if raw is not None:
                    return dict(raw) if not isinstance(raw, dict) else raw
            except Exception as exc:
                logger.warning("checkpoint backend load failed: %s", exc)

        path = os.path.join(self._checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(path):
            logger.warning("Checkpoint file not found: %s", path)
            return {}

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        return payload.get("state", {})

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def detect_loop(self, action_history: List[str], window: int = 10) -> bool:
        """
        Return True if the same action appears more than 3 times within
        the last *window* entries of *action_history*.
        """
        if self._loop_detector_backend is not None:
            try:
                result = self._loop_detector_backend.detect(  # type: ignore[union-attr]
                    history=action_history,
                    window=window,
                )
                return bool(result)
            except Exception as exc:
                logger.debug("loop_detector backend failed: %s", exc)

        recent = action_history[-window:] if len(action_history) > window else action_history
        counts = Counter(recent)
        return any(count > 3 for count in counts.values())

    # ------------------------------------------------------------------
    # Self-healing
    # ------------------------------------------------------------------

    def trigger_self_heal(self) -> bool:
        """
        Attempt to self-heal the system.

        Delegates to janus_selfheal when available; otherwise logs a
        warning and returns True (best-effort stub).
        """
        if self._selfheal_backend is not None:
            try:
                result = self._selfheal_backend.heal()  # type: ignore[union-attr]
                success = bool(result) if result is not None else True
                logger.info("Self-heal via backend: success=%s", success)
                return success
            except Exception as exc:
                logger.warning("selfheal backend failed: %s", exc)

        logger.warning("Self-heal triggered (stub) — no backend available")
        return True
