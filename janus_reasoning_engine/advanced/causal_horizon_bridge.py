"""
Causal Horizon Bridge for the Janus Autonomous Reasoning Engine.

Integrates with janus_causal_horizon (optional) to provide signal
propagation modeling and temporal causality tracking.

Requirements: REQ-13.1
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("janus.advanced.causal_horizon")

# Optional integration with janus_causal_horizon
try:
    import janus_causal_horizon as _causal_horizon  # type: ignore
    _CAUSAL_HORIZON_AVAILABLE = True
    logger.info("janus_causal_horizon loaded successfully")
except ImportError:
    _causal_horizon = None
    _CAUSAL_HORIZON_AVAILABLE = False
    logger.debug("janus_causal_horizon not available — using stub implementation")


class CausalHorizonBridge:
    """
    Bridge to janus_causal_horizon for causal reasoning.

    When janus_causal_horizon is available, delegates to it.
    Otherwise provides a lightweight in-process stub that tracks
    cause→effect relationships and temporal event sequences.

    Usage::

        bridge = CausalHorizonBridge()
        bridge.emit_signal("user_applied_for_job", "application_sent", {})
        effects = bridge.get_propagated_effects("user_applied_for_job")
        causality = bridge.track_temporal_causality([
            {"event": "apply", "timestamp": "2024-01-01T10:00:00"},
            {"event": "interview", "timestamp": "2024-01-02T14:00:00"},
        ])
    """

    def __init__(self) -> None:
        # In-process storage for stub mode
        self._signals: List[Dict[str, Any]] = []
        self._cause_effects: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_signal(
        self,
        cause: str,
        effect: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit a causal signal linking a cause to an effect.

        Args:
            cause: Description of the causal event.
            effect: Description of the resulting effect.
            metadata: Optional additional data about the signal.
        """
        metadata = metadata or {}
        signal = {
            "cause": cause,
            "effect": effect,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if _CAUSAL_HORIZON_AVAILABLE and _causal_horizon is not None:
            try:
                emit_fn = getattr(_causal_horizon, "emit_signal", None)
                if emit_fn is not None:
                    emit_fn(cause, effect, metadata)
                    logger.debug("Signal emitted via janus_causal_horizon")
                    return
            except Exception as exc:
                logger.debug("janus_causal_horizon emit failed: %s", exc)

        # Stub: store locally
        self._signals.append(signal)
        self._cause_effects[cause].append({"effect": effect, "metadata": metadata})
        logger.debug("Signal stored (stub): %s → %s", cause, effect)

    def get_propagated_effects(self, cause: str) -> List[Dict[str, Any]]:
        """
        Retrieve all effects that have been linked to a given cause.

        Args:
            cause: The causal event to look up.

        Returns:
            List of effect dicts, each with 'effect' and 'metadata' keys.
        """
        if _CAUSAL_HORIZON_AVAILABLE and _causal_horizon is not None:
            try:
                get_fn = getattr(_causal_horizon, "get_propagated_effects", None)
                if get_fn is not None:
                    return list(get_fn(cause))
            except Exception as exc:
                logger.debug("janus_causal_horizon get_effects failed: %s", exc)

        return list(self._cause_effects.get(cause, []))

    def track_temporal_causality(
        self, event_sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyse a sequence of events to identify temporal causal patterns.

        Args:
            event_sequence: Ordered list of event dicts. Each dict should
                contain at least an 'event' key; a 'timestamp' key is
                optional but improves analysis.

        Returns:
            Dict with 'chain' (ordered events), 'causal_pairs' (adjacent
            cause→effect pairs), and 'length'.
        """
        if _CAUSAL_HORIZON_AVAILABLE and _causal_horizon is not None:
            try:
                track_fn = getattr(_causal_horizon, "track_temporal_causality", None)
                if track_fn is not None:
                    return dict(track_fn(event_sequence))
            except Exception as exc:
                logger.debug("janus_causal_horizon track failed: %s", exc)

        # Stub: build simple chain analysis
        chain = [e.get("event", str(e)) for e in event_sequence]
        causal_pairs = [
            {"cause": chain[i], "effect": chain[i + 1]}
            for i in range(len(chain) - 1)
        ]
        return {
            "chain": chain,
            "causal_pairs": causal_pairs,
            "length": len(chain),
        }
