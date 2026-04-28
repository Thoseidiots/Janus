"""
Human-like capabilities for the Janus Autonomous Reasoning Engine.

Integrates with janus_human_core and janus_human_capabilities (optional)
to provide personality state, natural behaviour patterns, and social
context adaptation.

Requirements: REQ-13.3, REQ-7.3
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict

logger = logging.getLogger("janus.advanced.human_capabilities")

# Optional integrations
try:
    import janus_human_core as _human_core  # type: ignore
    _HUMAN_CORE_AVAILABLE = True
    logger.info("janus_human_core loaded successfully")
except ImportError:
    _human_core = None
    _HUMAN_CORE_AVAILABLE = False
    logger.debug("janus_human_core not available — using stub")

try:
    import janus_human_capabilities as _human_caps  # type: ignore
    _HUMAN_CAPS_AVAILABLE = True
    logger.info("janus_human_capabilities loaded successfully")
except ImportError:
    _human_caps = None
    _HUMAN_CAPS_AVAILABLE = False
    logger.debug("janus_human_capabilities not available — using stub")


# Stub personality defaults
_DEFAULT_MOOD = "neutral"
_DEFAULT_ENERGY = 0.7
_DEFAULT_CURIOSITY = 0.8

# Natural language filler patterns for humanisation
_FILLERS = [
    "I think ",
    "It seems like ",
    "From my perspective, ",
    "Interestingly, ",
    "Worth noting: ",
]

_SOCIAL_FORMAL = [
    "Dear {recipient},\n\n{message}\n\nBest regards,\nJanus",
    "Hello {recipient},\n\n{message}\n\nKind regards,\nJanus",
]

_SOCIAL_CASUAL = [
    "Hey {recipient}! {message} Cheers, Janus",
    "Hi {recipient}, {message} Thanks!",
]


class HumanCapabilities:
    """
    Provides human-like behaviour patterns for Janus.

    Delegates to janus_human_core / janus_human_capabilities when available;
    otherwise uses lightweight stubs.

    Usage::

        hc = HumanCapabilities()
        state = hc.get_personality_state()
        natural = hc.apply_natural_behavior("I completed the task.")
        adapted = hc.adapt_to_social_context("Hello!", {"formality": "formal"})
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_personality_state(self) -> Dict[str, Any]:
        """
        Return the current personality state (mood, energy, curiosity).

        Returns:
            Dict with keys: mood (str), energy (float 0–1), curiosity (float 0–1).
        """
        if _HUMAN_CORE_AVAILABLE and _human_core is not None:
            try:
                get_fn = getattr(_human_core, "get_personality_state", None) or getattr(
                    _human_core, "get_state", None
                )
                if get_fn is not None:
                    state = get_fn()
                    logger.debug("Personality state from janus_human_core: %s", state)
                    return dict(state)
            except Exception as exc:
                logger.debug("janus_human_core get_personality_state failed: %s", exc)

        # Stub: return slightly randomised defaults
        energy = min(1.0, max(0.0, _DEFAULT_ENERGY + random.uniform(-0.1, 0.1)))
        curiosity = min(1.0, max(0.0, _DEFAULT_CURIOSITY + random.uniform(-0.1, 0.1)))
        return {
            "mood": _DEFAULT_MOOD,
            "energy": round(energy, 2),
            "curiosity": round(curiosity, 2),
        }

    def apply_natural_behavior(self, text: str) -> str:
        """
        Add human-like patterns to a piece of text.

        Args:
            text: The original text to humanise.

        Returns:
            Text with natural-language patterns applied.
        """
        if _HUMAN_CAPS_AVAILABLE and _human_caps is not None:
            try:
                apply_fn = getattr(_human_caps, "apply_natural_behavior", None) or getattr(
                    _human_caps, "humanize", None
                )
                if apply_fn is not None:
                    result = apply_fn(text)
                    logger.debug("Natural behavior applied via janus_human_capabilities")
                    return str(result)
            except Exception as exc:
                logger.debug("janus_human_capabilities apply failed: %s", exc)

        if _HUMAN_CORE_AVAILABLE and _human_core is not None:
            try:
                apply_fn = getattr(_human_core, "apply_natural_behavior", None)
                if apply_fn is not None:
                    return str(apply_fn(text))
            except Exception as exc:
                logger.debug("janus_human_core apply_natural_behavior failed: %s", exc)

        # Stub: prepend a random filler phrase
        filler = random.choice(_FILLERS)
        # Only add filler if text doesn't already start with one
        if not any(text.startswith(f.strip()) for f in _FILLERS):
            return filler + text[0].lower() + text[1:] if text else text
        return text

    def adapt_to_social_context(self, message: str, context: Dict[str, Any]) -> str:
        """
        Adapt a message to the given social context.

        Args:
            message: The message to adapt.
            context: Social context dict. Recognised keys:
                - formality: "formal" | "casual" (default "casual")
                - recipient: name of the recipient (default "")

        Returns:
            Adapted message string.
        """
        if _HUMAN_CAPS_AVAILABLE and _human_caps is not None:
            try:
                adapt_fn = getattr(_human_caps, "adapt_to_social_context", None) or getattr(
                    _human_caps, "adapt_message", None
                )
                if adapt_fn is not None:
                    result = adapt_fn(message, context)
                    logger.debug("Social adaptation via janus_human_capabilities")
                    return str(result)
            except Exception as exc:
                logger.debug("janus_human_capabilities adapt failed: %s", exc)

        if _HUMAN_CORE_AVAILABLE and _human_core is not None:
            try:
                adapt_fn = getattr(_human_core, "adapt_to_social_context", None)
                if adapt_fn is not None:
                    return str(adapt_fn(message, context))
            except Exception as exc:
                logger.debug("janus_human_core adapt failed: %s", exc)

        # Stub: apply a template based on formality
        formality = context.get("formality", "casual")
        recipient = context.get("recipient", "")

        if formality == "formal":
            template = random.choice(_SOCIAL_FORMAL)
        else:
            template = random.choice(_SOCIAL_CASUAL)

        return template.format(recipient=recipient, message=message)
