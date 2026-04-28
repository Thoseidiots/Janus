"""
janus_reasoning_engine/communication/social_awareness.py
=========================================================
Social Awareness module for the Janus Reasoning Engine.

Provides:
  - SocialAwareness.adapt_tone(message, context) -> str
  - SocialAwareness.detect_professional_norms(platform) -> Dict[str, str]
  - SocialAwareness.build_rapport_opener(client_name, context) -> str

Integrates janus_humanization_layer.py optionally.

Requirements: REQ-7.3, REQ-13.3
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: janus_humanization_layer
# ---------------------------------------------------------------------------
try:
    from janus_humanization_layer import HumanizedJanus as _HumanizedJanus  # type: ignore
    _HUMANIZATION_AVAILABLE = True
except Exception:
    _HumanizedJanus = None  # type: ignore
    _HUMANIZATION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Platform norms database
# ---------------------------------------------------------------------------

_PLATFORM_NORMS: Dict[str, Dict[str, str]] = {
    "upwork": {
        "tone": "professional",
        "greeting": "Hello",
        "closing": "Best regards",
        "proposal_length": "medium",
        "formality": "formal",
        "notes": "Focus on value delivered, include relevant portfolio links, avoid generic openers.",
    },
    "fiverr": {
        "tone": "friendly",
        "greeting": "Hi",
        "closing": "Thanks",
        "proposal_length": "short",
        "formality": "casual",
        "notes": "Be concise, highlight quick turnaround, use bullet points.",
    },
    "freelancer": {
        "tone": "professional",
        "greeting": "Dear",
        "closing": "Regards",
        "proposal_length": "medium",
        "formality": "formal",
        "notes": "Bid competitively, mention milestones, show understanding of requirements.",
    },
    "linkedin": {
        "tone": "professional",
        "greeting": "Hi",
        "closing": "Best",
        "proposal_length": "short",
        "formality": "semi-formal",
        "notes": "Keep messages brief, reference shared connections or interests when possible.",
    },
    "twitter": {
        "tone": "casual",
        "greeting": "Hey",
        "closing": "",
        "proposal_length": "very_short",
        "formality": "casual",
        "notes": "Be concise (280 chars), use hashtags sparingly, be conversational.",
    },
    "reddit": {
        "tone": "casual",
        "greeting": "Hey",
        "closing": "",
        "proposal_length": "short",
        "formality": "casual",
        "notes": "Match subreddit culture, be helpful first, avoid overt self-promotion.",
    },
    "discord": {
        "tone": "casual",
        "greeting": "Hey",
        "closing": "",
        "proposal_length": "short",
        "formality": "casual",
        "notes": "Use server-appropriate language, be direct, emojis are acceptable.",
    },
    "email": {
        "tone": "professional",
        "greeting": "Dear",
        "closing": "Best regards",
        "proposal_length": "medium",
        "formality": "formal",
        "notes": "Clear subject line, structured paragraphs, professional signature.",
    },
    "github": {
        "tone": "technical",
        "greeting": "Hi",
        "closing": "Thanks",
        "proposal_length": "medium",
        "formality": "technical",
        "notes": "Be specific about technical details, reference issue numbers, use markdown.",
    },
    "default": {
        "tone": "professional",
        "greeting": "Hello",
        "closing": "Best regards",
        "proposal_length": "medium",
        "formality": "formal",
        "notes": "Default to professional and courteous communication.",
    },
}

# ---------------------------------------------------------------------------
# Tone adaptation helpers
# ---------------------------------------------------------------------------

_FORMAL_REPLACEMENTS = [
    (r"\bhi\b", "Hello"),
    (r"\bhey\b", "Hello"),
    (r"\bthanks\b", "Thank you"),
    (r"\bthx\b", "Thank you"),
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bcant\b", "cannot"),
    (r"\bdont\b", "do not"),
    (r"\bwont\b", "will not"),
    (r"\bim\b", "I am"),
    (r"\bits\b", "it is"),
    (r"\byeah\b", "yes"),
    (r"\bnope\b", "no"),
    (r"\bkinda\b", "somewhat"),
    (r"\bsorta\b", "somewhat"),
    (r"\bgotta\b", "have to"),
]

_CASUAL_REPLACEMENTS = [
    (r"\bHello\b", "Hi"),
    (r"\bThank you\b", "Thanks"),
    (r"\bI am\b", "I'm"),
    (r"\bdo not\b", "don't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bDear\b", "Hey"),
]


def _apply_replacements(text: str, replacements: list) -> str:
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _adapt_to_formal(message: str) -> str:
    return _apply_replacements(message, _FORMAL_REPLACEMENTS)


def _adapt_to_casual(message: str) -> str:
    return _apply_replacements(message, _CASUAL_REPLACEMENTS)


def _adapt_to_technical(message: str) -> str:
    # Technical tone: formal but concise, no fluff
    message = _adapt_to_formal(message)
    # Remove filler phrases
    fillers = [
        r"\bI hope this message finds you well\.?\s*",
        r"\bI trust you are doing well\.?\s*",
        r"\bPlease feel free to\.?\s*",
    ]
    for filler in fillers:
        message = re.sub(filler, "", message, flags=re.IGNORECASE)
    return message.strip()


# ---------------------------------------------------------------------------
# Rapport opener helpers
# ---------------------------------------------------------------------------

_RAPPORT_OPENERS = {
    "new_client": [
        "It's great to connect with you, {name}.",
        "Thank you for reaching out, {name} — I'm excited about this opportunity.",
        "Hi {name}, I've reviewed your project and I'm confident we can achieve great results together.",
    ],
    "returning_client": [
        "Great to hear from you again, {name}!",
        "Welcome back, {name} — always a pleasure working with you.",
        "Hi {name}, glad to continue our collaboration.",
    ],
    "technical": [
        "Hi {name}, I've looked at the technical requirements and have some thoughts.",
        "Hello {name}, I've reviewed the specs and I'm ready to dive in.",
    ],
    "default": [
        "Hello {name}, thank you for this opportunity.",
        "Hi {name}, I look forward to working with you.",
    ],
}


def _pick_opener(client_name: str, context: str) -> str:
    ctx_lower = context.lower() if context else ""
    if "return" in ctx_lower or "again" in ctx_lower or "previous" in ctx_lower:
        openers = _RAPPORT_OPENERS["returning_client"]
    elif any(w in ctx_lower for w in ["technical", "code", "api", "bug", "develop"]):
        openers = _RAPPORT_OPENERS["technical"]
    elif "new" in ctx_lower or "first" in ctx_lower:
        openers = _RAPPORT_OPENERS["new_client"]
    else:
        openers = _RAPPORT_OPENERS["default"]
    # Simple deterministic pick based on name length to avoid randomness
    idx = len(client_name) % len(openers)
    return openers[idx].format(name=client_name)


# ---------------------------------------------------------------------------
# SocialAwareness class
# ---------------------------------------------------------------------------

class SocialAwareness:
    """
    Social Awareness module.

    Adapts communication tone, detects platform norms, and builds rapport.
    Optionally integrates HumanizedJanus for natural behavior patterns.
    """

    def __init__(
        self,
        gpt: Optional[Any] = None,
        humanized_janus: Optional[Any] = None,
    ) -> None:
        """
        Args:
            gpt: Optional JanusGPT instance.
            humanized_janus: Optional HumanizedJanus instance.
        """
        self._gpt = gpt
        self._humanized = humanized_janus
        if self._humanized is None and _HUMANIZATION_AVAILABLE:
            try:
                self._humanized = _HumanizedJanus(core=None)
            except Exception as exc:
                log.debug("Could not instantiate HumanizedJanus: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt_tone(self, message: str, context: Dict[str, Any]) -> str:
        """
        Adapt the tone of a message based on context.

        Args:
            message: The original message text.
            context: Dict with optional keys:
                     - "tone": "formal" | "casual" | "technical"
                     - "platform": platform name (used to infer tone if tone not given)
                     - "recipient_type": "client" | "colleague" | "public"

        Returns:
            Tone-adapted message string.
        """
        if not message:
            return message

        tone = context.get("tone", "").lower()
        if not tone:
            platform = context.get("platform", "").lower()
            norms = _PLATFORM_NORMS.get(platform, _PLATFORM_NORMS["default"])
            tone = norms.get("tone", "professional")

        if self._gpt is not None:
            try:
                return self._adapt_tone_with_gpt(message, tone, context)
            except Exception as exc:
                log.warning("JanusGPT adapt_tone failed (%s), using heuristics", exc)

        return self._adapt_tone_heuristic(message, tone)

    def detect_professional_norms(self, platform: str) -> Dict[str, str]:
        """
        Return professional communication norms for a given platform.

        Args:
            platform: Platform name (e.g. "upwork", "linkedin", "email").

        Returns:
            Dict with keys: tone, greeting, closing, formality, notes, etc.
        """
        key = platform.lower().strip() if platform else "default"
        return dict(_PLATFORM_NORMS.get(key, _PLATFORM_NORMS["default"]))

    def build_rapport_opener(self, client_name: str, context: str) -> str:
        """
        Build a rapport-building opening line for a message.

        Args:
            client_name: The client's name.
            context: Brief context about the relationship or situation.

        Returns:
            A warm, professional opening sentence.
        """
        if not client_name:
            client_name = "there"

        if self._gpt is not None:
            try:
                return self._rapport_opener_with_gpt(client_name, context)
            except Exception as exc:
                log.warning("JanusGPT build_rapport_opener failed (%s), using template", exc)

        return _pick_opener(client_name, context)

    # ------------------------------------------------------------------
    # GPT-backed implementations
    # ------------------------------------------------------------------

    def _adapt_tone_with_gpt(
        self, message: str, tone: str, context: Dict[str, Any]
    ) -> str:
        platform = context.get("platform", "general")
        prompt = (
            f"Rewrite the following message in a {tone} tone suitable for {platform}. "
            f"Keep the same meaning.\nOriginal: {message[:400]}\nRewritten:"
        )
        result = self._gpt.generate(prompt, max_new=300, temperature=0.5)
        return result.strip() or self._adapt_tone_heuristic(message, tone)

    def _rapport_opener_with_gpt(self, client_name: str, context: str) -> str:
        prompt = (
            f"Write a single warm, professional opening sentence to start a message to {client_name}. "
            f"Context: {context[:200]}\nOpener:"
        )
        result = self._gpt.generate(prompt, max_new=80, temperature=0.7)
        line = result.strip().split("\n")[0].strip()
        return line if len(line) > 10 else _pick_opener(client_name, context)

    # ------------------------------------------------------------------
    # Heuristic fallback implementations
    # ------------------------------------------------------------------

    def _adapt_tone_heuristic(self, message: str, tone: str) -> str:
        if tone in ("formal", "professional"):
            return _adapt_to_formal(message)
        if tone == "casual":
            return _adapt_to_casual(message)
        if tone == "technical":
            return _adapt_to_technical(message)
        # semi-formal: light formalization
        return _adapt_to_formal(message)
