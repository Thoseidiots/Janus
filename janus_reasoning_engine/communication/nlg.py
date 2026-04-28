"""
janus_reasoning_engine/communication/nlg.py
============================================
Natural Language Generation for the Janus Reasoning Engine.

Provides:
  - NLG.write_proposal(job, skills, experience) -> str
  - NLG.compose_message(recipient, intent, context) -> str
  - NLG.generate_progress_report(goal, progress, completed_steps) -> str
  - NLG.ask_clarification(ambiguous_request) -> str

Uses JanusGPT when available, template fallback otherwise.
Integrates janus_email_composer.py optionally.

Requirements: REQ-7.2, REQ-8.5
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: janus_email_composer
# ---------------------------------------------------------------------------
try:
    from janus_email_composer import JanusEmailComposer as _JanusEmailComposer  # type: ignore
    _EMAIL_COMPOSER_AVAILABLE = True
except Exception:
    _JanusEmailComposer = None  # type: ignore
    _EMAIL_COMPOSER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Template fallbacks
# ---------------------------------------------------------------------------

def _proposal_template(job: str, skills: List[str], experience: str) -> str:
    skills_str = ", ".join(skills) if skills else "relevant skills"
    return (
        f"Dear Hiring Manager,\n\n"
        f"I am writing to express my interest in the following opportunity:\n\n"
        f"{job}\n\n"
        f"I bring expertise in {skills_str}. {experience}\n\n"
        f"I am confident I can deliver high-quality results on time and within budget. "
        f"I look forward to discussing this further.\n\n"
        f"Best regards,\nJanus"
    )


def _message_template(recipient: str, intent: str, context: str) -> str:
    intent_phrases = {
        "follow_up": "I wanted to follow up on our previous conversation",
        "introduction": "I am reaching out to introduce myself",
        "inquiry": "I have a question regarding",
        "update": "I wanted to provide you with an update",
        "thank_you": "I wanted to thank you for",
        "request": "I am writing to request",
    }
    opener = intent_phrases.get(intent, f"I am writing regarding {intent}")
    return (
        f"Dear {recipient},\n\n"
        f"{opener}. {context}\n\n"
        f"Please let me know if you have any questions or need further information.\n\n"
        f"Best regards,\nJanus"
    )


def _progress_report_template(goal: str, progress: float, completed_steps: List[str]) -> str:
    pct = int(progress * 100)
    steps_str = "\n".join(f"  ✓ {s}" for s in completed_steps) if completed_steps else "  (none yet)"
    return (
        f"Progress Report\n"
        f"===============\n"
        f"Goal: {goal}\n"
        f"Overall Progress: {pct}%\n\n"
        f"Completed Steps:\n{steps_str}\n\n"
        f"Status: {'Complete' if pct >= 100 else 'In Progress'}"
    )


def _clarification_template(ambiguous_request: str) -> str:
    return (
        f"I want to make sure I understand your request correctly.\n\n"
        f"You asked: \"{ambiguous_request}\"\n\n"
        f"Could you please clarify:\n"
        f"  1. What is the primary goal or outcome you are looking for?\n"
        f"  2. Are there any specific constraints (budget, timeline, format)?\n"
        f"  3. What does success look like for this task?\n\n"
        f"Thank you for helping me serve you better."
    )


# ---------------------------------------------------------------------------
# NLG class
# ---------------------------------------------------------------------------

class NLG:
    """
    Natural Language Generation module.

    Uses JanusGPT when available; falls back to professional templates.
    Optionally integrates JanusEmailComposer for email delivery.
    """

    def __init__(
        self,
        gpt: Optional[Any] = None,
        email_composer: Optional[Any] = None,
    ) -> None:
        """
        Args:
            gpt: Optional JanusGPT instance.
            email_composer: Optional JanusEmailComposer instance.
                            If None and the module is available, one is created lazily.
        """
        self._gpt = gpt
        self._email_composer = email_composer
        if self._email_composer is None and _EMAIL_COMPOSER_AVAILABLE:
            try:
                self._email_composer = _JanusEmailComposer()
            except Exception as exc:
                log.debug("Could not instantiate JanusEmailComposer: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_proposal(
        self,
        job: str,
        skills: List[str],
        experience: str,
    ) -> str:
        """
        Write a compelling job proposal.

        Args:
            job: Job title or description.
            skills: List of relevant skills to highlight.
            experience: Brief experience summary.

        Returns:
            Formatted proposal string.
        """
        if not job:
            return ""

        if self._gpt is not None:
            try:
                return self._proposal_with_gpt(job, skills, experience)
            except Exception as exc:
                log.warning("JanusGPT write_proposal failed (%s), using template", exc)

        return _proposal_template(job, skills, experience)

    def compose_message(
        self,
        recipient: str,
        intent: str,
        context: str,
    ) -> str:
        """
        Compose a professional message.

        Args:
            recipient: Name or role of the recipient.
            intent: Message intent (e.g. "follow_up", "introduction", "inquiry").
            context: Additional context or body content.

        Returns:
            Formatted message string.
        """
        if not recipient:
            recipient = "Hiring Manager"

        if self._gpt is not None:
            try:
                return self._message_with_gpt(recipient, intent, context)
            except Exception as exc:
                log.warning("JanusGPT compose_message failed (%s), using template", exc)

        return _message_template(recipient, intent, context)

    def generate_progress_report(
        self,
        goal: str,
        progress: float,
        completed_steps: List[str],
    ) -> str:
        """
        Generate a progress report for the owner.

        Args:
            goal: Description of the goal being tracked.
            progress: Completion fraction between 0.0 and 1.0.
            completed_steps: List of completed step descriptions.

        Returns:
            Formatted progress report string.
        """
        progress = max(0.0, min(1.0, progress))

        if self._gpt is not None:
            try:
                return self._progress_report_with_gpt(goal, progress, completed_steps)
            except Exception as exc:
                log.warning("JanusGPT generate_progress_report failed (%s), using template", exc)

        return _progress_report_template(goal, progress, completed_steps)

    def ask_clarification(self, ambiguous_request: str) -> str:
        """
        Generate a clarifying question for an ambiguous request.

        Args:
            ambiguous_request: The unclear or ambiguous instruction.

        Returns:
            Clarification question string.
        """
        if not ambiguous_request:
            return "Could you please provide more details about what you need?"

        if self._gpt is not None:
            try:
                return self._clarification_with_gpt(ambiguous_request)
            except Exception as exc:
                log.warning("JanusGPT ask_clarification failed (%s), using template", exc)

        return _clarification_template(ambiguous_request)

    # ------------------------------------------------------------------
    # GPT-backed implementations
    # ------------------------------------------------------------------

    def _proposal_with_gpt(self, job: str, skills: List[str], experience: str) -> str:
        skills_str = ", ".join(skills) if skills else "relevant skills"
        prompt = (
            f"Write a compelling, professional job proposal for the following:\n"
            f"Job: {job[:300]}\n"
            f"Skills: {skills_str}\n"
            f"Experience: {experience[:200]}\n"
            f"Proposal:\n"
        )
        result = self._gpt.generate(prompt, max_new=300, temperature=0.7)
        return result.strip() or _proposal_template(job, skills, experience)

    def _message_with_gpt(self, recipient: str, intent: str, context: str) -> str:
        prompt = (
            f"Write a professional message to {recipient}.\n"
            f"Intent: {intent}\n"
            f"Context: {context[:300]}\n"
            f"Message:\n"
        )
        result = self._gpt.generate(prompt, max_new=250, temperature=0.6)
        return result.strip() or _message_template(recipient, intent, context)

    def _progress_report_with_gpt(
        self, goal: str, progress: float, completed_steps: List[str]
    ) -> str:
        steps_str = "; ".join(completed_steps[:5]) if completed_steps else "none"
        pct = int(progress * 100)
        prompt = (
            f"Write a concise progress report.\n"
            f"Goal: {goal[:200]}\n"
            f"Progress: {pct}%\n"
            f"Completed: {steps_str}\n"
            f"Report:\n"
        )
        result = self._gpt.generate(prompt, max_new=200, temperature=0.5)
        return result.strip() or _progress_report_template(goal, progress, completed_steps)

    def _clarification_with_gpt(self, ambiguous_request: str) -> str:
        prompt = (
            f"The following request is ambiguous. Write a polite clarifying question:\n"
            f"Request: {ambiguous_request[:300]}\n"
            f"Clarification:\n"
        )
        result = self._gpt.generate(prompt, max_new=150, temperature=0.6)
        return result.strip() or _clarification_template(ambiguous_request)
