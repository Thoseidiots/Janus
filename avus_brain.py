"""
avus_brain.py
=============
Bridge between AvusInference and the rest of Janus.

Provides the get_brain() / ask() / summarize() interface that
janus_video_comprehension.py (and any other module) calls.

All reasoning routes through the locally-trained Avus model.
No API keys. No external services.

Usage:
    from avus_brain import get_brain

    brain = get_brain()
    answer  = brain.ask("What were the main steps shown?")
    summary = brain.summarize("Long block of observations...")
    code    = brain.generate_code("A health regen system in Python")
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional

from avus_inference import AvusInference


# ─────────────────────────────────────────────────────────────────────────────
# AvusBrain
# ─────────────────────────────────────────────────────────────────────────────

class AvusBrain:
    """
    High-level reasoning interface backed by AvusInference.

    Wraps the raw generate() call with task-specific prompt templates
    so callers don't need to know about tokens, temperatures, or tags.
    """

    def __init__(self, avus: Optional[AvusInference] = None):
        self._avus = avus or AvusInference()
        self._loaded = False

    def ensure_loaded(self) -> bool:
        """Load Avus weights on first use (lazy init)."""
        if not self._loaded:
            self._loaded = self._avus.load()
        return self._loaded

    # ── Core interface ────────────────────────────────────────────────────────

    def ask(self, question: str, context: str = "", max_tokens: int = 300,
            temperature: float = 0.7) -> str:
        """
        Answer a question, optionally with supporting context.

        Args:
            question:    The question to answer
            context:     Optional background text (observations, notes, etc.)
            max_tokens:  Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Answer string
        """
        self.ensure_loaded()

        if context:
            prompt = (
                f"Context:\n{_trim(context, 800)}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        raw = self._avus.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return _clean(raw)

    def summarize(self, text: str, max_tokens: int = 200,
                  temperature: float = 0.6) -> str:
        """
        Summarize a block of text.

        Args:
            text:        Text to summarize
            max_tokens:  Max tokens for the summary
            temperature: Sampling temperature

        Returns:
            Summary string
        """
        self.ensure_loaded()

        prompt = (
            f"Summarize the following in 2-4 sentences:\n\n"
            f"{_trim(text, 1000)}\n\n"
            f"Summary:"
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return _clean(raw)

    def extract_steps(self, text: str, max_tokens: int = 250) -> list[str]:
        """
        Extract ordered action steps from a block of text.

        Args:
            text: Observations or instructions to parse

        Returns:
            List of step strings
        """
        self.ensure_loaded()

        prompt = (
            f"Extract the step-by-step instructions from the following. "
            f"Number each step.\n\n"
            f"{_trim(text, 800)}\n\n"
            f"Steps:\n1."
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.5,
        )

        # Parse numbered list — handle both "1." and "1)" formats
        steps = re.findall(r'(?:^|\n)\s*\d+[.)]\s*(.+)', "1." + raw)
        if not steps:
            # Fallback: split on newlines and clean up
            steps = [s.strip() for s in raw.split('\n') if s.strip()]
        return steps[:10]

    def extract_concepts(self, text: str, n: int = 5) -> list[str]:
        """
        Extract key concepts from a block of text.

        Args:
            text: Source text
            n:    Number of concepts to extract

        Returns:
            List of concept strings
        """
        self.ensure_loaded()

        prompt = (
            f"List the {n} most important concepts from the following text. "
            f"Use bullet points.\n\n"
            f"{_trim(text, 800)}\n\n"
            f"Key concepts:\n-"
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=150,
            temperature=0.5,
        )

        concepts = re.findall(r'(?:^|\n)\s*[-•*]\s*(.+)', "-" + raw)
        if not concepts:
            concepts = [s.strip() for s in raw.split('\n') if s.strip()]
        return concepts[:n]

    def generate_code(self, description: str, language: str = "python",
                      max_tokens: int = 400) -> str:
        """
        Generate code from a natural language description.

        Args:
            description: What the code should do
            language:    Target language
            max_tokens:  Max tokens to generate

        Returns:
            Generated code string
        """
        self.ensure_loaded()

        prompt = (
            f"Write {language} code that does the following:\n"
            f"{description}\n\n"
            f"```{language}\n"
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.6,
        )

        # Strip closing fence if present
        if "```" in raw:
            raw = raw[:raw.index("```")]
        return raw.strip()

    def reflect(self, observations: list[str], max_tokens: int = 200) -> str:
        """
        Generate a self-reflection from a list of recent observations.

        Args:
            observations: Recent events or memory entries
            max_tokens:   Max tokens for reflection

        Returns:
            Reflection string
        """
        self.ensure_loaded()

        joined = "\n".join(f"- {o}" for o in observations[-15:])
        prompt = (
            f"Recent observations:\n{joined}\n\n"
            f"Reflection on what was learned and what to do next:"
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.75,
        )
        return _clean(raw)

    def classify_content(self, description: str) -> str:
        """
        Classify what type of content a video frame or scene shows.

        Returns one of: instruction, demo, transition, observation
        """
        self.ensure_loaded()

        prompt = (
            f"Classify this video frame description into one of: "
            f"instruction, demo, transition, observation.\n\n"
            f"Description: {description}\n\n"
            f"Classification:"
        )

        raw = self._avus.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.3,
        ).strip().lower()

        for label in ("instruction", "demo", "transition", "observation"):
            if label in raw:
                return label
        return "observation"

    # ── Pass-through to raw Avus ──────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        """Direct pass-through to AvusInference.generate()."""
        self.ensure_loaded()
        return self._avus.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"AvusBrain(loaded={self._loaded}, device={self._avus.device})"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trim(text: str, max_chars: int) -> str:
    """Trim text to max_chars, preserving whole words."""
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and special tokens from output."""
    text = text.strip()
    # Remove any leaked special tokens
    for tok in ("<|endoftext|>", "<|startoftext|>", "[JSON_END]", "[ACT_END]"):
        text = text.replace(tok, "")
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_brain: Optional[AvusBrain] = None


def get_brain() -> AvusBrain:
    """
    Return the shared AvusBrain instance.
    Loads Avus weights on first call (lazy init).
    """
    global _brain
    if _brain is None:
        _brain = AvusBrain()
    return _brain


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    brain = get_brain()
    print(f"Brain: {brain}")

    print("\n── ask ─────────────────────────────────────────────────────")
    print(brain.ask("What is a transformer neural network?", max_tokens=80))

    print("\n── summarize ───────────────────────────────────────────────")
    sample = (
        "[0s] Browser opens YouTube. [5s] Tutorial on Python decorators begins. "
        "[10s] Instructor explains @property. [20s] Code example shown. "
        "[30s] Output printed to terminal. [40s] Summary slide appears."
    )
    print(brain.summarize(sample))

    print("\n── extract_steps ───────────────────────────────────────────")
    steps = brain.extract_steps(sample)
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    print("\n── extract_concepts ────────────────────────────────────────")
    concepts = brain.extract_concepts(sample, n=3)
    for c in concepts:
        print(f"  - {c}")
