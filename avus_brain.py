"""
avus_brain.py
==============
Single entry point for all Janus components that need language generation.
Wraps the trained Avus model with a clean, consistent interface.

No API keys. No external services. Runs locally.

Usage:
    from avus_brain import JanusBrain

    brain = JanusBrain()
    brain.load()  # loads avus_instruct.pt if available, else avus.py random init

    answer = brain.ask("What should I focus on to grow revenue this month?")
    plan   = brain.plan("Launch a freelance writing service")
    decision = brain.decide("Should I raise my prices?", ["yes", "no", "not yet"])
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Set

import torch

REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Weight search order — prefer instruction-tuned, fall back to base
_WEIGHT_CANDIDATES = [
    REPO_ROOT / "avus_instruct.pt",
    REPO_ROOT / "avus_1b_weights.pt",
    REPO_ROOT / "weights" / "avus_instruct.pt",
]

_SPECIAL_TOKENS: Set[str] = {
    "<|startoftext|>", "<|endoftext|>",
    "[INST]", "[/INST]",
}


class _Tokenizer:
    def __init__(self):
        try:
            import tiktoken
        except ImportError:
            os.system("pip install tiktoken -q")
            import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text, allowed_special=_SPECIAL_TOKENS)

    def decode(self, tokens: List[int]) -> str:
        valid = [t for t in tokens if 0 <= t < self._enc.max_token_value]
        try:
            return self._enc.decode(valid)
        except Exception:
            return ""

    @property
    def vocab_size(self) -> int:
        return self._enc.max_token_value + 1


class JanusBrain:
    """
    The reasoning engine for Janus.
    Wraps Avus with instruction-following prompts.

    All CEO, planner, and agent components should use this
    instead of calling model.py directly.
    """

    def __init__(self, device: Optional[str] = None):
        self._model     = None
        self._tokenizer = None
        self._loaded    = False
        self._device    = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, weights_path: Optional[str] = None) -> bool:
        """
        Load Avus weights. Tries instruction-tuned weights first,
        falls back to base weights, then random init.
        Returns True on success.
        """
        from avus import Avus, AvusConfig

        self._tokenizer = _Tokenizer()

        config = AvusConfig(
            vocab_size  = self._tokenizer.vocab_size,
            dim         = 384,
            n_layers    = 6,
            n_heads     = 6,
            n_kv_heads  = 3,
            max_seq_len = 512,
            dropout     = 0.0,  # no dropout at inference
        )

        self._model = Avus(config).to(self._device)
        self._model.eval()

        # Find weights
        candidates = ([Path(weights_path)] if weights_path else []) + _WEIGHT_CANDIDATES
        loaded_from = None
        for p in candidates:
            if p.exists():
                try:
                    sd = torch.load(p, map_location=self._device)
                    sd = sd.get("model_state_dict", sd)
                    # Drop incompatible keys
                    model_keys = set(self._model.state_dict().keys())
                    sd = {k: v for k, v in sd.items() if k in model_keys}
                    self._model.load_state_dict(sd, strict=False)
                    loaded_from = p
                    break
                except Exception as e:
                    print(f"[JanusBrain] Could not load {p}: {e}")

        if loaded_from:
            print(f"[JanusBrain] Loaded weights from {loaded_from}")
        else:
            print("[JanusBrain] No weights found — using random init (run avus_instruction_trainer.py first)")

        total = sum(p.numel() for p in self._model.parameters())
        print(f"[JanusBrain] Ready | params={total:,} | device={self._device}")
        self._loaded = True
        return True

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        """Raw generation from a prompt string."""
        if not self._loaded:
            return self._stub(prompt)

        tokens = self._tokenizer.encode(prompt)
        idx    = torch.tensor([tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            out = self._model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        new_tokens = out[0][len(tokens):].tolist()
        return self._tokenizer.decode(new_tokens).strip()

    # ── High-level interfaces ─────────────────────────────────────────────────

    def ask(self, question: str, max_tokens: int = 300) -> str:
        """
        Ask Avus a question. Returns a plain text answer.
        This is the main interface for the CEO loop.
        """
        prompt = f"<|startoftext|>[INST] {question} [/INST]\n"
        return self.generate(prompt, max_new_tokens=max_tokens)

    def plan(self, goal: str, context: str = "", max_tokens: int = 400) -> str:
        """Generate a step-by-step plan for a goal."""
        ctx = f" Context: {context}" if context else ""
        question = f"Create a step-by-step plan to achieve: {goal}.{ctx}"
        return self.ask(question, max_tokens=max_tokens)

    def decide(self, situation: str, options: List[str], max_tokens: int = 300) -> str:
        """Evaluate options and recommend one."""
        opts = ", ".join(f'"{o}"' for o in options)
        question = (
            f"Given this situation: {situation}\n"
            f"Evaluate these options: {opts}\n"
            f"Which should I choose and why?"
        )
        return self.ask(question, max_tokens=max_tokens)

    def summarize(self, content: str, max_tokens: int = 200) -> str:
        """Summarize a block of text."""
        question = f"Summarize the following concisely:\n\n{content}"
        return self.ask(question, max_tokens=max_tokens)

    def solve(self, problem: str, max_tokens: int = 350) -> str:
        """Generate a solution to a problem."""
        question = f"I have this problem: {problem}\nWhat should I do? Think step by step."
        return self.ask(question, max_tokens=max_tokens)

    def reflect(self, completed: List[str], pending: List[str]) -> str:
        """Generate a status reflection and next-action recommendation."""
        done_str    = "\n".join(f"- {t}" for t in completed) or "- None"
        pending_str = "\n".join(f"- {t}" for t in pending)   or "- None"
        question = (
            f"Completed tasks:\n{done_str}\n\n"
            f"Pending tasks:\n{pending_str}\n\n"
            f"Give me a brief status report and recommend the single most important next action."
        )
        return self.ask(question, max_tokens=250)

    # ── Stub fallback (before training) ──────────────────────────────────────

    @staticmethod
    def _stub(prompt: str) -> str:
        """
        Returns a template response when no weights are loaded.
        Keeps the system functional while Avus is being trained.
        """
        p = prompt.lower()
        if "plan" in p:
            return ("1. Define the goal clearly.\n"
                    "2. Break it into 3-5 concrete steps.\n"
                    "3. Execute the first step today.\n"
                    "4. Review progress daily.\n"
                    "5. Adjust based on results.")
        if "decide" in p or "choose" in p or "option" in p:
            return ("Evaluate each option on: risk, upside, and time to result. "
                    "Choose the option with the best risk-adjusted return. "
                    "Start with the lowest-risk option to build momentum.")
        if "revenue" in p or "money" in p or "earn" in p:
            return ("Focus on: 1) Delivering value to existing clients first. "
                    "2) Asking for referrals. 3) Adding one new service. "
                    "Track revenue weekly and double down on what works.")
        if "problem" in p or "issue" in p or "fix" in p:
            return ("1. Identify the root cause (not just symptoms). "
                    "2. Implement a quick fix to stop the bleeding. "
                    "3. Build a permanent solution. "
                    "4. Document to prevent recurrence.")
        return ("Understood. Analyzing the situation and preparing a response. "
                "For best results, run avus_instruction_trainer.py to train Avus "
                "on instruction-following data.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"JanusBrain(device={self._device}, status={status})"


# ── Module-level singleton ────────────────────────────────────────────────────

_brain: Optional[JanusBrain] = None

def get_brain() -> JanusBrain:
    """Get or create the global JanusBrain instance."""
    global _brain
    if _brain is None:
        _brain = JanusBrain()
        _brain.load()
    return _brain


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    brain = JanusBrain()
    brain.load()

    tests = [
        "What should I focus on to grow revenue this month?",
        "Create a plan to launch a freelance writing service.",
        "I have a problem: my biggest client just cancelled. What do I do?",
    ]

    for q in tests:
        print(f"\nQ: {q}")
        print(f"A: {brain.ask(q)}")
