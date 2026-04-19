"""
janus_inference_pipeline.py
============================
Wires janus_gpt.py (JanusGPT) into WorkGenerator so actual model output
becomes job deliverables instead of templates.

Drop-in replacement for WorkGenerator._generate_with_avus().
Also tries AvusBrain, then JanusGPT, then falls back to templates.

Usage (automatic — WorkGenerator imports this at init):
    from janus_inference_pipeline import InferencePipeline
    pipeline = InferencePipeline()
    work = await pipeline.generate(prompt, max_tokens=1500)
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Weight search paths ───────────────────────────────────────────────────────
_WEIGHT_DIRS = [
    "my-llm-project/weights",
    "weights",
    "models",
    "avus_3b_weights.pt",   # single-file fallback
]

_SUMMARY_NAMES = [
    "janus_training_summary.json",
    "config.json",
]


def _find_weights() -> Optional[tuple]:
    """
    Search common locations for JanusGPT weights.
    Returns (weights_path, summary_path) or None.
    """
    for d in _WEIGHT_DIRS:
        p = Path(d)
        # Single .pt file
        if p.suffix == ".pt" and p.exists():
            return str(p), None
        # Directory with weights
        if p.is_dir():
            for fname in ("janus_best.pt", "janus_final.pt", "model.pt"):
                wp = p / fname
                if wp.exists():
                    sp = None
                    for sname in _SUMMARY_NAMES:
                        candidate = p / sname
                        if candidate.exists():
                            sp = str(candidate)
                            break
                    return str(wp), sp
    return None


class InferencePipeline:
    """
    Unified inference pipeline for Janus work generation.

    Priority order:
      1. AvusBrain (existing system)
      2. JanusGPT  (janus_gpt.py — local transformer)
      3. Template  (always available fallback)
    """

    def __init__(self) -> None:
        self._avus: Optional[object] = None
        self._janus_gpt: Optional[object] = None
        self._active_backend: str = "template"
        self._init()

    def _init(self) -> None:
        """Try to load backends in priority order."""
        # 1. AvusBrain
        try:
            from avus_brain import AvusBrain  # type: ignore
            self._avus = AvusBrain()
            self._active_backend = "avus"
            logger.info("[InferencePipeline] AvusBrain loaded — using as primary backend")
            return
        except Exception as e:
            logger.debug("[InferencePipeline] AvusBrain unavailable: %s", e)

        # 2. JanusGPT
        try:
            from janus_gpt import load_janus_brain, JanusGPT  # type: ignore
            result = _find_weights()
            if result:
                weights_path, summary_path = result
                self._janus_gpt = load_janus_brain.__func__ if hasattr(load_janus_brain, '__func__') else None
                # Direct load
                from janus_gpt import JanusGPT, JanusConfig  # type: ignore
                if summary_path:
                    config = JanusConfig.from_summary(summary_path)
                else:
                    config = JanusConfig()
                model = JanusGPT(config)
                import torch
                state = torch.load(weights_path, map_location="cpu")
                if isinstance(state, dict):
                    sd = (state.get("model_state_dict") or state.get("model")
                          or state.get("state_dict") or state)
                else:
                    sd = state
                cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
                model.load_state_dict(cleaned, strict=False)
                model.eval()
                self._janus_gpt = model
                self._active_backend = "janus_gpt"
                logger.info("[InferencePipeline] JanusGPT loaded from %s", weights_path)
            else:
                logger.info("[InferencePipeline] JanusGPT weights not found — using template fallback")
        except Exception as e:
            logger.debug("[InferencePipeline] JanusGPT unavailable: %s", e)

        if self._active_backend == "template":
            logger.info("[InferencePipeline] No AI backend available — template mode only")

    @property
    def backend(self) -> str:
        """Name of the active inference backend."""
        return self._active_backend

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Optional[str]:
        """
        Generate text from a prompt using the best available backend.

        Returns None only if all backends fail (template never fails).
        """
        t0 = time.time()

        # ── AvusBrain ─────────────────────────────────────────────────────────
        if self._avus is not None:
            try:
                result = await self._call_avus(prompt, max_tokens)
                if result and len(result.strip()) > 50:
                    logger.info(
                        "[InferencePipeline] avus generated %d chars in %.1fs",
                        len(result), time.time() - t0,
                    )
                    return result
            except Exception as e:
                logger.warning("[InferencePipeline] AvusBrain error: %s — falling back", e)

        # ── JanusGPT ──────────────────────────────────────────────────────────
        if self._janus_gpt is not None:
            try:
                result = await asyncio.to_thread(
                    self._janus_gpt.generate,
                    prompt,
                    max_tokens,
                    temperature,
                    top_k,
                )
                if result and len(result.strip()) > 50:
                    logger.info(
                        "[InferencePipeline] janus_gpt generated %d chars in %.1fs",
                        len(result), time.time() - t0,
                    )
                    return result
            except Exception as e:
                logger.warning("[InferencePipeline] JanusGPT error: %s — falling back", e)

        # ── Template fallback ─────────────────────────────────────────────────
        logger.debug("[InferencePipeline] using template fallback")
        return None  # caller handles template generation

    async def _call_avus(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call AvusBrain with the prompt, handling sync and async variants."""
        model = self._avus
        if hasattr(model, "generate"):
            fn = model.generate
            if asyncio.iscoroutinefunction(fn):
                return await fn(prompt, max_length=max_tokens)
            return await asyncio.to_thread(fn, prompt, max_length=max_tokens)
        if hasattr(model, "inference"):
            fn = model.inference
            if asyncio.iscoroutinefunction(fn):
                return await fn(prompt)
            return await asyncio.to_thread(fn, prompt)
        # Last resort: call the model directly
        return await asyncio.to_thread(model, prompt)

    def get_embedding(self, text: str):
        """
        Return a text embedding from JanusGPT if available, else None.
        Used by the feedback loop for semantic job matching.
        """
        if self._janus_gpt is not None and hasattr(self._janus_gpt, "get_embedding"):
            try:
                return self._janus_gpt.get_embedding(text)
            except Exception as e:
                logger.debug("[InferencePipeline] get_embedding error: %s", e)
        return None

    def status(self) -> dict:
        return {
            "backend":       self._active_backend,
            "avus_loaded":   self._avus is not None,
            "gpt_loaded":    self._janus_gpt is not None,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Return the module-level InferencePipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline
