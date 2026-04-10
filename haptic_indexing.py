"""
haptic_indexing.py
==================
Tactile Metadata layer for Janus's Holographic Brain Memory.

Adds a "physical" property to every memory so the retriever doesn't
just look for keywords — it feels for structural properties.

Concepts implemented:
  - TactileSignature   : the "feel" of a memory (hardness, finish,
                         malleability, crystallization state)
  - HapticIndexer      : stamps every write with a tactile signature
                         and geometric anchor to prevent semantic collision
  - HapticRetriever    : filters results by grip — only returns memories
                         whose tactile signature matches the query's grip
  - ConsolidationEngine: hardens/melts memories over time.
                         Verified facts crystallize. Speculative thoughts
                         stay fluid. Old facts can be remelted for update.
  - HapticMemory       : drop-in wrapper around InfiniteJanusMemory that
                         adds the full haptic layer transparently

Integration into janus_memory_integration.py:
    from haptic_indexing import HapticMemory
    self._mem = HapticMemory(dim=2048)
    # All existing API calls work identically — haptic layer is invisible
    # unless you explicitly use grip-filtered retrieval:
    results = self._mem.recall_with_grip(query, grip="solid")
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tactile enums and signature
# ─────────────────────────────────────────────────────────────────────────────

class Hardness(float, Enum):
    """How structurally solid the memory is."""
    LIQUID      = 0.0   # raw, unprocessed, just arrived
    SOFT        = 0.25  # speculative, creative, uncertain
    SEMI_LIQUID = 0.45  # background knowledge, loosely verified
    FIRM        = 0.65  # working knowledge, frequently used
    HARD        = 0.85  # well-established fact
    CRYSTALLINE = 1.0   # fully verified, long-term potentiated


class Finish(str, Enum):
    """Surface texture — distinguishes similar-feeling memories."""
    ROUGH     = "rough"     # unprocessed, raw input
    GRAINY    = "grainy"    # partially processed
    MATTE     = "matte"     # standard processed memory
    POLISHED  = "polished"  # verified, cross-referenced
    MIRROR    = "mirror"    # ground truth, immutable


class Malleability(float, Enum):
    """How easily the memory can be updated."""
    RIGID     = 0.0   # cannot be changed (moral core, identity)
    STIFF     = 0.25  # requires strong evidence to update
    NORMAL    = 0.5   # standard update resistance
    FLEXIBLE  = 0.75  # updates easily with new info
    FLUID     = 1.0   # overwrites freely (working context)


@dataclass
class TactileSignature:
    """The feel of a memory. Stamped at write time, checked at retrieval time."""
    hardness:      float = Hardness.SEMI_LIQUID
    finish:        str   = Finish.MATTE
    malleability:  float = Malleability.NORMAL
    crystallized:  bool  = False
    created_at:    float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count:  int   = 0
    verified:      bool  = False
    source:        str   = "unknown"

    def feel_score(self) -> float:
        finish_scores = {
            Finish.ROUGH: 0.1, Finish.GRAINY: 0.3,
            Finish.MATTE: 0.5, Finish.POLISHED: 0.8, Finish.MIRROR: 1.0
        }
        fs = finish_scores.get(self.finish, 0.5)
        return (self.hardness * 0.5 + fs * 0.3 + (1.0 - self.malleability) * 0.2)

    def grip_matches(self, grip: str) -> bool:
        score = self.feel_score()
        if grip == "any":      return True
        if grip == "solid":    return score >= 0.6
        if grip == "soft":     return score < 0.4
        if grip == "verified": return self.verified or self.finish == Finish.MIRROR
        if grip == "fluid":    return self.malleability >= Malleability.FLEXIBLE
        if grip == "polished": return self.finish in (Finish.POLISHED, Finish.MIRROR)
        return True

    def to_dict(self) -> dict:
        return {
            "hardness":      self.hardness,
            "finish":        self.finish,
            "malleability":  self.malleability,
            "crystallized":  self.crystallized,
            "created_at":    self.created_at,
            "last_accessed": self.last_accessed,
            "access_count":  self.access_count,
            "verified":      self.verified,
            "source":        self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> "TactileSignature":
        return TactileSignature(**d)

    @staticmethod
    def for_fact(source: str = "user") -> "TactileSignature":
        return TactileSignature(hardness=Hardness.HARD, finish=Finish.POLISHED,
                                malleability=Malleability.STIFF, verified=True, source=source)

    @staticmethod
    def for_inference() -> "TactileSignature":
        return TactileSignature(hardness=Hardness.SOFT, finish=Finish.GRAINY,
                                malleability=Malleability.FLEXIBLE, verified=False, source="inference")

    @staticmethod
    def for_observation() -> "TactileSignature":
        return TactileSignature(hardness=Hardness.FIRM, finish=Finish.MATTE,
                                malleability=Malleability.NORMAL, verified=False, source="observation")

    @staticmethod
    def for_working_context() -> "TactileSignature":
        return TactileSignature(hardness=Hardness.LIQUID, finish=Finish.ROUGH,
                                malleability=Malleability.FLUID, verified=False, source="working")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Geometric anchor
# ─────────────────────────────────────────────────────────────────────────────

class GeometricAnchor:
    @staticmethod
    def compute_anchor_offset(key: str, signature: TactileSignature, dim: int) -> np.ndarray:
        anchor_seed = hashlib.sha256(
            f"{key}:{signature.finish}:{signature.source}:{float(signature.hardness):.2f}".encode()
        ).digest()
        seed_int = int.from_bytes(anchor_seed[:8], "big") % (2**31)
        rng = np.random.default_rng(seed_int)
        magnitude = 0.05 + signature.malleability * 0.15
        return rng.uniform(-magnitude, magnitude, size=(dim,)).astype(np.float32)

    @staticmethod
    def anchored_theta(base_theta: np.ndarray, key: str,
                       signature: TactileSignature, dim: int) -> np.ndarray:
        offset = GeometricAnchor.compute_anchor_offset(key, signature, dim)
        return (base_theta + offset) % (2 * np.pi)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Consolidation engine
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    HARDEN_THRESHOLD   = 5
    SOFTEN_AFTER_HOURS = 48.0
    HARDEN_RATE        = 0.08
    SOFTEN_RATE        = 0.005

    def consolidate(self, sig: TactileSignature) -> TactileSignature:
        if sig.crystallized:
            return sig
        now = time.time()
        hours_since_access = (now - sig.last_accessed) / 3600.0
        new_hardness = sig.hardness
        if sig.access_count >= self.HARDEN_THRESHOLD:
            excess = sig.access_count - self.HARDEN_THRESHOLD
            new_hardness = min(Hardness.CRYSTALLINE, new_hardness + excess * self.HARDEN_RATE)
        if hours_since_access > self.SOFTEN_AFTER_HOURS:
            decay = (hours_since_access - self.SOFTEN_AFTER_HOURS) * self.SOFTEN_RATE
            floor = Hardness.FIRM if sig.verified else Hardness.LIQUID
            new_hardness = max(floor, new_hardness - decay)
        new_finish = sig.finish
        if new_hardness >= Hardness.CRYSTALLINE and not sig.crystallized:
            new_finish = Finish.POLISHED
        elif new_hardness >= Hardness.HARD and sig.finish == Finish.MATTE:
            new_finish = Finish.POLISHED
        sig.hardness = new_hardness
        sig.finish   = new_finish
        return sig

    def crystallize(self, sig: TactileSignature) -> TactileSignature:
        sig.hardness     = Hardness.CRYSTALLINE
        sig.finish       = Finish.MIRROR
        sig.malleability = Malleability.RIGID
        sig.crystallized = True
        sig.verified     = True
        return sig

    def melt(self, sig: TactileSignature,
             target_hardness: float = Hardness.SEMI_LIQUID) -> TactileSignature:
        sig.crystallized = False
        sig.hardness     = min(sig.hardness, target_hardness)
        sig.finish       = Finish.MATTE
        sig.malleability = Malleability.NORMAL
        return sig

    def on_access(self, sig: TactileSignature) -> TactileSignature:
        sig.last_accessed = time.time()
        sig.access_count += 1
        return self.consolidate(sig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Haptic Index
# ─────────────────────────────────────────────────────────────────────────────

class HapticIndex:
    def __init__(self):
        self._index:       Dict[str, TactileSignature] = {}
        self._anchors:     Dict[str, np.ndarray]        = {}
        self._consolidator = ConsolidationEngine()

    def stamp(self, key: str, signature: TactileSignature,
              base_theta: np.ndarray, dim: int) -> np.ndarray:
        self._index[key] = signature
        anchored = GeometricAnchor.anchored_theta(base_theta, key, signature, dim)
        self._anchors[key] = anchored
        return anchored

    def get_signature(self, key: str) -> Optional[TactileSignature]:
        sig = self._index.get(key)
        if sig:
            sig = self._consolidator.on_access(sig)
            self._index[key] = sig
        return sig

    def get_anchored_theta(self, key: str) -> Optional[np.ndarray]:
        return self._anchors.get(key)

    def filter_by_grip(self, keys: List[str], grip: str) -> List[str]:
        result = []
        for k in keys:
            sig = self._index.get(k)
            if sig and sig.grip_matches(grip):
                result.append(k)
        return result

    def crystallize(self, key: str) -> bool:
        if key not in self._index:
            return False
        self._index[key] = self._consolidator.crystallize(self._index[key])
        return True

    def melt(self, key: str) -> bool:
        if key not in self._index:
            return False
        self._index[key] = self._consolidator.melt(self._index[key])
        return True

    def status(self) -> dict:
        total        = len(self._index)
        crystallized = sum(1 for s in self._index.values() if s.crystallized)
        hard         = sum(1 for s in self._index.values()
                          if s.hardness >= Hardness.HARD and not s.crystallized)
        soft         = sum(1 for s in self._index.values() if s.hardness < Hardness.FIRM)
        return {
            "total_indexed": total,
            "crystallized":  crystallized,
            "hard":          hard,
            "soft":          soft,
            "fluid":         total - crystallized - hard - soft,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. HapticMemory
# ─────────────────────────────────────────────────────────────────────────────

class HapticMemory:
    """
    Drop-in replacement for InfiniteJanusMemory with haptic indexing.

    Grip options for recall_with_grip:
      'any'      — no filter
      'solid'    — only hard/crystalline memories (feel_score >= 0.6)
      'soft'     — only speculative/fluid memories
      'verified' — only memories marked verified=True or mirror finish
      'fluid'    — only easily-updatable memories
      'polished' — polished or mirror finish only
    """

    def __init__(self, dim: int = 2048, decay: float = 0.97):
        try:
            from holographic_memory import InfiniteJanusMemory
            self._mem = InfiniteJanusMemory(dim=dim, decay=decay)
        except ImportError:
            self._mem = _FallbackMemory()
        self._haptic = HapticIndex()
        self._dim    = dim

    def add_episode(self, task_id: str, action: str, result: Any, success: bool,
                    signature: Optional[TactileSignature] = None):
        self._mem.add_episode(task_id, action, result, success)
        sig   = signature or (TactileSignature.for_fact("episode") if success
                              else TactileSignature.for_observation())
        theta = _make_theta_local(hash(task_id) % (2**31), self._dim)
        self._haptic.stamp(task_id, sig, theta, self._dim)

    def update_semantic(self, key: str, value: Any, confidence: float = 1.0,
                        signature: Optional[TactileSignature] = None):
        self._mem.update_semantic(key, value, confidence)
        sig   = signature or (TactileSignature.for_fact() if confidence >= 0.8
                              else TactileSignature.for_inference())
        seed  = int(hashlib.sha256(key.encode()).digest()[:8].hex(), 16) % (2**31)
        theta = _make_theta_local(seed, self._dim)
        self._haptic.stamp(key, sig, theta, self._dim)

    def update_semantic_verified(self, key: str, value: Any, source: str = "user"):
        self.update_semantic(key, value, confidence=1.0,
                             signature=TactileSignature.for_fact(source=source))

    def update_semantic_inference(self, key: str, value: Any):
        self.update_semantic(key, value, confidence=0.5,
                             signature=TactileSignature.for_inference())

    def get_semantic(self, key: str) -> Optional[dict]:
        result = self._mem.get_semantic(key)
        if result:
            sig = self._haptic.get_signature(key)
            if sig:
                result["tactile"]    = sig.to_dict()
                result["feel_score"] = sig.feel_score()
        return result

    def set_working_context(self, key: str, value: Any):
        self._mem.set_working_context(key, value)
        sig   = TactileSignature.for_working_context()
        seed  = int(hashlib.sha256(f"wc:{key}".encode()).digest()[:8].hex(), 16) % (2**31)
        theta = _make_theta_local(seed, self._dim)
        self._haptic.stamp(f"wc:{key}", sig, theta, self._dim)

    def get_working_context(self, key: str) -> Any:
        return self._mem.get_working_context(key)

    def recall_with_grip(self, query: str, grip: str = "solid",
                         top_k: int = 5) -> List[dict]:
        all_results = self._mem.recall_similar(query, top_k=top_k * 3)
        filtered = []
        for episode in all_results:
            task_id = episode.get("task_id", "")
            sig     = self._haptic.get_signature(task_id)
            if sig is None or sig.grip_matches(grip):
                if sig:
                    episode["tactile"]    = sig.to_dict()
                    episode["feel_score"] = sig.feel_score()
                filtered.append(episode)
                if len(filtered) >= top_k:
                    break
        return filtered

    def search_semantic_with_grip(self, query: str, grip: str = "solid",
                                  top_k: int = 5) -> List[Tuple[str, float]]:
        all_results = self._mem.search_semantic(query, top_k=top_k * 3)
        filtered = []
        for key, score in all_results:
            sig = self._haptic.get_signature(key)
            if sig is None or sig.grip_matches(grip):
                filtered.append((key, score))
                if len(filtered) >= top_k:
                    break
        return filtered

    def crystallize_key(self, key: str) -> bool:
        return self._haptic.crystallize(key)

    def melt_key(self, key: str) -> bool:
        return self._haptic.melt(key)

    def haptic_status(self) -> dict:
        base = self._mem.status()
        base["haptic"] = self._haptic.status()
        return base

    def recall_similar(self, query: str, top_k: int = 5) -> List[dict]:
        return self._mem.recall_similar(query, top_k)

    def search_semantic(self, query: str, top_k: int = 5):
        return self._mem.search_semantic(query, top_k)

    def get_recent_episodes(self, n: int = 10) -> List[dict]:
        return self._mem.get_recent_episodes(n)

    def status(self) -> dict:
        return self.haptic_status()

    @property
    def item_count(self) -> int:
        return self._mem.item_count

    @property
    def byte_size(self) -> int:
        return self._mem.byte_size


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_seed_local(text: str) -> int:
    h = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _make_theta_local(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0, 2 * np.pi, size=(dim,)).astype(np.float32)
    if _TORCH:
        return torch.tensor(arr)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Fallback memory
# ─────────────────────────────────────────────────────────────────────────────

class _FallbackMemory:
    def __init__(self):
        self._ep = []; self._sem = {}; self._wc = {}
        self.item_count = 0; self.byte_size = 0

    def add_episode(self, task_id, action, result, success):
        self._ep.append({"task_id": task_id, "action": action,
                         "result": str(result), "success": success})
        self.item_count += 1

    def update_semantic(self, key, value, confidence=1.0):
        self._sem[key] = {"value": value, "confidence": confidence}
        self.item_count += 1

    def get_semantic(self, key):        return self._sem.get(key)
    def set_working_context(self, k, v): self._wc[k] = v
    def get_working_context(self, k):   return self._wc.get(k)
    def recall_similar(self, q, top_k=5): return self._ep[-top_k:]
    def search_semantic(self, q, top_k=5): return list(self._sem.items())[:top_k]
    def get_recent_episodes(self, n=10): return self._ep[-n:]
    def status(self): return {"mode": "fallback", "items": self.item_count}


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HapticMemory...\n")
    mem = HapticMemory(dim=256)

    mem.update_semantic_verified("avus_loss", "0.2253", source="training")
    mem.update_semantic_verified("engine_status", "all systems passing")
    mem.crystallize_key("avus_loss")

    mem.update_semantic_inference("user_mood", "focused today")
    mem.update_semantic_inference("next_goal_estimate", "Phase 2 training soon")
    mem.set_working_context("current_task", "building haptic memory")

    mem.add_episode("train_001", "train_epoch", "loss=0.2253", success=True,
                    signature=TactileSignature.for_fact("kaggle"))
    mem.add_episode("infer_001", "generate_asset", "dungeon_pillar", success=True,
                    signature=TactileSignature.for_observation())
    mem.add_episode("fail_001", "connect_db", "timeout", success=False,
                    signature=TactileSignature.for_inference())

    print("── Semantic: avus_loss ──")
    r = mem.get_semantic("avus_loss")
    if r:
        t = r.get("tactile", {})
        print(f"  feel_score:   {r.get('feel_score', 0):.3f}")
        print(f"  hardness:     {float(t.get('hardness', 0)):.2f}")
        print(f"  finish:       {t.get('finish')}")
        print(f"  crystallized: {t.get('crystallized')}")

    print("\n── Grip: verified ──")
    for key, score in mem.search_semantic_with_grip("avus", grip="verified", top_k=5):
        print(f"  {key}")

    print("\n── Grip: soft ──")
    for key, score in mem.search_semantic_with_grip("goal", grip="soft", top_k=5):
        print(f"  {key}")

    print("\n── Haptic status ──")
    for k, v in mem.haptic_status().get("haptic", {}).items():
        print(f"  {k}: {v}")

    print("\nDone.")
