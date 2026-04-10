"""
janus_memory_integration.py
=============================
Wires InfiniteJanusMemory into the Janus system orchestrator.

This is the only file you need to touch to activate holographic memory
across the entire Janus stack. Import this instead of any previous
StructuredMemory or ReflectionMemory references.

Usage in janus_system_orchestrator.py:
    from janus_memory_integration import JanusMemory
    self.memory = JanusMemory()

    # Store a completed cycle
    self.memory.record_cycle(result)

    # Store a goal update
    self.memory.record_goal("Generate dungeon asset pack", success=True)

    # Store screen observation
    self.memory.record_observation(screen_description)

    # Recall similar past cycles
    similar = self.memory.recall_similar("dungeon asset generation")

    # Check memory health
    print(self.memory.status())
"""

import time
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

# ─────────────────────────────────────────────────────────────────────────────
# Import holographic memory — graceful fallback if torch not available
# ─────────────────────────────────────────────────────────────────────────────

try:
    from holographic_memory import (
        InfiniteJanusMemory,
        HolographicReflection,
    )
    _HBM_AVAILABLE = True
except ImportError:
    _HBM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Fallback memory (plain dict — used if torch not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _FallbackMemory:
    """Simple dict-based memory used when HBM is unavailable."""

    def __init__(self):
        self._episodes: List[Dict] = []
        self._semantic: Dict[str, Any] = {}
        self._working:  Dict[str, Any] = {}

    def add_episode(self, task_id, action, result, success):
        self._episodes.append({
            "task_id": task_id, "action": action,
            "result": str(result)[:256], "success": success,
            "timestamp": time.time()
        })

    def update_semantic(self, key, value, confidence=1.0):
        self._semantic[key] = {"value": value, "confidence": confidence}

    def get_semantic(self, key):
        return self._semantic.get(key)

    def set_working_context(self, key, value):
        self._working[key] = value

    def get_working_context(self, key):
        return self._working.get(key)

    def recall_similar(self, query, top_k=5):
        return self._episodes[-top_k:]

    def search_semantic(self, query, top_k=5):
        return list(self._semantic.items())[:top_k]

    def get_recent_episodes(self, n=10):
        return self._episodes[-n:]

    def status(self):
        return {
            "physical_bytes": "N/A (fallback mode)",
            "logical_items": len(self._episodes) + len(self._semantic),
            "episodic_events": len(self._episodes),
            "semantic_keys": len(self._semantic),
            "working_context_keys": len(self._working),
            "snr_estimate": "N/A",
            "memory_dim": "N/A",
            "mode": "fallback",
        }

    @property
    def item_count(self):
        return len(self._episodes)

    @property
    def byte_size(self):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# JanusMemory — unified interface used by the orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class JanusMemory:
    """
    Unified memory interface for the Janus system.

    Automatically uses InfiniteJanusMemory (holographic, fixed-size)
    if PyTorch is available, otherwise falls back to a plain dict store.

    Adds Janus-specific convenience methods on top of the HBM API:
        record_cycle(result)          — store a CycleResult
        record_goal(goal, success)    — store goal completion
        record_observation(desc)      — store screen observation
        recall_similar(query)         — find similar past cycles
        get_context()                 — get current working context
        save(path) / load(path)       — persist metadata to disk
    """

    def __init__(self, dim: int = 2048, decay: float = 0.97,
                 persist_path: Optional[str] = None):
        if _HBM_AVAILABLE:
            self._mem    = InfiniteJanusMemory(dim=dim, decay=decay)
            self._reflect= HolographicReflection(self._mem)
            self._mode   = "holographic"
            print(f"[JanusMemory] Holographic mode — "
                  f"{self._mem.byte_size // 1024}KB fixed footprint")
        else:
            self._mem    = _FallbackMemory()
            self._reflect= None
            self._mode   = "fallback"
            print("[JanusMemory] Fallback mode — install torch for holographic memory")

        self._persist_path = Path(persist_path) if persist_path else None
        self._cycle_count  = 0

        # Load persisted metadata if available
        if self._persist_path and self._persist_path.exists():
            self._load_meta()

    # ── Janus-specific record methods ─────────────────────────────────────────

    def record_cycle(self, result) -> None:
        """
        Store a CycleResult from the orchestrator.
        Accepts a CycleResult dataclass or any object with .to_dict().
        """
        self._cycle_count += 1

        if hasattr(result, "to_dict"):
            d = result.to_dict()
        elif isinstance(result, dict):
            d = result
        else:
            d = {"raw": str(result)}

        task_id = f"cycle_{d.get('cycle', self._cycle_count)}"
        action  = d.get("phase", "unknown")
        success = d.get("success", False)
        goal    = d.get("goal") or "no goal"
        notes   = " | ".join(d.get("notes", []))
        result_str = f"goal={goal} notes={notes}"

        self._mem.add_episode(task_id, action, result_str, success)

        # Update working context with latest cycle info
        self._mem.set_working_context("last_cycle", task_id)
        self._mem.set_working_context("last_goal", goal)
        self._mem.set_working_context("last_success", success)

        # Persist metadata periodically
        if self._cycle_count % 10 == 0:
            self._save_meta()

    def record_goal(self, goal: str, success: bool,
                    notes: str = "") -> None:
        """Store a goal completion event."""
        self._mem.add_episode(
            task_id=f"goal_{int(time.time())}",
            action="goal_complete" if success else "goal_failed",
            result=f"{goal} | {notes}",
            success=success,
        )
        # Update semantic memory with goal outcome
        key = f"goal_outcome_{goal[:64]}"
        self._mem.update_semantic(key,
                                  f"success={success} notes={notes}",
                                  confidence=0.9 if success else 0.5)

    def record_observation(self, screen_description: str) -> None:
        """Store a screen observation."""
        self._mem.add_episode(
            task_id=f"obs_{int(time.time()*1000) % 1_000_000}",
            action="observe",
            result=screen_description[:256],
            success=True,
        )
        # Keep latest observation in working context
        self._mem.set_working_context("last_observation", screen_description[:256])

    def record_asset(self, prompt: str, asset_name: str,
                     success: bool) -> None:
        """Store an asset generation event."""
        self._mem.add_episode(
            task_id=f"asset_{int(time.time())}",
            action="generate_asset",
            result=f"name={asset_name} prompt={prompt[:128]}",
            success=success,
        )
        if success:
            self._mem.update_semantic(
                f"asset_{asset_name}", prompt, confidence=1.0)

    def update_goal_knowledge(self, key: str, value: Any,
                               confidence: float = 1.0) -> None:
        """Store a semantic fact about Janus's world knowledge."""
        self._mem.update_semantic(key, value, confidence)

    # ── Recall methods ────────────────────────────────────────────────────────

    def recall_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find past cycles/observations similar to a query."""
        return self._mem.recall_similar(query, top_k)

    def search_knowledge(self, query: str,
                          top_k: int = 5) -> List:
        """Search semantic knowledge base."""
        return self._mem.search_semantic(query, top_k)

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get n most recent episodes."""
        return self._mem.get_recent_episodes(n)

    def get_context(self) -> Dict:
        """Get current working context (active goal, last observation, etc.)."""
        if hasattr(self._mem, 'working'):
            return self._mem.working.context
        return {}

    def get_knowledge(self, key: str) -> Optional[Dict]:
        """Retrieve a specific semantic fact."""
        return self._mem.get_semantic(key)

    def mine_themes(self, n: int = 5) -> List[str]:
        """Extract recurring themes from recent activity."""
        if self._reflect:
            return self._reflect.mine_themes(n)
        return []

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> Dict:
        s = self._mem.status()
        s["mode"]        = self._mode
        s["cycle_count"] = self._cycle_count
        return s

    def print_status(self) -> None:
        s = self.status()
        print("\n── JanusMemory Status ──────────────────────────────")
        for k, v in s.items():
            print(f"  {k:30s}: {v}")
        print("────────────────────────────────────────────────────\n")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_holographic(self) -> bool:
        return self._mode == "holographic"

    # ── Persistence (metadata only — core vector is in-memory) ────────────────

    def _save_meta(self) -> None:
        """Save lightweight metadata to disk for session continuity."""
        if not self._persist_path:
            return
        try:
            meta = {
                "cycle_count": self._cycle_count,
                "mode": self._mode,
                "saved_at": time.time(),
                "status": {k: str(v) for k, v in self.status().items()},
            }
            with open(self._persist_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[JanusMemory] Meta save failed: {e}")

    def _load_meta(self) -> None:
        """Restore lightweight metadata from disk."""
        try:
            with open(self._persist_path) as f:
                meta = json.load(f)
            self._cycle_count = meta.get("cycle_count", 0)
            print(f"[JanusMemory] Restored metadata: "
                  f"{self._cycle_count} previous cycles")
        except Exception as e:
            print(f"[JanusMemory] Meta load failed: {e}")

    def save(self, path: Optional[str] = None) -> None:
        """Manually save metadata."""
        if path:
            self._persist_path = Path(path)
        self._save_meta()

    def __repr__(self) -> str:
        return (f"JanusMemory(mode={self._mode}, "
                f"cycles={self._cycle_count}, "
                f"items={self._mem.item_count})")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing JanusMemory...\n")

    mem = JanusMemory(dim=512)  # small dim for quick test

    # Record some cycles
    for i in range(5):
        mem.record_cycle({
            "cycle": i,
            "phase": "complete",
            "goal": f"Generate dungeon asset {i}",
            "success": i % 2 == 0,
            "notes": [f"step {i} done"],
        })

    # Record observations
    mem.record_observation("Chrome is open. A Submit button is at (640, 400).")
    mem.record_observation("File Explorer shows a folder at (320, 240).")

    # Record goals
    mem.record_goal("Generate dungeon asset pack", success=True)
    mem.record_goal("Train Avus Phase 2", success=False, notes="needs more GPU time")

    # Store knowledge
    mem.update_goal_knowledge("avus_loss", "0.2253", confidence=1.0)
    mem.update_goal_knowledge("kaggle_dataset", "janus-avus-weights", confidence=1.0)
    mem.update_goal_knowledge("engine_status", "all 9 crates passing", confidence=1.0)

    # Recall
    print("── Similar to 'dungeon generation' ────────────────────")
    results = mem.recall_similar("dungeon generation", top_k=3)
    for r in results:
        print(f"  {r.get('task_id')} | {r.get('result', '')[:60]}")

    print("\n── Knowledge search: 'avus' ────────────────────────────")
    matches = mem.search_knowledge("avus", top_k=3)
    for item in matches:
        if isinstance(item, tuple):
            key, score = item
            print(f"  {key}: score={score:.3f}" if isinstance(score, float) else f"  {key}: {score}")
        else:
            print(f"  {item}")

    print("\n── Current context ─────────────────────────────────────")
    ctx = mem.get_context()
    for k, v in ctx.items():
        print(f"  {k}: {v}")

    print()
    mem.print_status()
