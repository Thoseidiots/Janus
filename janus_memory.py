"""
janus_memory.py
================
Unified memory system for Janus. Single entry point for all memory operations.

Replaces the fragmented memory files with one coherent system:

  Working Memory   — current session context (fast, in-RAM)
  Episodic Memory  — what happened (timestamped events, persisted)
  Semantic Memory  — what Janus knows (facts, skills, heuristics)
  Associative Memory — similarity search over all memories
  Consolidation    — automatic promotion from episodic → semantic

Architecture:
  - All memories persist to disk (survive restarts)
  - Similarity search via TF-IDF (no external dependencies)
  - Automatic consolidation during idle periods
  - Forgetting curve — old low-importance memories decay
  - Integrates with janus_identity.py for heuristics

No API keys. No ChromaDB. No external vector stores.
Pure Python stdlib + optional numpy for faster similarity.

Usage:
    from janus_memory import get_memory, Memory
    mem = get_memory()

    # Store
    mem.remember("Client Alex paid $250 for code review", category="finance", importance=0.8)
    mem.observe("Screen shows Chrome with a Submit button at (640, 400)")
    mem.learn("Always follow up with clients after 3 days", confidence=0.9)

    # Recall
    results = mem.recall("client payment")
    context = mem.working_context()
    recent  = mem.recent(10)
"""

from __future__ import annotations

import json
import math
import re
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MEMORY_FILE      = Path("janus_memory.json")
_EPISODIC_FILE    = Path("janus_episodic.jsonl")
_CONSOLIDATE_EVERY = 50   # consolidate after every N new episodic memories


# ── Memory entry ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    memory_id:    str
    content:      str
    category:     str       # episodic | semantic | working | observation | lesson
    importance:   float     # 0.0 – 1.0
    created_at:   float     # unix timestamp
    last_accessed: float
    access_count: int       = 0
    reinforced:   int       = 0   # times this was confirmed/repeated
    tags:         List[str] = field(default_factory=list)
    source:       str       = ""  # where this came from

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def recency_score(self) -> float:
        """Higher = more recent. Decays over time."""
        days = (time.time() - self.last_accessed) / 86400
        return 1.0 / (1.0 + days * 0.2)

    @property
    def strength(self) -> float:
        """Combined importance + recency + reinforcement."""
        base = self.importance * 0.5 + self.recency_score * 0.3
        boost = min(0.2, self.reinforced * 0.04)
        return min(1.0, base + boost)

    def access(self):
        self.last_accessed = time.time()
        self.access_count += 1

    def reinforce(self, amount: float = 0.05):
        self.reinforced += 1
        self.importance = min(1.0, self.importance + amount)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── TF-IDF similarity (no external deps) ─────────────────────────────────────

class _TFIDF:
    """Minimal TF-IDF for similarity search over memory contents."""

    def __init__(self):
        self._docs:  List[str]        = []
        self._ids:   List[str]        = []
        self._idf:   Dict[str, float] = {}
        self._dirty  = True

    def add(self, doc_id: str, text: str):
        self._docs.append(text.lower())
        self._ids.append(doc_id)
        self._dirty = True

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def _build_idf(self):
        N = len(self._docs)
        if N == 0:
            return
        df: Dict[str, int] = defaultdict(int)
        for doc in self._docs:
            for word in set(self._tokenize(doc)):
                df[word] += 1
        self._idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
        self._dirty = False

    def _tf_idf_vec(self, text: str) -> Dict[str, float]:
        if self._dirty:
            self._build_idf()
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0 / len(tokens)
        return {t: tf[t] * self._idf.get(t, 1.0) for t in tf}

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot  = sum(a[k] * b[k] for k in keys)
        na   = math.sqrt(sum(v*v for v in a.values()))
        nb   = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb + 1e-9)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._dirty:
            self._build_idf()
        q_vec = self._tf_idf_vec(query)
        if not q_vec:
            return []
        scores = []
        for i, doc in enumerate(self._docs):
            d_vec = self._tf_idf_vec(doc)
            sim   = self._cosine(q_vec, d_vec)
            if sim > 0:
                scores.append((self._ids[i], sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def rebuild(self, docs: List[Tuple[str, str]]):
        """Rebuild index from (id, text) pairs."""
        self._docs  = [t.lower() for _, t in docs]
        self._ids   = [i for i, _ in docs]
        self._dirty = True


# ── Main memory system ────────────────────────────────────────────────────────

class JanusMemory:
    """
    Unified memory for Janus. Single source of truth.

    Three tiers:
      Working   — current session, in-RAM, fast
      Episodic  — what happened, persisted to JSONL
      Semantic  — what Janus knows, persisted to JSON
    """

    MAX_WORKING   = 200    # working memory cap
    MAX_EPISODIC  = 5000   # episodic memory cap (oldest pruned)
    FORGET_BELOW  = 0.15   # strength threshold for forgetting

    def __init__(
        self,
        memory_file:   Path = _MEMORY_FILE,
        episodic_file: Path = _EPISODIC_FILE,
    ):
        self._mem_file = memory_file
        self._epi_file = episodic_file
        self._lock     = threading.Lock()

        # Three memory stores
        self._working:  Dict[str, MemoryEntry] = {}   # key → entry
        self._episodic: List[MemoryEntry]       = []
        self._semantic: Dict[str, MemoryEntry]  = {}  # content hash → entry

        # Search index
        self._index = _TFIDF()
        self._index_dirty = True

        # Stats
        self._new_since_consolidate = 0
        self._total_stored = 0

        self._load()

    # ── Store ─────────────────────────────────────────────────────────────────

    def remember(
        self,
        content:    str,
        category:   str   = "episodic",
        importance: float = 0.5,
        tags:       Optional[List[str]] = None,
        source:     str   = "",
    ) -> MemoryEntry:
        """Store a new memory."""
        import uuid
        entry = MemoryEntry(
            memory_id    = str(uuid.uuid4())[:8],
            content      = content,
            category     = category,
            importance   = importance,
            created_at   = time.time(),
            last_accessed= time.time(),
            tags         = tags or [],
            source       = source,
        )

        with self._lock:
            if category == "working":
                self._working[content[:50]] = entry
                if len(self._working) > self.MAX_WORKING:
                    # Evict lowest strength
                    oldest = min(self._working.values(), key=lambda e: e.strength)
                    del self._working[oldest.content[:50]]
            elif category == "semantic":
                key = self._semantic_key(content)
                if key in self._semantic:
                    self._semantic[key].reinforce()
                    return self._semantic[key]
                self._semantic[key] = entry
            else:
                self._episodic.append(entry)
                self._new_since_consolidate += 1
                if len(self._episodic) > self.MAX_EPISODIC:
                    self._prune_episodic()

            self._total_stored += 1
            self._index_dirty = True

        # Auto-consolidate
        if self._new_since_consolidate >= _CONSOLIDATE_EVERY:
            threading.Thread(target=self._consolidate, daemon=True).start()

        self._save_entry(entry)
        return entry

    def observe(self, description: str, importance: float = 0.4) -> MemoryEntry:
        """Store a screen/environment observation."""
        return self.remember(description, category="observation",
                             importance=importance, source="screen")

    def learn(self, lesson: str, confidence: float = 0.7,
              context: str = "") -> MemoryEntry:
        """Store a learned lesson or heuristic."""
        content = f"{lesson}" + (f" [context: {context}]" if context else "")
        entry = self.remember(content, category="semantic",
                              importance=confidence, source="learning")
        # Also sync to identity heuristics
        try:
            from janus_identity import get_identity
            get_identity().learn(lesson, context=context, confidence=confidence)
        except Exception:
            pass
        return entry

    def set_context(self, key: str, value: Any):
        """Set a working context value (current task, goal, etc.)."""
        self.remember(f"{key}={value}", category="working", importance=0.9)

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 10,
               category: Optional[str] = None) -> List[MemoryEntry]:
        """
        Find memories most relevant to a query.
        Uses TF-IDF similarity across all memory tiers.
        """
        self._rebuild_index_if_needed()

        results = self._index.search(query, top_k=top_k * 3)
        entries = []
        seen    = set()

        for mem_id, score in results:
            entry = self._find_by_id(mem_id)
            if entry and entry.memory_id not in seen:
                if category and entry.category != category:
                    continue
                entry.access()
                entries.append((entry, score))
                seen.add(entry.memory_id)

        # Sort by combined similarity + strength
        entries.sort(key=lambda x: x[1] * 0.6 + x[0].strength * 0.4, reverse=True)
        return [e for e, _ in entries[:top_k]]

    def recent(self, n: int = 20, category: Optional[str] = None) -> List[MemoryEntry]:
        """Get the N most recent memories."""
        all_entries = list(self._episodic) + list(self._semantic.values())
        if category:
            all_entries = [e for e in all_entries if e.category == category]
        all_entries.sort(key=lambda e: e.created_at, reverse=True)
        return all_entries[:n]

    def working_context(self) -> Dict[str, str]:
        """Get current working context as a dict."""
        ctx = {}
        for entry in self._working.values():
            if "=" in entry.content:
                k, v = entry.content.split("=", 1)
                ctx[k.strip()] = v.strip()
        return ctx

    def get_lessons(self, min_confidence: float = 0.5) -> List[MemoryEntry]:
        """Get all learned lessons above a confidence threshold."""
        return [e for e in self._semantic.values()
                if e.importance >= min_confidence]

    def get_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Get all memories with a specific tag."""
        all_entries = (list(self._episodic) + list(self._semantic.values()) +
                       list(self._working.values()))
        return [e for e in all_entries if tag in e.tags]

    def reinforce(self, memory_id: str, amount: float = 0.05):
        """Strengthen a memory (called when it proves useful)."""
        entry = self._find_by_id(memory_id)
        if entry:
            entry.reinforce(amount)

    # ── Consolidation ─────────────────────────────────────────────────────────

    def _consolidate(self):
        """
        Promote important episodic memories to semantic.
        Forget weak old memories.
        Called automatically after every N new episodic memories.
        """
        with self._lock:
            self._new_since_consolidate = 0

            # Promote high-importance episodic → semantic
            promoted = 0
            for entry in list(self._episodic):
                if entry.importance >= 0.75 and entry.access_count >= 2:
                    key = self._semantic_key(entry.content)
                    if key not in self._semantic:
                        entry.category = "semantic"
                        self._semantic[key] = entry
                        promoted += 1

            # Forget weak old memories
            cutoff = time.time() - 30 * 86400  # 30 days
            before = len(self._episodic)
            self._episodic = [
                e for e in self._episodic
                if e.strength > self.FORGET_BELOW or e.created_at > cutoff
            ]
            forgotten = before - len(self._episodic)

            self._index_dirty = True

        if promoted or forgotten:
            print(f"[Memory] Consolidation: promoted={promoted}, forgotten={forgotten}")

        self._save_all()

    def _prune_episodic(self):
        """Remove weakest episodic memories when at capacity."""
        self._episodic.sort(key=lambda e: e.strength)
        self._episodic = self._episodic[len(self._episodic) // 4:]  # drop bottom 25%

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "working":   len(self._working),
            "episodic":  len(self._episodic),
            "semantic":  len(self._semantic),
            "total":     len(self._working) + len(self._episodic) + len(self._semantic),
            "total_stored_ever": self._total_stored,
        }

    def summary(self) -> str:
        s = self.stats()
        return (f"Memory: {s['episodic']} episodic, {s['semantic']} semantic, "
                f"{s['working']} working ({s['total_stored_ever']} total stored)")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_entry(self, entry: MemoryEntry):
        """Append a single entry to the episodic log."""
        if entry.category in ("episodic", "observation"):
            try:
                with self._epi_file.open("a") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception:
                pass

    def _save_all(self):
        """Save semantic memory to JSON."""
        try:
            data = {
                "semantic": {k: v.to_dict() for k, v in self._semantic.items()},
                "saved_at": datetime.now().isoformat(),
            }
            self._mem_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Memory] Save failed: {e}")

    def _load(self):
        """Load persisted memories on startup."""
        # Load semantic
        if self._mem_file.exists():
            try:
                data = json.loads(self._mem_file.read_text())
                for k, v in data.get("semantic", {}).items():
                    self._semantic[k] = MemoryEntry.from_dict(v)
                print(f"[Memory] Loaded {len(self._semantic)} semantic memories")
            except Exception as e:
                print(f"[Memory] Semantic load failed: {e}")

        # Load recent episodic (last 500)
        if self._epi_file.exists():
            try:
                lines = self._epi_file.read_text().strip().splitlines()
                for line in lines[-500:]:
                    try:
                        self._episodic.append(MemoryEntry.from_dict(json.loads(line)))
                    except Exception:
                        pass
                print(f"[Memory] Loaded {len(self._episodic)} episodic memories")
            except Exception as e:
                print(f"[Memory] Episodic load failed: {e}")

        self._index_dirty = True

    # ── Index ─────────────────────────────────────────────────────────────────

    def _rebuild_index_if_needed(self):
        if not self._index_dirty:
            return
        docs = []
        for e in self._episodic:
            docs.append((e.memory_id, e.content))
        for e in self._semantic.values():
            docs.append((e.memory_id, e.content))
        for e in self._working.values():
            docs.append((e.memory_id, e.content))
        self._index.rebuild(docs)
        self._index_dirty = False

    def _find_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        for e in self._episodic:
            if e.memory_id == memory_id:
                return e
        for e in self._semantic.values():
            if e.memory_id == memory_id:
                return e
        for e in self._working.values():
            if e.memory_id == memory_id:
                return e
        return None

    @staticmethod
    def _semantic_key(content: str) -> str:
        """Stable key for deduplication."""
        import hashlib
        return hashlib.md5(content[:100].lower().encode()).hexdigest()[:12]


# ── Module-level singleton ────────────────────────────────────────────────────

_memory: Optional[JanusMemory] = None

def get_memory() -> JanusMemory:
    global _memory
    if _memory is None:
        _memory = JanusMemory()
    return _memory


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Memory")
    parser.add_argument("--stats",   action="store_true")
    parser.add_argument("--recall",  type=str, metavar="QUERY")
    parser.add_argument("--recent",  type=int, metavar="N", default=10)
    parser.add_argument("--learn",   type=str, metavar="LESSON")
    parser.add_argument("--lessons", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()

    mem = JanusMemory()

    if args.stats:
        print(json.dumps(mem.stats(), indent=2))
        print(mem.summary())

    elif args.recall:
        results = mem.recall(args.recall, top_k=10)
        if not results:
            print("No relevant memories found.")
        for e in results:
            age = f"{e.age_days:.0f}d ago"
            print(f"[{e.category}] [{age}] {e.content[:80]}")

    elif args.recent:
        for e in mem.recent(args.recent):
            ts = datetime.fromtimestamp(e.created_at).strftime("%m-%d %H:%M")
            print(f"[{ts}] [{e.category}] {e.content[:80]}")

    elif args.learn:
        entry = mem.learn(args.learn)
        print(f"Learned: {entry.memory_id} — {args.learn[:60]}")

    elif args.lessons:
        lessons = mem.get_lessons()
        if not lessons:
            print("No lessons learned yet.")
        for e in sorted(lessons, key=lambda x: -x.importance):
            print(f"[{e.importance:.0%}] {e.content[:80]}")

    elif args.consolidate:
        mem._consolidate()
        print("Consolidation complete.")
        print(json.dumps(mem.stats(), indent=2))

    else:
        parser.print_help()
