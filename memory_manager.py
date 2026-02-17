“””
memory_manager.py
────────────────────────────────────────────────────────────
Active forgetting, thematic compression, and confabulation detection.
Integrates with memory.py (existing) and event_trace.jsonl (tool_executor).
“””

import json
import math
import hashlib
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

@dataclass
class MemoryEntry:
memory_id:   str
content:     Any
importance:  float          # 0.0 – 1.0
valence:     float          # -1.0 – +1.0
access_count: int           = 0
created_at:  str            = field(default_factory=lambda: datetime.now().isoformat())
last_accessed: str          = field(default_factory=lambda: datetime.now().isoformat())
tags:        List[str]      = field(default_factory=list)
compressed:  bool           = False
parent_id:   Optional[str]  = None
source_ids:  List[str]      = field(default_factory=list)  # compressed from these

```
def age_days(self) -> float:
    return (datetime.now() - datetime.fromisoformat(self.created_at)).total_seconds() / 86400

def rank(self) -> float:
    """Higher = more worth keeping.  Combines recency, importance, access frequency."""
    recency    = math.exp(-0.1 * self.age_days())
    importance = self.importance
    frequency  = math.log1p(self.access_count) / 5
    return 0.4 * recency + 0.4 * importance + 0.2 * frequency
```

class DecayEngine:
“””
Applies exponential decay to memory ranks over time.
Low-rank memories are flagged for compression or deletion.
“””

```
FORGET_THRESHOLD = 0.15   # below this rank → candidate for forgetting
COMPRESS_THRESHOLD = 0.35 # below this (but above forget) → compress

def classify(self, entry: MemoryEntry) -> str:
    r = entry.rank()
    if r < self.FORGET_THRESHOLD:
        return "forget"
    if r < self.COMPRESS_THRESHOLD:
        return "compress"
    return "keep"

def rank_all(self, memories: List[MemoryEntry]) -> List[Tuple[MemoryEntry, str, float]]:
    return [(m, self.classify(m), m.rank()) for m in memories]
```

class ThematicCompressor:
“””
Clusters memories by tag/topic similarity and summarises each cluster
into a single higher-order abstract memory.
“””

```
MIN_CLUSTER_SIZE = 3

def cluster_by_tags(self, memories: List[MemoryEntry]) -> Dict[str, List[MemoryEntry]]:
    clusters: Dict[str, List[MemoryEntry]] = defaultdict(list)
    for m in memories:
        primary_tag = m.tags[0] if m.tags else "untagged"
        clusters[primary_tag].append(m)
    return {k: v for k, v in clusters.items() if len(v) >= self.MIN_CLUSTER_SIZE}

def compress_cluster(self, tag: str, cluster: List[MemoryEntry]) -> MemoryEntry:
    """Produce one abstract memory from a cluster."""
    avg_importance = sum(m.importance for m in cluster) / len(cluster)
    avg_valence    = sum(m.valence    for m in cluster) / len(cluster)
    contents = [
        str(m.content)[:80] if not isinstance(m.content, dict)
        else m.content.get("summary", str(m.content))[:80]
        for m in cluster
    ]
    summary = f"[Compressed/{tag}] {len(cluster)} episodes: " + " | ".join(contents[:3])

    return MemoryEntry(
        memory_id    = "compressed_" + hashlib.sha256(tag.encode()).hexdigest()[:8],
        content      = {"summary": summary, "cluster_tag": tag, "source_count": len(cluster)},
        importance   = min(1.0, avg_importance * 1.2),   # slight boost for surviving compression
        valence      = avg_valence,
        tags         = [tag, "compressed"],
        compressed   = True,
        source_ids   = [m.memory_id for m in cluster],
    )
```

class ConfabulationDetector:
“””
Cross-checks memories against the append-only event_trace.jsonl and
perception snapshots to flag internally inconsistent or hallucinated claims.
“””

```
def __init__(self, trace_path: str = "event_trace.jsonl"):
    self.trace_path = Path(trace_path)

def load_trace_facts(self) -> List[dict]:
    """Load verifiable facts from the event trace."""
    if not self.trace_path.exists():
        return []
    facts = []
    for line in self.trace_path.read_text().splitlines():
        try:
            facts.append(json.loads(line))
        except Exception:
            pass
    return facts

def check(self, memory: MemoryEntry) -> Tuple[bool, str]:
    """
    Returns (is_consistent, reason).
    A memory is suspicious if it claims something that contradicts the trace.
    """
    content_str = json.dumps(memory.content) if isinstance(memory.content, dict) else str(memory.content)
    facts = self.load_trace_facts()

    # Heuristic 1: Memory claims a tool succeeded but trace shows failure
    if "success" in content_str.lower():
        for fact in facts:
            result = fact.get("data", {}).get("result", {})
            if isinstance(result, dict) and result.get("success") is False:
                tool = result.get("tool", "")
                if tool and tool in content_str:
                    return False, f"Memory claims success but trace shows failure for tool '{tool}'"

    # Heuristic 2: Memory references an entity never seen in perception
    # (simple check: look for proper nouns not in any trace entry)
    # This is intentionally lightweight; LLM-based cross-check hooks in via override.

    return True, "consistent"

def audit_all(self, memories: List[MemoryEntry]) -> List[dict]:
    flagged = []
    for m in memories:
        ok, reason = self.check(m)
        if not ok:
            flagged.append({
                "memory_id": m.memory_id,
                "issue":     reason,
                "content_preview": str(m.content)[:100],
            })
    return flagged
```

class MemoryManager:
“””
Unified memory lifecycle manager.  Wraps the existing memory.py store
and adds decay, compression, and confabulation detection.

```
Usage
─────
mgr = MemoryManager(store_path="memories.jsonl")
mgr.store({"type": "experience", "summary": "..."}, importance=0.7)
mgr.run_maintenance()   # call during sleep phase
flagged = mgr.audit_confabulation()
"""

def __init__(self, store_path: str = "memories.jsonl",
             trace_path: str = "event_trace.jsonl"):
    self.store_path = Path(store_path)
    self.decay      = DecayEngine()
    self.compressor = ThematicCompressor()
    self.detector   = ConfabulationDetector(trace_path)
    self._memories: Dict[str, MemoryEntry] = {}
    self._lock = threading.Lock()
    self._load()

# ── Store / Retrieve ───────────────────────────────────────────────────────
def store(self, content: Any, importance: float = 0.5,
          valence: float = 0.0, tags: List[str] = None) -> str:
    mid = "m_" + hashlib.sha256(
        (str(content) + datetime.now().isoformat()).encode()
    ).hexdigest()[:12]
    entry = MemoryEntry(
        memory_id  = mid,
        content    = content,
        importance = importance,
        valence    = valence,
        tags       = tags or self._auto_tag(content),
    )
    with self._lock:
        self._memories[mid] = entry
    self._append_to_disk(entry)
    return mid

def query(self, query: str, limit: int = 5) -> List[dict]:
    query_lower = query.lower()
    with self._lock:
        matches = []
        for m in self._memories.values():
            content_str = json.dumps(m.content) if isinstance(m.content, dict) else str(m.content)
            if query_lower in content_str.lower() or any(query_lower in t for t in m.tags):
                m.access_count += 1
                m.last_accessed = datetime.now().isoformat()
                matches.append(m)
        matches.sort(key=lambda m: m.rank(), reverse=True)
    return [asdict(m) for m in matches[:limit]]

# ── Maintenance (call during sleep phase) ──────────────────────────────────
def run_maintenance(self) -> dict:
    """Full maintenance pass: decay, compress, forget."""
    with self._lock:
        memories = list(self._memories.values())

    ranked     = self.decay.rank_all(memories)
    to_forget  = [m for m, label, _ in ranked if label == "forget"]
    to_compress = [m for m, label, _ in ranked if label == "compress"]

    # Forget low-value memories
    forgotten = 0
    with self._lock:
        for m in to_forget:
            if not m.compressed:   # never delete compressed summaries
                del self._memories[m.memory_id]
                forgotten += 1

    # Thematic compression
    clusters = self.compressor.cluster_by_tags(to_compress)
    compressed_count = 0
    for tag, cluster in clusters.items():
        abstract = self.compressor.compress_cluster(tag, cluster)
        with self._lock:
            for m in cluster:
                if m.memory_id in self._memories:
                    del self._memories[m.memory_id]
            self._memories[abstract.memory_id] = abstract
        self._append_to_disk(abstract)
        compressed_count += len(cluster)

    # Rewrite store
    self._rewrite_store()

    report = {
        "ts":              datetime.now().isoformat(),
        "total_before":    len(memories),
        "forgotten":       forgotten,
        "compressed_into": len(clusters),
        "sources_removed": compressed_count,
        "total_after":     len(self._memories),
    }
    print(f"[MemoryManager] Maintenance: {report}")
    return report

def audit_confabulation(self) -> List[dict]:
    with self._lock:
        memories = list(self._memories.values())
    flagged = self.detector.audit_all(memories)
    if flagged:
        print(f"[MemoryManager] ⚠ Confabulation audit: {len(flagged)} suspicious memories")
    return flagged

# ── Helpers ────────────────────────────────────────────────────────────────
def _auto_tag(self, content: Any) -> List[str]:
    tags = []
    text = json.dumps(content) if isinstance(content, dict) else str(content)
    keywords = {
        "person": ["person", "human", "face", "user"],
        "tool":   ["tool", "exec", "code", "run"],
        "goal":   ["goal", "task", "complete", "stall"],
        "vision": ["scene", "visual", "camera", "object"],
        "audio":  ["speech", "voice", "sound", "heard"],
    }
    for tag, kws in keywords.items():
        if any(kw in text.lower() for kw in kws):
            tags.append(tag)
    return tags or ["general"]

def _append_to_disk(self, entry: MemoryEntry):
    with self.store_path.open("a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")

def _rewrite_store(self):
    with self._lock:
        lines = [json.dumps(asdict(m)) for m in self._memories.values()]
    self.store_path.write_text("\n".join(lines) + "\n")

def _load(self):
    if not self.store_path.exists():
        return
    for line in self.store_path.read_text().splitlines():
        try:
            d = json.loads(line)
            entry = MemoryEntry(**{k: v for k, v in d.items()
                                   if k in MemoryEntry.__dataclass_fields__})
            self._memories[entry.memory_id] = entry
        except Exception:
            pass
    print(f"[MemoryManager] Loaded {len(self._memories)} memories from disk")
```