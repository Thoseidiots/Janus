"""
holographic_memory.py — Infinite Holographic Brain Memory for Janus

Replaces StructuredMemory and ReflectionMemory with a fixed-byte,
unbounded-capacity associative memory system based on Holographic
Reduced Representations (HRR) and Vector Symbolic Architectures (VSA).

Architecture:
  - HolographicMemoryCore   : fixed-size complex vector (the physical store)
  - HolographicEpisodicLayer: write/read episodic events via phase keys
  - HolographicSemanticLayer: key→value associative store with decay
  - HolographicWorkingBuffer: small, fast scratch-pad (real-valued)
  - InfiniteJanusMemory     : drop-in replacement for StructuredMemory
  - HolographicReflection   : drop-in replacement for ReflectionMemory

Drop-in usage in core.py:
    from holographic_memory import InfiniteJanusMemory, HolographicReflection
    self.memory   = InfiniteJanusMemory()
    self.reflect  = HolographicReflection(self.memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1. Core holographic memory (fixed bytes)
# ─────────────────────────────────────────────────────────────

class HolographicMemoryCore(nn.Module):
    """
    Fixed-size complex-valued memory vector.
    All information is superposed into this single buffer via
    circular convolution (FFT-based binding).

    Physical size is constant regardless of how many items are stored.
    """

    def __init__(self, dim: int = 2048, decay: float = 0.97):
        super().__init__()
        self.dim = dim
        self.decay = decay
        # The single fixed buffer — this is ALL the physical memory
        init = torch.randn(dim, dtype=torch.cfloat) * 0.01
        self.memory = nn.Parameter(init)

    # ── binding / unbinding ──────────────────────────────────

    def bind(self, signal: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Phase-modulate signal with theta key via circular convolution."""
        sig_c = signal.to(torch.cfloat)
        key_c = torch.exp(1j * theta).to(torch.cfloat)
        return torch.fft.ifft(torch.fft.fft(sig_c) * torch.fft.fft(key_c)).real.to(torch.cfloat)

    def write(self, signal: torch.Tensor, theta: torch.Tensor):
        """Superpose a new signal into the holographic buffer."""
        bound = self.bind(signal.to(torch.cfloat), theta)
        with torch.no_grad():
            self.memory.data = self.decay * self.memory.data + bound
            # Soft normalisation — keeps SNR healthy
            norm = self.memory.data.abs().max()
            if norm > 10.0:
                self.memory.data = self.memory.data / norm * 10.0

    def read(self, theta: torch.Tensor) -> torch.Tensor:
        """Retrieve a signal using the inverse phase key."""
        inv_key = torch.exp(-1j * theta).to(torch.cfloat)
        unbound = torch.fft.ifft(torch.fft.fft(self.memory.data) * torch.fft.fft(inv_key))
        return unbound.real

    @property
    def byte_size(self) -> int:
        return self.memory.numel() * 8  # complex64 = 8 bytes

    def snr_estimate(self, n_items: int) -> float:
        return 1.0 / max(np.sqrt(n_items), 1.0)


# ─────────────────────────────────────────────────────────────
# 2. Seed-based phase key generator (zero extra parameters)
# ─────────────────────────────────────────────────────────────

def _make_theta(seed: int, dim: int) -> torch.Tensor:
    """Deterministically generate a phase key from a seed integer."""
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.uniform(0, 2 * np.pi, size=(dim,)).astype(np.float32)
    )


def _text_seed(text: str) -> int:
    """Hash arbitrary text to a stable integer seed."""
    h = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _encode_text(text: str, dim: int) -> torch.Tensor:
    """Encode a string as a deterministic float vector."""
    rng = np.random.default_rng(_text_seed(text + "_enc"))
    raw = rng.standard_normal(dim).astype(np.float32)
    t = torch.tensor(raw)
    return t / (t.norm() + 1e-8)


# ─────────────────────────────────────────────────────────────
# 3. Episodic layer
# ─────────────────────────────────────────────────────────────

class HolographicEpisodicLayer:
    """
    Stores episodic events (task_id, action, result, success, timestamp)
    in the shared holographic core.

    Each event gets a unique seed → unique phase key → written to core.
    The event index list grows (it's just integers), but the physical
    memory footprint (the core buffer) stays fixed.
    """

    def __init__(self, core: HolographicMemoryCore):
        self.core = core
        self.event_seeds: List[int] = []
        self.event_meta: List[Dict] = []   # tiny metadata (no vectors)
        self._seed_counter = 1_000_000     # episodic seeds start here

    def add_episode(self, task_id: str, action: str, result: Any, success: bool):
        seed = self._seed_counter
        self._seed_counter += 1

        meta = {
            "seed": seed,
            "task_id": task_id,
            "action": action,
            "result": str(result)[:256],
            "success": success,
            "timestamp": time.time(),
        }
        self.event_meta.append(meta)
        self.event_seeds.append(seed)

        # Encode the event as a vector and write
        text_repr = f"{task_id}:{action}:{result}:{success}"
        signal = _encode_text(text_repr, self.core.dim)
        theta = _make_theta(seed, self.core.dim)
        self.core.write(signal, theta)

    def recall_recent(self, n: int = 10) -> List[Dict]:
        """Return metadata for the n most recent episodes."""
        return self.event_meta[-n:]

    def recall_by_task(self, task_id: str) -> List[Dict]:
        return [m for m in self.event_meta if m["task_id"] == task_id]

    def recall_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve episodes whose stored signal is most similar to a query.
        Uses cosine similarity between retrieved vectors and query encoding.
        """
        if not self.event_seeds:
            return []

        q_vec = _encode_text(query, self.core.dim)
        scores = []
        for i, seed in enumerate(self.event_seeds):
            theta = _make_theta(seed, self.core.dim)
            retrieved = self.core.read(theta)
            sim = F.cosine_similarity(
                q_vec.unsqueeze(0), retrieved.unsqueeze(0)
            ).item()
            scores.append((sim, i))

        scores.sort(reverse=True)
        return [self.event_meta[i] for _, i in scores[:top_k]]

    @property
    def count(self) -> int:
        return len(self.event_seeds)


# ─────────────────────────────────────────────────────────────
# 4. Semantic layer
# ─────────────────────────────────────────────────────────────

class HolographicSemanticLayer:
    """
    Key→value associative store backed by the holographic core.

    Keys are arbitrary strings; values are stored as encoded vectors.
    The key string is hashed to a phase key for retrieval.
    Confidence scores and timestamps are stored as lightweight metadata.
    """

    def __init__(self, core: HolographicMemoryCore):
        self.core = core
        self.key_registry: Dict[str, Dict] = {}  # key → metadata only

    def update(self, key: str, value: Any, confidence: float = 1.0):
        seed = _text_seed(key)
        signal = _encode_text(str(value), self.core.dim)
        theta = _make_theta(seed, self.core.dim)
        self.core.write(signal, theta)

        self.key_registry[key] = {
            "seed": seed,
            "confidence": confidence,
            "last_updated": time.time(),
            "value_preview": str(value)[:128],
        }

    def get(self, key: str) -> Optional[Dict]:
        if key not in self.key_registry:
            return None
        meta = self.key_registry[key]
        seed = meta["seed"]
        theta = _make_theta(seed, self.core.dim)
        retrieved = self.core.read(theta)
        return {
            "vector": retrieved,
            "confidence": meta["confidence"],
            "last_updated": meta["last_updated"],
            "value_preview": meta["value_preview"],
        }

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantic keys most similar to a query string."""
        q_vec = _encode_text(query, self.core.dim)
        scores = []
        for key, meta in self.key_registry.items():
            theta = _make_theta(meta["seed"], self.core.dim)
            retrieved = self.core.read(theta)
            sim = F.cosine_similarity(
                q_vec.unsqueeze(0), retrieved.unsqueeze(0)
            ).item()
            scores.append((key, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def keys(self) -> List[str]:
        return list(self.key_registry.keys())


# ─────────────────────────────────────────────────────────────
# 5. Working memory buffer (fast, real-valued scratch-pad)
# ─────────────────────────────────────────────────────────────

class HolographicWorkingBuffer:
    """
    Small, fast working memory — stores current context items.
    Uses the holographic core for persistence but also keeps a
    plain dict for instant retrieval of active context.
    """

    def __init__(self, core: HolographicMemoryCore, capacity: int = 32):
        self.core = core
        self.capacity = capacity
        self._buffer: Dict[str, Any] = {}
        self._seed_base = 2_000_000

    def set(self, key: str, value: Any):
        if len(self._buffer) >= self.capacity:
            # Evict oldest
            oldest = next(iter(self._buffer))
            del self._buffer[oldest]
        self._buffer[key] = value
        # Also write to holographic core for persistence
        seed = self._seed_base + _text_seed(key) % 500_000
        signal = _encode_text(str(value), self.core.dim)
        theta = _make_theta(seed, self.core.dim)
        self.core.write(signal, theta)

    def get(self, key: str) -> Any:
        return self._buffer.get(key)

    def clear(self):
        self._buffer.clear()

    @property
    def context(self) -> Dict[str, Any]:
        return dict(self._buffer)


# ─────────────────────────────────────────────────────────────
# 6. Cleanup network (denoises retrieved vectors)
# ─────────────────────────────────────────────────────────────

class CleanupNetwork(nn.Module):
    """
    Small fixed-size auto-associative network.
    Learns the manifold of valid signals and projects
    noisy retrievals back onto it.
    Fixed size — does NOT grow with memory load.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 7. InfiniteJanusMemory — drop-in for StructuredMemory
# ─────────────────────────────────────────────────────────────

class InfiniteJanusMemory:
    """
    Drop-in replacement for StructuredMemory.

    API compatibility:
        add_episode(task_id, action, result, success)
        update_semantic(key, value, confidence)
        get_semantic(key) → dict or None
        set_working_context(key, value)
        get_working_context(key) → Any

    Extensions:
        recall_similar(query)      → episodic similarity search
        search_semantic(query)     → semantic similarity search
        byte_size                  → physical footprint (stays fixed)
        item_count                 → total logical items stored
        snr                        → current signal-to-noise estimate
    """

    def __init__(self, dim: int = 2048, decay: float = 0.97):
        self.core = HolographicMemoryCore(dim=dim, decay=decay)
        self.cleanup = CleanupNetwork(dim=dim)
        self.episodic = HolographicEpisodicLayer(self.core)
        self.semantic = HolographicSemanticLayer(self.core)
        self.working = HolographicWorkingBuffer(self.core)
        self._total_writes = 0

    # ── StructuredMemory-compatible API ─────────────────────

    def add_episode(self, task_id: str, action: str, result: Any, success: bool):
        self.episodic.add_episode(task_id, action, result, success)
        self._total_writes += 1

    def update_semantic(self, key: str, value: Any, confidence: float = 1.0):
        self.semantic.update(key, value, confidence)
        self._total_writes += 1

    def get_semantic(self, key: str) -> Optional[Dict]:
        return self.semantic.get(key)

    def set_working_context(self, key: str, value: Any):
        self.working.set(key, value)

    def get_working_context(self, key: str) -> Any:
        return self.working.get(key)

    # ── Extended API ────────────────────────────────────────

    def recall_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve episodes most similar to a query string."""
        return self.episodic.recall_similar(query, top_k)

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find semantic keys most similar to a query string."""
        return self.semantic.search(query, top_k)

    def get_recent_episodes(self, n: int = 10) -> List[Dict]:
        return self.episodic.recall_recent(n)

    def denoise(self, vector: torch.Tensor) -> torch.Tensor:
        """Run a retrieved vector through the cleanup network."""
        with torch.no_grad():
            return self.cleanup(vector.unsqueeze(0)).squeeze(0)

    # ── Diagnostics ─────────────────────────────────────────

    @property
    def byte_size(self) -> int:
        """Physical memory footprint — stays fixed."""
        core_bytes = self.core.byte_size
        cleanup_bytes = sum(
            p.numel() * 4 for p in self.cleanup.parameters()
        )
        return core_bytes + cleanup_bytes

    @property
    def item_count(self) -> int:
        """Total logical items stored — grows without bound."""
        return self._total_writes

    @property
    def snr(self) -> float:
        return self.core.snr_estimate(max(self._total_writes, 1))

    def status(self) -> Dict:
        return {
            "physical_bytes": self.byte_size,
            "logical_items": self.item_count,
            "episodic_events": self.episodic.count,
            "semantic_keys": len(self.semantic.keys),
            "working_context_keys": len(self.working.context),
            "snr_estimate": round(self.snr, 4),
            "memory_dim": self.core.dim,
        }


# ─────────────────────────────────────────────────────────────
# 8. HolographicReflection — drop-in for ReflectionMemory
# ─────────────────────────────────────────────────────────────

class HolographicReflection:
    """
    Drop-in replacement for ReflectionMemory.

    Stores valence states and associated stimuli in the holographic core.
    mine_themes() does similarity clustering over recent episodes.

    API compatibility:
        add(valence, stimulus_text)
        mine_themes() → List[str]
    """

    def __init__(self, memory: InfiniteJanusMemory):
        self.memory = memory
        self._theme_seed_base = 3_000_000
        self._recent_stimuli: List[str] = []

    def add(self, valence, stimulus_text: str):
        """
        Store a valence+stimulus pair.
        valence can be a ValenceVector or any object with a to_tensor() method,
        or a plain dict/tensor.
        """
        # Convert valence to a string summary for storage
        if hasattr(valence, "to_tensor"):
            v_tensor = valence.to_tensor()
            v_summary = ",".join(f"{x:.2f}" for x in v_tensor.tolist())
        elif isinstance(valence, dict):
            v_summary = ",".join(f"{k}={v:.2f}" for k, v in valence.items())
        else:
            v_summary = str(valence)

        task_id = f"reflection_{int(time.time()*1000) % 1_000_000}"
        self.memory.add_episode(
            task_id=task_id,
            action="perceive",
            result=f"{v_summary}|{stimulus_text[:128]}",
            success=True,
        )
        self._recent_stimuli.append(stimulus_text)
        # Keep local list bounded
        if len(self._recent_stimuli) > 50:
            self._recent_stimuli = self._recent_stimuli[-50:]

    def mine_themes(self, n_themes: int = 5) -> List[str]:
        """
        Extract recurring themes from recent stimuli using
        holographic similarity search.
        """
        if not self._recent_stimuli:
            return []

        themes = []
        seen_seeds = set()

        for stimulus in self._recent_stimuli[-20:]:
            # Search semantic memory for concepts related to this stimulus
            matches = self.memory.search_semantic(stimulus, top_k=2)
            for key, score in matches:
                if score > 0.1 and key not in seen_seeds:
                    themes.append(key)
                    seen_seeds.add(key)

        # Fall back: extract noun-like tokens from recent stimuli
        if not themes:
            words: Dict[str, int] = {}
            for s in self._recent_stimuli[-10:]:
                for w in s.split():
                    w = w.strip(".,!?;:'\"").lower()
                    if len(w) > 4:
                        words[w] = words.get(w, 0) + 1
            sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
            themes = [w for w, _ in sorted_words[:n_themes]]

        return themes[:n_themes]
