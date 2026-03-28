# Janus × Holographic Brain Memory — Integration Guide

## What changed

| Component | Before | After |
|-----------|--------|-------|
| `StructuredMemory` | Plain Python dict/list | `InfiniteJanusMemory` (HRR-backed) |
| `ReflectionMemory` | Simple list + theme counting | `HolographicReflection` (similarity search) |
| `core.py` memory | `ReflectionMemory()` | `InfiniteJanusMemory` + `HolographicReflection` |
| Physical footprint | Grows with every stored item | **Fixed at ~16 KB regardless of item count** |
| Retrieval | Exact key lookup | Associative similarity search |

---

## File map

```
Janus-main/
├── holographic_memory.py          ← NEW: drop-in memory system (this is the main integration)
├── core_updated.py                ← NEW: core.py with holographic memory wired in
├── holographic_brain_memory/      ← HRR library (from Manus conversation)
│   ├── core.py                    HolographicBrainMemory, PhaseBrainLayer
│   ├── spawning.py                SpawningBrain
│   ├── real_valued.py             RealHolographicMemory
│   └── __init__.py                exports all of the above
├── solution.py                    ← InfiniteHolographicBrain (seed-based neurons)
├── infinite_demo.py               ← Demo: 1000x logical growth, zero parameter growth
└── [all original Janus files]
```

---

## How to activate

**Step 1** — In `consciousness.py`, replace:
```python
import memory
```
with:
```python
import memory
from holographic_memory import InfiniteJanusMemory
```

Then replace the `StructuredMemory()` instantiation with:
```python
self.holo_memory = InfiniteJanusMemory(dim=2048)
```

**Step 2** — Swap `core.py` for `core_updated.py`:
```bash
cp core_updated.py core.py
```

**Step 3** — Verify the HRR library is on your path:
```python
from holographic_brain_memory import HolographicBrainMemory, SpawningBrain
```

---

## New capabilities Janus now has

### Unlimited episodic memory
```python
memory = InfiniteJanusMemory()

# Store thousands of episodes — physical bytes stay fixed
for i in range(10000):
    memory.add_episode(
        task_id=f"task_{i}",
        action="respond",
        result="...",
        success=True
    )

print(memory.status())
# {'physical_bytes': 16640, 'logical_items': 10000, 'snr_estimate': 0.01, ...}
```

### Associative recall (not just key lookup)
```python
# Find episodes similar to a query — no exact key needed
results = memory.recall_similar("user asked about physics", top_k=5)

# Find semantic facts related to a concept
matches = memory.search_semantic("energy harvesting", top_k=3)
```

### Semantic memory with confidence
```python
memory.update_semantic("user_preference_humor", "dry wit", confidence=0.9)
memory.update_semantic("user_location", "unknown", confidence=0.3)

# Retrieve
fact = memory.get_semantic("user_preference_humor")
# {'vector': tensor([...]), 'confidence': 0.9, 'value_preview': 'dry wit', ...}
```

### Working context buffer
```python
memory.set_working_context("current_topic", "thermodynamics")
memory.set_working_context("user_mood", "curious")

topic = memory.get_working_context("current_topic")  # "thermodynamics"
```

### Memory diagnostics
```python
print(memory.status())
# {
#   'physical_bytes': 16640,    ← stays fixed
#   'logical_items': 847,       ← grows freely
#   'episodic_events': 312,
#   'semantic_keys': 89,
#   'working_context_keys': 4,
#   'snr_estimate': 0.034,
#   'memory_dim': 2048
# }
```

---

## How HRR memory works (brief)

All information is stored in a **single fixed-size complex vector** (the holographic core).
Each item is **bound** to a unique phase key via circular convolution (FFT), then
**superposed** (added) into the core. Retrieval uses the inverse key.

This means:
- Physical bytes = constant (~16 KB for dim=2048)
- Logical capacity = unlimited (bounded only by SNR degradation)
- Retrieval = approximate (cosine similarity), not exact
- The `CleanupNetwork` denoises retrieved vectors to compensate

The SNR drops as `1/√N` where N is items stored. The `decay` parameter
(default 0.97) ages old memories to keep the SNR healthy for recent items.

---

## Connecting to the Prometheus architecture (future)

The holographic memory is the software realization of the **Librarian** engine
from the Prometheus chip design. In the chip:
- The Librarian sorts high-entropy data to extract energy
- Here, the HRR core sorts/binds information to maximize associative capacity

The `InfiniteHolographicBrain` from `solution.py` (the Manus deliverable)
can be used as Janus's **long-term knowledge store** — a neural network
with fixed parameters that grows in logical capacity via seed-based spawning.

Future integration path:
```
Janus AutonomousCore
  └── InfiniteJanusMemory (working + episodic + semantic)
        └── HolographicMemoryCore (the fixed physical buffer)
  └── InfiniteHolographicBrain (long-term pattern store)
        └── SeedBasedPhaseLayers (unlimited logical neurons)
        └── CleanupNetwork (denoiser, fixed size)
```
