“””
state_sync.py
────────────────────────────────────────────────────────────
Conflict-free state synchronisation for Janus across laptop/phone/server.
Uses a Last-Write-Wins CRDT (LWW-Register per key) with vector clocks,
serialised to a single sync manifest that can be rsync’d, git-committed,
or placed in a shared folder with zero server infrastructure.
“””

import json
import uuid
import time
import hashlib
import threading
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ── Vector clock ──────────────────────────────────────────────────────────────

VectorClock = Dict[str, int]   # device_id → logical timestamp

def vc_increment(vc: VectorClock, device_id: str) -> VectorClock:
new = dict(vc)
new[device_id] = new.get(device_id, 0) + 1
return new

def vc_merge(a: VectorClock, b: VectorClock) -> VectorClock:
keys = set(a) | set(b)
return {k: max(a.get(k, 0), b.get(k, 0)) for k in keys}

def vc_dominates(a: VectorClock, b: VectorClock) -> bool:
“”“True if a is strictly newer than b on every device.”””
all_keys = set(a) | set(b)
return all(a.get(k, 0) >= b.get(k, 0) for k in all_keys) and a != b

# ── LWW Register (one key in the state space) ─────────────────────────────────

@dataclass
class LWWEntry:
key:        str
value:      Any
wall_time:  float           # unix timestamp (tiebreaker)
vector:     VectorClock     = field(default_factory=dict)
device_id:  str             = “”
tombstone:  bool            = False   # True = deleted

# ── Sync manifest ─────────────────────────────────────────────────────────────

@dataclass
class SyncManifest:
device_id:   str
entries:     Dict[str, LWWEntry]   # key → LWWEntry
vector:      VectorClock
created_at:  str = field(default_factory=lambda: datetime.now().isoformat())
updated_at:  str = field(default_factory=lambda: datetime.now().isoformat())

```
def to_dict(self) -> dict:
    return {
        "device_id":  self.device_id,
        "vector":     self.vector,
        "created_at": self.created_at,
        "updated_at": self.updated_at,
        "entries":    {k: asdict(v) for k, v in self.entries.items()},
    }

@staticmethod
def from_dict(d: dict) -> "SyncManifest":
    entries = {
        k: LWWEntry(**v) for k, v in d.get("entries", {}).items()
    }
    return SyncManifest(
        device_id  = d["device_id"],
        entries    = entries,
        vector     = d.get("vector", {}),
        created_at = d.get("created_at", datetime.now().isoformat()),
        updated_at = d.get("updated_at", datetime.now().isoformat()),
    )
```

class JanusStateSync:
“””
Drop-in replacement / wrapper around identity_object.json + persistent_state.json.

```
Each write goes through the CRDT; each sync merges two manifests.
Produce a manifest file (sync_manifest.json) that any other device can
pull and merge with its local manifest.

Usage
─────
sync = JanusStateSync(device_id="laptop-01")
sync.set("valence.pleasure", 0.7)
sync.set("identity.name",    "Janus")
sync.save_manifest("sync_manifest.json")

# On another device:
sync2 = JanusStateSync(device_id="server-01")
sync2.merge_manifest("sync_manifest.json")
print(sync2.get("valence.pleasure"))   # → 0.7
"""

MANIFEST_PATH    = Path("sync_manifest.json")
CHECKPOINT_DIR   = Path("sync_checkpoints")
MAX_CHECKPOINTS  = 10

def __init__(self, device_id: Optional[str] = None,
             manifest_path: Optional[str] = None):
    self.device_id = device_id or self._load_or_create_device_id()
    if manifest_path:
        self.MANIFEST_PATH = Path(manifest_path)

    self._lock = threading.Lock()
    self._manifest = self._load_manifest()
    self.CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"[StateSync] Device: {self.device_id}  |  Keys: {len(self._manifest.entries)}")

# ── Read / Write API ──────────────────────────────────────────────────────
def set(self, key: str, value: Any):
    with self._lock:
        new_vc = vc_increment(self._manifest.vector, self.device_id)
        self._manifest.vector = new_vc
        self._manifest.entries[key] = LWWEntry(
            key       = key,
            value     = value,
            wall_time = time.time(),
            vector    = dict(new_vc),
            device_id = self.device_id,
        )
        self._manifest.updated_at = datetime.now().isoformat()

def get(self, key: str, default: Any = None) -> Any:
    with self._lock:
        entry = self._manifest.entries.get(key)
        if entry is None or entry.tombstone:
            return default
        return entry.value

def delete(self, key: str):
    """Soft delete via tombstone."""
    with self._lock:
        if key in self._manifest.entries:
            self._manifest.entries[key].tombstone = True
            self._manifest.entries[key].wall_time = time.time()

def all_keys(self) -> List[str]:
    with self._lock:
        return [k for k, v in self._manifest.entries.items() if not v.tombstone]

def snapshot(self) -> dict:
    """Export current state as plain dict (for backward compat with persistent_state.json)."""
    with self._lock:
        return {k: v.value for k, v in self._manifest.entries.items() if not v.tombstone}

# ── Merge (CRDT sync) ─────────────────────────────────────────────────────
def merge_manifest(self, path: str) -> dict:
    """
    Merge a remote manifest file into local state.
    Returns a report of conflicts resolved.
    """
    remote_data  = json.loads(Path(path).read_text())
    remote       = SyncManifest.from_dict(remote_data)
    report       = {"merged": 0, "kept_local": 0, "kept_remote": 0, "conflicts": []}

    with self._lock:
        for key, remote_entry in remote.entries.items():
            local_entry = self._manifest.entries.get(key)

            if local_entry is None:
                # New key from remote
                self._manifest.entries[key] = remote_entry
                report["merged"] += 1
            else:
                # LWW: prefer entry with higher wall_time;
                # tie-break by device_id (deterministic)
                if remote_entry.wall_time > local_entry.wall_time:
                    self._manifest.entries[key] = remote_entry
                    report["kept_remote"] += 1
                    report["conflicts"].append({
                        "key": key,
                        "winner": remote_entry.device_id,
                        "delta_ms": round((remote_entry.wall_time - local_entry.wall_time) * 1000),
                    })
                elif remote_entry.wall_time == local_entry.wall_time:
                    # Deterministic tie-break
                    winner = max(remote_entry.device_id, local_entry.device_id)
                    if winner == remote_entry.device_id:
                        self._manifest.entries[key] = remote_entry
                    report["kept_local"] += 1
                else:
                    report["kept_local"] += 1

        # Merge vector clocks
        self._manifest.vector = vc_merge(self._manifest.vector, remote.vector)
        self._manifest.updated_at = datetime.now().isoformat()

    print(f"[StateSync] Merge complete: {report}")
    return report

# ── Persistence ───────────────────────────────────────────────────────────
def save_manifest(self, path: Optional[str] = None):
    target = Path(path) if path else self.MANIFEST_PATH
    with self._lock:
        data = self._manifest.to_dict()
    target.write_text(json.dumps(data, indent=2))
    self._maybe_checkpoint()

def export_to_legacy(self, persistent_state_path: str = "persistent_state.json",
                      identity_path: str = "identity_object.json"):
    """Write back to the legacy JSON files Janus expects."""
    snap = self.snapshot()

    # Merge into existing persistent_state.json
    ps_path = Path(persistent_state_path)
    existing = json.loads(ps_path.read_text()) if ps_path.exists() else {}
    existing.update(snap)
    ps_path.write_text(json.dumps(existing, indent=2))

    # Write identity keys
    id_keys = {k: v for k, v in snap.items() if k.startswith("identity.")}
    if id_keys:
        id_path = Path(identity_path)
        existing_id = json.loads(id_path.read_text()) if id_path.exists() else {}
        for k, v in id_keys.items():
            sub_key = k.replace("identity.", "")
            existing_id[sub_key] = v
        id_path.write_text(json.dumps(existing_id, indent=2))

    print(f"[StateSync] Exported {len(snap)} keys to legacy files")

def _load_manifest(self) -> SyncManifest:
    if self.MANIFEST_PATH.exists():
        try:
            data = json.loads(self.MANIFEST_PATH.read_text())
            return SyncManifest.from_dict(data)
        except Exception as e:
            print(f"[StateSync] Load error: {e}, starting fresh")
    return SyncManifest(device_id=self.device_id, entries={}, vector={})

def _maybe_checkpoint(self):
    """Keep rolling checkpoints for rollback."""
    if not self.MANIFEST_PATH.exists():
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = self.CHECKPOINT_DIR / f"manifest_{ts}.json"
    shutil.copy2(self.MANIFEST_PATH, dest)
    # Prune old checkpoints
    checkpoints = sorted(self.CHECKPOINT_DIR.glob("manifest_*.json"))
    for old in checkpoints[:-self.MAX_CHECKPOINTS]:
        old.unlink()

def _load_or_create_device_id(self) -> str:
    id_file = Path(".janus_device_id")
    if id_file.exists():
        return id_file.read_text().strip()
    device_id = "device_" + str(uuid.uuid4())[:8]
    id_file.write_text(device_id)
    return device_id

def get_status(self) -> dict:
    with self._lock:
        return {
            "device_id":  self.device_id,
            "key_count":  len([k for k, v in self._manifest.entries.items() if not v.tombstone]),
            "vector":     self._manifest.vector,
            "updated_at": self._manifest.updated_at,
        }
```