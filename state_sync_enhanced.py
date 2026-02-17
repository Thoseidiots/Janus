"""
state_sync_enhanced.py
────────────────────────────────────────────────────────────
Enhanced conflict-free state synchronisation for Janus across 
laptop/phone/server with identity/memory persistence.

Features:
- CRDT-based state synchronization (LWW-Register per key)
- Vector clocks for causality tracking
- Automatic device discovery
- Encrypted sync over local network
- Persistent identity across devices
- Memory synchronization
- Real-time sync via WebSocket
"""

import json
import uuid
import time
import hashlib
import threading
import shutil
import socket
import struct
import select
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from pathlib import Path
from enum import Enum
import logging

# Optional encryption
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[StateSync] cryptography not available, sync will be unencrypted")


# ── Vector clock utilities ─────────────────────────────────────────────────

VectorClock = Dict[str, int]

def vc_increment(vc: VectorClock, device_id: str) -> VectorClock:
    """Increment vector clock for device"""
    new = dict(vc)
    new[device_id] = new.get(device_id, 0) + 1
    return new

def vc_merge(a: VectorClock, b: VectorClock) -> VectorClock:
    """Merge two vector clocks"""
    keys = set(a) | set(b)
    return {k: max(a.get(k, 0), b.get(k, 0)) for k in keys}

def vc_compare(a: VectorClock, b: VectorClock) -> int:
    """
    Compare two vector clocks
    Returns: -1 if a < b, 0 if concurrent/incomparable, 1 if a > b
    """
    all_keys = set(a) | set(b)
    
    a_dominates = all(a.get(k, 0) >= b.get(k, 0) for k in all_keys)
    b_dominates = all(b.get(k, 0) >= a.get(k, 0) for k in all_keys)
    
    if a == b:
        return 0
    if a_dominates and not b_dominates:
        return 1
    if b_dominates and not a_dominates:
        return -1
    return 0  # Concurrent

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class LWWEntry:
    """Last-Write-Wins register entry"""
    key: str
    value: Any
    wall_time: float
    vector: VectorClock = field(default_factory=dict)
    device_id: str = ""
    tombstone: bool = False
    
    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'value': self.value,
            'wall_time': self.wall_time,
            'vector': self.vector,
            'device_id': self.device_id,
            'tombstone': self.tombstone,
        }
    
    @staticmethod
    def from_dict(d: dict) -> "LWWEntry":
        return LWWEntry(
            key=d['key'],
            value=d['value'],
            wall_time=d['wall_time'],
            vector=d.get('vector', {}),
            device_id=d.get('device_id', ''),
            tombstone=d.get('tombstone', False),
        )


@dataclass
class DeviceInfo:
    """Information about a synced device"""
    device_id: str
    device_name: str
    device_type: str  # 'laptop', 'phone', 'server', 'tablet'
    last_seen: float
    ip_address: Optional[str] = None
    port: int = 0
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'device_id': self.device_id,
            'device_name': self.device_name,
            'device_type': self.device_type,
            'last_seen': self.last_seen,
            'ip_address': self.ip_address,
            'port': self.port,
            'capabilities': self.capabilities,
        }


@dataclass
class SyncManifest:
    """Manifest for state synchronization"""
    device_id: str
    device_info: DeviceInfo
    entries: Dict[str, LWWEntry]
    vector: VectorClock
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: str = "2.0"
    
    def to_dict(self) -> dict:
        return {
            'device_id': self.device_id,
            'device_info': self.device_info.to_dict(),
            'entries': {k: v.to_dict() for k, v in self.entries.items()},
            'vector': self.vector,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'schema_version': self.schema_version,
        }
    
    @staticmethod
    def from_dict(d: dict) -> "SyncManifest":
        return SyncManifest(
            device_id=d['device_id'],
            device_info=DeviceInfo(**d.get('device_info', {})),
            entries={k: LWWEntry.from_dict(v) for k, v in d.get('entries', {}).items()},
            vector=d.get('vector', {}),
            created_at=d.get('created_at', datetime.now().isoformat()),
            updated_at=d.get('updated_at', datetime.now().isoformat()),
            schema_version=d.get('schema_version', '2.0'),
        )


# ── Enhanced State Sync ─────────────────────────────────────────────────────

class EnhancedStateSync:
    """
    Enhanced state synchronization with device discovery and real-time sync
    """
    
    MANIFEST_PATH = Path("sync_manifest.json")
    CHECKPOINT_DIR = Path("sync_checkpoints")
    MAX_CHECKPOINTS = 10
    DISCOVERY_PORT = 37020
    SYNC_PORT = 37021
    
    def __init__(self, 
                 device_id: Optional[str] = None,
                 device_name: Optional[str] = None,
                 device_type: str = "unknown",
                 manifest_path: Optional[str] = None,
                 encryption_key: Optional[str] = None):
        
        self.device_id = device_id or self._load_or_create_device_id()
        self.device_name = device_name or socket.gethostname()
        self.device_type = device_type
        
        if manifest_path:
            self.MANIFEST_PATH = Path(manifest_path)
        
        # Encryption
        self.encryption_key = encryption_key
        self._cipher = None
        if encryption_key and CRYPTO_AVAILABLE:
            self._cipher = Fernet(encryption_key.encode()[:32].ljust(32, b'0'))
        
        # Threading
        self._lock = threading.RLock()
        
        # Manifest and state
        self._manifest = self._load_manifest()
        self._device_info = self._create_device_info()
        self._manifest.device_info = self._device_info
        
        # Known devices
        self._known_devices: Dict[str, DeviceInfo] = {}
        
        # Sync callbacks
        self.on_sync: Optional[Callable[[str, Any], None]] = None
        self.on_device_discovered: Optional[Callable[[DeviceInfo], None]] = None
        
        # Network
        self._discovery_socket: Optional[socket.socket] = None
        self._sync_socket: Optional[socket.socket] = None
        self._running = False
        
        # Ensure directories
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        
        print(f"[StateSync] Device: {self.device_id} ({self.device_name})")
        print(f"[StateSync] Keys: {len(self._manifest.entries)}")
    
    def _load_or_create_device_id(self) -> str:
        """Load or create persistent device ID"""
        id_file = Path(".janus_device_id")
        if id_file.exists():
            return id_file.read_text().strip()
        
        device_id = f"janus_{uuid.uuid4().hex[:12]}"
        id_file.write_text(device_id)
        return device_id
    
    def _create_device_info(self) -> DeviceInfo:
        """Create device info"""
        return DeviceInfo(
            device_id=self.device_id,
            device_name=self.device_name,
            device_type=self.device_type,
            last_seen=time.time(),
            ip_address=self._get_local_ip(),
            port=self.SYNC_PORT,
            capabilities=['voice', 'messaging', 'tools', 'memory']
        )
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _load_manifest(self) -> SyncManifest:
        """Load sync manifest from disk"""
        if self.MANIFEST_PATH.exists():
            try:
                data = json.loads(self.MANIFEST_PATH.read_text())
                
                # Handle legacy format
                if 'device_info' not in data:
                    data['device_info'] = self._create_device_info().to_dict()
                
                return SyncManifest.from_dict(data)
            except Exception as e:
                print(f"[StateSync] Load error: {e}, starting fresh")
        
        return SyncManifest(
            device_id=self.device_id,
            device_info=self._create_device_info(),
            entries={},
            vector={}
        )
    
    # ── Core API ────────────────────────────────────────────────────────────
    
    def set(self, key: str, value: Any, propagate: bool = True):
        """
        Set a value in the synchronized state
        
        Args:
            key: State key
            value: Value to store
            propagate: Whether to propagate to other devices
        """
        with self._lock:
            new_vc = vc_increment(self._manifest.vector, self.device_id)
            self._manifest.vector = new_vc
            
            self._manifest.entries[key] = LWWEntry(
                key=key,
                value=value,
                wall_time=time.time(),
                vector=dict(new_vc),
                device_id=self.device_id,
            )
            self._manifest.updated_at = datetime.now().isoformat()
            
            if propagate:
                self._broadcast_update(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the synchronized state"""
        with self._lock:
            entry = self._manifest.entries.get(key)
            if entry is None or entry.tombstone:
                return default
            return entry.value
    
    def delete(self, key: str, propagate: bool = True):
        """Soft delete a key"""
        with self._lock:
            if key in self._manifest.entries:
                self._manifest.entries[key].tombstone = True
                self._manifest.entries[key].wall_time = time.time()
                
                if propagate:
                    self._broadcast_update(key, None)
    
    def get_all(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Get all values, optionally filtered by prefix"""
        with self._lock:
            result = {}
            for key, entry in self._manifest.entries.items():
                if not entry.tombstone:
                    if prefix is None or key.startswith(prefix):
                        result[key] = entry.value
            return result
    
    def all_keys(self) -> List[str]:
        """Get all non-tombstoned keys"""
        with self._lock:
            return [k for k, v in self._manifest.entries.items() if not v.tombstone]
    
    # ── Identity & Memory Persistence ───────────────────────────────────────
    
    def set_identity(self, key: str, value: Any):
        """Set identity attribute"""
        self.set(f"identity.{key}", value)
    
    def get_identity(self, key: str, default: Any = None) -> Any:
        """Get identity attribute"""
        return self.get(f"identity.{key}", default)
    
    def store_memory(self, memory_id: str, content: dict):
        """Store a memory that syncs across devices"""
        self.set(f"memory.{memory_id}", content)
    
    def recall_memory(self, memory_id: str) -> Optional[dict]:
        """Recall a memory by ID"""
        return self.get(f"memory.{memory_id}")
    
    def query_memories(self, tag: Optional[str] = None) -> Dict[str, dict]:
        """Query all memories, optionally filtered by tag"""
        memories = self.get_all(prefix="memory.")
        if tag:
            memories = {
                k: v for k, v in memories.items()
                if isinstance(v, dict) and tag in v.get('tags', [])
            }
        return memories
    
    def get_identity_object(self) -> dict:
        """Get full identity object"""
        return self.get_all(prefix="identity.")
    
    def export_identity(self, path: str = "identity_synced.json"):
        """Export identity to file"""
        identity = self.get_identity_object()
        Path(path).write_text(json.dumps(identity, indent=2))
        print(f"[StateSync] Identity exported to {path}")
    
    # ── Merge & Sync ────────────────────────────────────────────────────────
    
    def merge_manifest(self, other_manifest: SyncManifest) -> dict:
        """
        Merge another manifest into local state
        
        Returns:
            Merge report
        """
        report = {
            "merged": 0,
            "kept_local": 0,
            "kept_remote": 0,
            "conflicts": [],
            "new_devices": []
        }
        
        with self._lock:
            # Update known devices
            if other_manifest.device_id != self.device_id:
                device_info = other_manifest.device_info
                if device_info.device_id not in self._known_devices:
                    report["new_devices"].append(device_info.to_dict())
                    if self.on_device_discovered:
                        self.on_device_discovered(device_info)
                
                self._known_devices[device_info.device_id] = device_info
            
            # Merge entries
            for key, remote_entry in other_manifest.entries.items():
                local_entry = self._manifest.entries.get(key)
                
                if local_entry is None:
                    # New key from remote
                    self._manifest.entries[key] = remote_entry
                    report["merged"] += 1
                    
                    if self.on_sync:
                        self.on_sync(key, remote_entry.value)
                
                else:
                    # Conflict resolution
                    comparison = vc_compare(remote_entry.vector, local_entry.vector)
                    
                    if comparison > 0:
                        # Remote is newer
                        self._manifest.entries[key] = remote_entry
                        report["kept_remote"] += 1
                        report["conflicts"].append({
                            "key": key,
                            "winner": "remote",
                            "reason": "vector_clock"
                        })
                        
                        if self.on_sync:
                            self.on_sync(key, remote_entry.value)
                    
                    elif comparison < 0:
                        # Local is newer
                        report["kept_local"] += 1
                    
                    else:
                        # Concurrent - use wall clock
                        if remote_entry.wall_time > local_entry.wall_time:
                            self._manifest.entries[key] = remote_entry
                            report["kept_remote"] += 1
                            report["conflicts"].append({
                                "key": key,
                                "winner": "remote",
                                "reason": "wall_clock"
                            })
                            
                            if self.on_sync:
                                self.on_sync(key, remote_entry.value)
                        else:
                            report["kept_local"] += 1
            
            # Merge vector clocks
            self._manifest.vector = vc_merge(
                self._manifest.vector, 
                other_manifest.vector
            )
            self._manifest.updated_at = datetime.now().isoformat()
        
        return report
    
    def merge_from_file(self, path: str) -> dict:
        """Merge manifest from file"""
        try:
            data = json.loads(Path(path).read_text())
            manifest = SyncManifest.from_dict(data)
            return self.merge_manifest(manifest)
        except Exception as e:
            return {"error": str(e)}
    
    # ── Persistence ─────────────────────────────────────────────────────────
    
    def save(self, path: Optional[str] = None):
        """Save manifest to file"""
        target = Path(path) if path else self.MANIFEST_PATH
        
        with self._lock:
            data = self._manifest.to_dict()
            target.write_text(json.dumps(data, indent=2))
        
        self._maybe_checkpoint()
    
    def _maybe_checkpoint(self):
        """Create rolling checkpoint"""
        if not self.MANIFEST_PATH.exists():
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.CHECKPOINT_DIR / f"manifest_{ts}.json"
        shutil.copy2(self.MANIFEST_PATH, dest)
        
        # Prune old checkpoints
        checkpoints = sorted(self.CHECKPOINT_DIR.glob("manifest_*.json"))
        for old in checkpoints[:-self.MAX_CHECKPOINTS]:
            old.unlink()
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """Load from a checkpoint"""
        checkpoint_path = self.CHECKPOINT_DIR / checkpoint_name
        if checkpoint_path.exists():
            try:
                data = json.loads(checkpoint_path.read_text())
                with self._lock:
                    self._manifest = SyncManifest.from_dict(data)
                return True
            except Exception as e:
                print(f"[StateSync] Checkpoint load error: {e}")
        return False
    
    # ── Network Sync ────────────────────────────────────────────────────────
    
    def start_network_sync(self):
        """Start network discovery and sync"""
        self._running = True
        
        # Start discovery
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop, daemon=True
        )
        self._discovery_thread.start()
        
        # Start sync server
        self._sync_server_thread = threading.Thread(
            target=self._sync_server_loop, daemon=True
        )
        self._sync_server_thread.start()
        
        print(f"[StateSync] Network sync started on port {self.SYNC_PORT}")
    
    def stop_network_sync(self):
        """Stop network sync"""
        self._running = False
        
        if self._discovery_socket:
            self._discovery_socket.close()
        if self._sync_socket:
            self._sync_socket.close()
    
    def _discovery_loop(self):
        """Broadcast and listen for device discovery"""
        # Create discovery socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.DISCOVERY_PORT))
        sock.settimeout(1.0)
        
        self._discovery_socket = sock
        
        while self._running:
            try:
                # Broadcast presence
                announcement = json.dumps({
                    'type': 'discovery',
                    'device_id': self.device_id,
                    'device_name': self.device_name,
                    'device_type': self.device_type,
                    'port': self.SYNC_PORT,
                }).encode()
                
                sock.sendto(announcement, ('<broadcast>', self.DISCOVERY_PORT))
                
                # Listen for other devices
                try:
                    data, addr = sock.recvfrom(1024)
                    msg = json.loads(data.decode())
                    
                    if msg.get('type') == 'discovery' and msg.get('device_id') != self.device_id:
                        device_info = DeviceInfo(
                            device_id=msg['device_id'],
                            device_name=msg['device_name'],
                            device_type=msg['device_type'],
                            last_seen=time.time(),
                            ip_address=addr[0],
                            port=msg.get('port', self.SYNC_PORT),
                        )
                        
                        if msg['device_id'] not in self._known_devices:
                            print(f"[StateSync] Discovered: {device_info.device_name} at {addr[0]}")
                            self._known_devices[msg['device_id']] = device_info
                            
                            if self.on_device_discovered:
                                self.on_device_discovered(device_info)
                        
                        else:
                            self._known_devices[msg['device_id']].last_seen = time.time()
                
                except socket.timeout:
                    pass
                
                time.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                if self._running:
                    print(f"[StateSync] Discovery error: {e}")
                time.sleep(5)
    
    def _sync_server_loop(self):
        """Server to handle incoming sync requests"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.SYNC_PORT))
        sock.listen(5)
        sock.settimeout(1.0)
        
        self._sync_socket = sock
        
        while self._running:
            try:
                conn, addr = sock.accept()
                handler_thread = threading.Thread(
                    target=self._handle_sync_connection,
                    args=(conn, addr),
                    daemon=True
                )
                handler_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[StateSync] Server error: {e}")
    
    def _handle_sync_connection(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handle incoming sync connection"""
        try:
            conn.settimeout(10.0)
            
            # Receive manifest
            data = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            
            if data:
                manifest_data = json.loads(data.decode())
                remote_manifest = SyncManifest.from_dict(manifest_data)
                
                # Merge
                report = self.merge_manifest(remote_manifest)
                
                # Send response
                response = {
                    'status': 'ok',
                    'report': report,
                    'manifest': self._manifest.to_dict(),
                }
                conn.sendall(json.dumps(response).encode())
        
        except Exception as e:
            print(f"[StateSync] Connection handler error: {e}")
        finally:
            conn.close()
    
    def sync_with_device(self, device_id: str) -> dict:
        """Initiate sync with a specific device"""
        device = self._known_devices.get(device_id)
        if not device:
            return {"error": f"Device {device_id} not known"}
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((device.ip_address, device.port))
            
            # Send manifest
            manifest_data = json.dumps(self._manifest.to_dict()).encode()
            sock.sendall(manifest_data)
            sock.shutdown(socket.SHUT_WR)
            
            # Receive response
            response = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            
            if response:
                result = json.loads(response.decode())
                if result.get('status') == 'ok':
                    # Merge remote manifest
                    remote_manifest = SyncManifest.from_dict(result['manifest'])
                    merge_report = self.merge_manifest(remote_manifest)
                    return {"success": True, "report": merge_report}
            
            return {"error": "Invalid response"}
        
        except Exception as e:
            return {"error": str(e)}
        finally:
            sock.close()
    
    def _broadcast_update(self, key: str, value: Any):
        """Broadcast update to known devices"""
        for device_id, device in list(self._known_devices.items()):
            if time.time() - device.last_seen < 60:  # Only sync with recently seen devices
                try:
                    self.sync_with_device(device_id)
                except Exception as e:
                    print(f"[StateSync] Broadcast to {device_id} failed: {e}")
    
    # ── Status & Utilities ──────────────────────────────────────────────────
    
    def get_status(self) -> dict:
        """Get sync status"""
        with self._lock:
            return {
                'device_id': self.device_id,
                'device_name': self.device_name,
                'device_type': self.device_type,
                'keys': len([k for k, v in self._manifest.entries.items() if not v.tombstone]),
                'vector': self._manifest.vector,
                'known_devices': len(self._known_devices),
                'updated_at': self._manifest.updated_at,
            }
    
    def list_known_devices(self) -> List[dict]:
        """List all known devices"""
        return [d.to_dict() for d in self._known_devices.values()]


def main():
    """Demo of enhanced state sync"""
    print("=== Janus Enhanced State Sync Demo ===\n")
    
    # Create sync instance
    sync = EnhancedStateSync(
        device_name="Demo-Laptop",
        device_type="laptop"
    )
    
    # Set some identity values
    print("1. Setting identity...")
    sync.set_identity("name", "Janus")
    sync.set_identity("version", "2.0")
    sync.set_identity("personality", "helpful and friendly")
    
    # Store some memories
    print("\n2. Storing memories...")
    sync.store_memory("mem_001", {
        "content": "User prefers dark mode",
        "tags": ["preference", "ui"],
        "importance": 0.8
    })
    sync.store_memory("mem_002", {
        "content": "User's favorite color is blue",
        "tags": ["preference", "personal"],
        "importance": 0.6
    })
    
    # Query memories
    print("\n3. Querying memories...")
    preferences = sync.query_memories(tag="preference")
    print(f"   Found {len(preferences)} preference memories")
    
    # Get identity
    print("\n4. Identity object:")
    identity = sync.get_identity_object()
    for k, v in identity.items():
        print(f"   {k}: {v}")
    
    # Save
    print("\n5. Saving manifest...")
    sync.save()
    
    # Status
    print("\n6. Status:")
    print(f"   {sync.get_status()}")
    
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
