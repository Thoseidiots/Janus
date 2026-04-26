"""
Fault Tolerance module for the Software NVMe Engine.

Provides N-way replication, automatic failover, and data integrity
verification for storage backends.
"""

from nvme_engine.fault_tolerance.replication import ReplicatedBackend, ReplicationConfig

__all__ = ["ReplicatedBackend", "ReplicationConfig"]
