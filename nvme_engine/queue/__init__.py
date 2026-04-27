"""
NVMe Queue Processor package.

Provides submission queues, completion queues, and the queue processor
that drives I/O through the storage backend.
"""

from nvme_engine.queue.completion_queue import CompletionQueue
from nvme_engine.queue.queue_processor import PollingMode, QueueProcessor
from nvme_engine.queue.submission_queue import SubmissionQueue

__all__ = [
    "SubmissionQueue",
    "CompletionQueue",
    "PollingMode",
    "QueueProcessor",
]
