"""
Working memory system for the Janus Reasoning Engine.

Manages active context, attention, and multiple threads of thought during task execution.
Provides resume-after-interruption capability.

**Validates: Requirements REQ-6.3**
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import uuid


class AttentionLevel(Enum):
    """Attention levels for context items."""
    FOCUSED = "focused"  # Primary focus
    ACTIVE = "active"  # Active but not primary
    BACKGROUND = "background"  # Background awareness
    SUSPENDED = "suspended"  # Temporarily suspended


class ThreadStatus(Enum):
    """Status of thought threads."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ContextItem:
    """
    An item in working memory context.
    
    Represents a piece of information, task, or thought that's currently active.
    """
    item_id: str
    content: Dict[str, Any]
    attention_level: AttentionLevel
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Record an access to this context item."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class ThoughtThread:
    """
    A thread of thought or reasoning.
    
    Represents a line of reasoning, task execution, or problem-solving process.
    """
    thread_id: str
    name: str
    status: ThreadStatus
    context_items: List[str]  # IDs of context items
    parent_thread: Optional[str] = None  # Parent thread ID
    child_threads: List[str] = field(default_factory=list)  # Child thread IDs
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self) -> None:
        """Update the thread's timestamp."""
        self.updated_at = datetime.now()
    
    def complete(self, success: bool = True) -> None:
        """Mark the thread as completed."""
        self.status = ThreadStatus.COMPLETED if success else ThreadStatus.FAILED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()


@dataclass
class InterruptionCheckpoint:
    """
    Checkpoint for resuming after interruption.
    
    Stores the state needed to resume work after an interruption.
    """
    checkpoint_id: str
    active_threads: List[str]  # Thread IDs
    focused_items: List[str]  # Context item IDs
    attention_state: Dict[str, AttentionLevel]  # item_id -> attention level
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """
    Working memory system for managing active context and attention.
    
    Provides:
    - Context management for active tasks
    - Multiple thread tracking
    - Attention mechanism for focus management
    - Resume-after-interruption capability
    """
    
    def __init__(
        self,
        max_context_items: int = 50,
        max_focused_items: int = 5,
        max_active_threads: int = 10,
    ):
        """
        Initialize working memory.
        
        Args:
            max_context_items: Maximum number of context items to maintain
            max_focused_items: Maximum number of items that can be focused
            max_active_threads: Maximum number of active threads
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_context_items = max_context_items
        self.max_focused_items = max_focused_items
        self.max_active_threads = max_active_threads
        
        # Context storage
        self.context_items: Dict[str, ContextItem] = {}
        self.threads: Dict[str, ThoughtThread] = {}
        self.checkpoints: Dict[str, InterruptionCheckpoint] = {}
        
        # Attention tracking
        self.focused_items: Set[str] = set()
        self.active_thread_ids: Set[str] = set()
        
        self.logger.info(
            f"Working memory initialized: max_context={max_context_items}, "
            f"max_focused={max_focused_items}, max_threads={max_active_threads}"
        )
    
    # Context Management
    
    def add_context(
        self,
        content: Dict[str, Any],
        attention_level: AttentionLevel = AttentionLevel.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an item to working memory context.
        
        Args:
            content: Content of the context item
            attention_level: Initial attention level
            metadata: Optional metadata
            
        Returns:
            Context item ID
        """
        # Check capacity
        if len(self.context_items) >= self.max_context_items:
            self._evict_context_item()
        
        # Create context item
        item_id = str(uuid.uuid4())
        now = datetime.now()
        
        item = ContextItem(
            item_id=item_id,
            content=content,
            attention_level=attention_level,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
        )
        
        self.context_items[item_id] = item
        
        # Update attention tracking
        if attention_level == AttentionLevel.FOCUSED:
            self._add_to_focus(item_id)
        
        self.logger.debug(f"Added context item {item_id} with attention {attention_level.value}")
        return item_id
    
    def get_context(self, item_id: str) -> Optional[ContextItem]:
        """
        Get a context item by ID.
        
        Args:
            item_id: Context item ID
            
        Returns:
            Context item or None if not found
        """
        item = self.context_items.get(item_id)
        if item:
            item.access()
        return item
    
    def update_context(
        self,
        item_id: str,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a context item.
        
        Args:
            item_id: Context item ID
            content: New content (if provided)
            metadata: New metadata (if provided)
        """
        item = self.context_items.get(item_id)
        if not item:
            self.logger.warning(f"Context item {item_id} not found")
            return
        
        if content is not None:
            item.content = content
        
        if metadata is not None:
            item.metadata.update(metadata)
        
        item.access()
        self.logger.debug(f"Updated context item {item_id}")
    
    def remove_context(self, item_id: str) -> None:
        """
        Remove a context item.
        
        Args:
            item_id: Context item ID
        """
        if item_id in self.context_items:
            del self.context_items[item_id]
            self.focused_items.discard(item_id)
            self.logger.debug(f"Removed context item {item_id}")
    
    def get_all_context(
        self,
        attention_level: Optional[AttentionLevel] = None,
    ) -> List[ContextItem]:
        """
        Get all context items, optionally filtered by attention level.
        
        Args:
            attention_level: Filter by attention level (optional)
            
        Returns:
            List of context items
        """
        items = list(self.context_items.values())
        
        if attention_level:
            items = [item for item in items if item.attention_level == attention_level]
        
        # Sort by last accessed (most recent first)
        items.sort(key=lambda x: x.last_accessed, reverse=True)
        
        return items
    
    def _evict_context_item(self) -> None:
        """Evict the least recently used context item."""
        if not self.context_items:
            return
        
        # Find least recently used item that's not focused
        non_focused = [
            item for item in self.context_items.values()
            if item.item_id not in self.focused_items
        ]
        
        if not non_focused:
            # All items are focused, evict oldest focused item
            non_focused = list(self.context_items.values())
        
        # Sort by last accessed (oldest first)
        non_focused.sort(key=lambda x: x.last_accessed)
        
        # Evict oldest
        to_evict = non_focused[0]
        self.remove_context(to_evict.item_id)
        self.logger.debug(f"Evicted context item {to_evict.item_id}")
    
    # Attention Management
    
    def set_attention(self, item_id: str, attention_level: AttentionLevel) -> None:
        """
        Set the attention level for a context item.
        
        Args:
            item_id: Context item ID
            attention_level: New attention level
        """
        item = self.context_items.get(item_id)
        if not item:
            self.logger.warning(f"Context item {item_id} not found")
            return
        
        old_level = item.attention_level
        item.attention_level = attention_level
        item.access()
        
        # Update focus tracking
        if attention_level == AttentionLevel.FOCUSED:
            self._add_to_focus(item_id)
        elif old_level == AttentionLevel.FOCUSED:
            self.focused_items.discard(item_id)
        
        self.logger.debug(
            f"Changed attention for {item_id}: {old_level.value} -> {attention_level.value}"
        )
    
    def focus_on(self, item_id: str) -> None:
        """
        Focus attention on a context item.
        
        Args:
            item_id: Context item ID
        """
        self.set_attention(item_id, AttentionLevel.FOCUSED)
    
    def unfocus(self, item_id: str) -> None:
        """
        Remove focus from a context item (set to active).
        
        Args:
            item_id: Context item ID
        """
        self.set_attention(item_id, AttentionLevel.ACTIVE)
    
    def get_focused_items(self) -> List[ContextItem]:
        """
        Get all focused context items.
        
        Returns:
            List of focused items
        """
        return [
            self.context_items[item_id]
            for item_id in self.focused_items
            if item_id in self.context_items
        ]
    
    def _add_to_focus(self, item_id: str) -> None:
        """Add an item to focus, managing capacity."""
        # Check focus capacity
        if len(self.focused_items) >= self.max_focused_items:
            # Unfocus oldest focused item
            oldest_focused = None
            oldest_time = datetime.now()
            
            for fid in self.focused_items:
                item = self.context_items.get(fid)
                if item and item.last_accessed < oldest_time:
                    oldest_time = item.last_accessed
                    oldest_focused = fid
            
            if oldest_focused:
                self.focused_items.discard(oldest_focused)
                if oldest_focused in self.context_items:
                    self.context_items[oldest_focused].attention_level = AttentionLevel.ACTIVE
                self.logger.debug(f"Unfocused {oldest_focused} to make room")
        
        self.focused_items.add(item_id)
    
    # Thread Management
    
    def create_thread(
        self,
        name: str,
        parent_thread: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new thought thread.
        
        Args:
            name: Thread name
            parent_thread: Parent thread ID (optional)
            metadata: Optional metadata
            
        Returns:
            Thread ID
        """
        # Check capacity
        active_count = len(self.active_thread_ids)
        if active_count >= self.max_active_threads:
            self.logger.warning(
                f"Maximum active threads ({self.max_active_threads}) reached"
            )
            # Pause oldest active thread
            self._pause_oldest_thread()
        
        # Create thread
        thread_id = str(uuid.uuid4())
        
        thread = ThoughtThread(
            thread_id=thread_id,
            name=name,
            status=ThreadStatus.ACTIVE,
            context_items=[],
            parent_thread=parent_thread,
            metadata=metadata or {},
        )
        
        self.threads[thread_id] = thread
        self.active_thread_ids.add(thread_id)
        
        # Update parent
        if parent_thread and parent_thread in self.threads:
            self.threads[parent_thread].child_threads.append(thread_id)
        
        self.logger.info(f"Created thread {thread_id}: {name}")
        return thread_id
    
    def get_thread(self, thread_id: str) -> Optional[ThoughtThread]:
        """
        Get a thread by ID.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            Thread or None if not found
        """
        return self.threads.get(thread_id)
    
    def update_thread(
        self,
        thread_id: str,
        status: Optional[ThreadStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a thread.
        
        Args:
            thread_id: Thread ID
            status: New status (optional)
            metadata: New metadata (optional)
        """
        thread = self.threads.get(thread_id)
        if not thread:
            self.logger.warning(f"Thread {thread_id} not found")
            return
        
        if status is not None:
            old_status = thread.status
            thread.status = status
            
            # Update active tracking
            if status == ThreadStatus.ACTIVE:
                self.active_thread_ids.add(thread_id)
            else:
                self.active_thread_ids.discard(thread_id)
            
            self.logger.debug(
                f"Thread {thread_id} status: {old_status.value} -> {status.value}"
            )
        
        if metadata is not None:
            thread.metadata.update(metadata)
        
        thread.update()
    
    def add_context_to_thread(self, thread_id: str, context_id: str) -> None:
        """
        Add a context item to a thread.
        
        Args:
            thread_id: Thread ID
            context_id: Context item ID
        """
        thread = self.threads.get(thread_id)
        if not thread:
            self.logger.warning(f"Thread {thread_id} not found")
            return
        
        if context_id not in thread.context_items:
            thread.context_items.append(context_id)
            thread.update()
            self.logger.debug(f"Added context {context_id} to thread {thread_id}")
    
    def get_thread_context(self, thread_id: str) -> List[ContextItem]:
        """
        Get all context items for a thread.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            List of context items
        """
        thread = self.threads.get(thread_id)
        if not thread:
            return []
        
        return [
            self.context_items[cid]
            for cid in thread.context_items
            if cid in self.context_items
        ]
    
    def complete_thread(self, thread_id: str, success: bool = True) -> None:
        """
        Mark a thread as completed.
        
        Args:
            thread_id: Thread ID
            success: Whether the thread completed successfully
        """
        thread = self.threads.get(thread_id)
        if not thread:
            self.logger.warning(f"Thread {thread_id} not found")
            return
        
        thread.complete(success)
        self.active_thread_ids.discard(thread_id)
        
        status_str = "completed" if success else "failed"
        self.logger.info(f"Thread {thread_id} {status_str}")
    
    def get_active_threads(self) -> List[ThoughtThread]:
        """
        Get all active threads.
        
        Returns:
            List of active threads
        """
        return [
            self.threads[tid]
            for tid in self.active_thread_ids
            if tid in self.threads
        ]
    
    def _pause_oldest_thread(self) -> None:
        """Pause the oldest active thread."""
        if not self.active_thread_ids:
            return
        
        # Find oldest active thread
        oldest_thread = None
        oldest_time = datetime.now()
        
        for tid in self.active_thread_ids:
            thread = self.threads.get(tid)
            if thread and thread.updated_at < oldest_time:
                oldest_time = thread.updated_at
                oldest_thread = tid
        
        if oldest_thread:
            self.update_thread(oldest_thread, status=ThreadStatus.PAUSED)
            self.logger.info(f"Paused thread {oldest_thread} to make room")
    
    # Interruption and Resume
    
    def create_checkpoint(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a checkpoint for resuming after interruption.
        
        Args:
            metadata: Optional metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())
        
        # Capture current state
        attention_state = {
            item_id: item.attention_level
            for item_id, item in self.context_items.items()
        }
        
        checkpoint = InterruptionCheckpoint(
            checkpoint_id=checkpoint_id,
            active_threads=list(self.active_thread_ids),
            focused_items=list(self.focused_items),
            attention_state=attention_state,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        
        self.logger.info(
            f"Created checkpoint {checkpoint_id}: "
            f"{len(checkpoint.active_threads)} threads, "
            f"{len(checkpoint.focused_items)} focused items"
        )
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            True if restored successfully
        """
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            self.logger.warning(f"Checkpoint {checkpoint_id} not found")
            return False
        
        # Restore active threads
        self.active_thread_ids = set(checkpoint.active_threads)
        for tid in checkpoint.active_threads:
            if tid in self.threads:
                self.threads[tid].status = ThreadStatus.ACTIVE
        
        # Restore focused items
        self.focused_items = set(checkpoint.focused_items)
        
        # Restore attention state
        for item_id, attention_level in checkpoint.attention_state.items():
            if item_id in self.context_items:
                self.context_items[item_id].attention_level = attention_level
        
        self.logger.info(f"Restored checkpoint {checkpoint_id}")
        return True
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[InterruptionCheckpoint]:
        """
        Get a checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint or None if not found
        """
        return self.checkpoints.get(checkpoint_id)
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
        """
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            self.logger.debug(f"Deleted checkpoint {checkpoint_id}")
    
    # Statistics and Introspection
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get working memory statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "context_items": {
                "total": len(self.context_items),
                "focused": len(self.focused_items),
                "active": len([
                    item for item in self.context_items.values()
                    if item.attention_level == AttentionLevel.ACTIVE
                ]),
                "background": len([
                    item for item in self.context_items.values()
                    if item.attention_level == AttentionLevel.BACKGROUND
                ]),
                "suspended": len([
                    item for item in self.context_items.values()
                    if item.attention_level == AttentionLevel.SUSPENDED
                ]),
            },
            "threads": {
                "total": len(self.threads),
                "active": len(self.active_thread_ids),
                "paused": len([
                    t for t in self.threads.values()
                    if t.status == ThreadStatus.PAUSED
                ]),
                "completed": len([
                    t for t in self.threads.values()
                    if t.status == ThreadStatus.COMPLETED
                ]),
                "failed": len([
                    t for t in self.threads.values()
                    if t.status == ThreadStatus.FAILED
                ]),
            },
            "checkpoints": len(self.checkpoints),
            "capacity": {
                "max_context_items": self.max_context_items,
                "max_focused_items": self.max_focused_items,
                "max_active_threads": self.max_active_threads,
            },
        }
    
    def clear(self) -> None:
        """Clear all working memory (for testing or reset)."""
        self.context_items.clear()
        self.threads.clear()
        self.checkpoints.clear()
        self.focused_items.clear()
        self.active_thread_ids.clear()
        self.logger.info("Working memory cleared")
