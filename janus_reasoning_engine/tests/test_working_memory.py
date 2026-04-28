"""
Unit tests for the working memory system.

Tests context management, attention mechanisms, thread tracking,
and interruption/resume capabilities.
"""

import pytest
from datetime import datetime, timedelta
import time

from janus_reasoning_engine.memory.working_memory import (
    WorkingMemory,
    ContextItem,
    ThoughtThread,
    InterruptionCheckpoint,
    AttentionLevel,
    ThreadStatus,
)


class TestContextManagement:
    """Test context item management."""
    
    def test_add_context(self):
        """Test adding context items."""
        wm = WorkingMemory(max_context_items=10)
        
        content = {"task": "test task", "data": "test data"}
        item_id = wm.add_context(content, AttentionLevel.ACTIVE)
        
        assert item_id is not None
        assert len(wm.context_items) == 1
        
        item = wm.get_context(item_id)
        assert item is not None
        assert item.content == content
        assert item.attention_level == AttentionLevel.ACTIVE
    
    def test_get_context(self):
        """Test retrieving context items."""
        wm = WorkingMemory()
        
        content = {"task": "test"}
        item_id = wm.add_context(content)
        
        item = wm.get_context(item_id)
        assert item is not None
        assert item.content == content
        assert item.access_count == 1  # Accessed once by get_context
    
    def test_update_context(self):
        """Test updating context items."""
        wm = WorkingMemory()
        
        item_id = wm.add_context({"task": "original"})
        
        new_content = {"task": "updated"}
        new_metadata = {"updated": True}
        wm.update_context(item_id, content=new_content, metadata=new_metadata)
        
        item = wm.get_context(item_id)
        assert item.content == new_content
        assert item.metadata["updated"] is True
    
    def test_remove_context(self):
        """Test removing context items."""
        wm = WorkingMemory()
        
        item_id = wm.add_context({"task": "test"})
        assert len(wm.context_items) == 1
        
        wm.remove_context(item_id)
        assert len(wm.context_items) == 0
        assert wm.get_context(item_id) is None
    
    def test_get_all_context(self):
        """Test retrieving all context items."""
        wm = WorkingMemory()
        
        # Add items with different attention levels
        wm.add_context({"task": "1"}, AttentionLevel.FOCUSED)
        wm.add_context({"task": "2"}, AttentionLevel.ACTIVE)
        wm.add_context({"task": "3"}, AttentionLevel.BACKGROUND)
        
        all_items = wm.get_all_context()
        assert len(all_items) == 3
        
        focused_items = wm.get_all_context(AttentionLevel.FOCUSED)
        assert len(focused_items) == 1
        assert focused_items[0].content["task"] == "1"
    
    def test_context_capacity_eviction(self):
        """Test that context items are evicted when capacity is reached."""
        wm = WorkingMemory(max_context_items=3)
        
        # Add 3 items
        id1 = wm.add_context({"task": "1"})
        id2 = wm.add_context({"task": "2"})
        id3 = wm.add_context({"task": "3"})
        
        assert len(wm.context_items) == 3
        
        # Add 4th item - should evict oldest
        time.sleep(0.01)  # Ensure time difference
        id4 = wm.add_context({"task": "4"})
        
        assert len(wm.context_items) == 3
        assert wm.get_context(id1) is None  # Oldest evicted
        assert wm.get_context(id4) is not None  # New item present


class TestAttentionManagement:
    """Test attention mechanism."""
    
    def test_set_attention(self):
        """Test setting attention levels."""
        wm = WorkingMemory()
        
        item_id = wm.add_context({"task": "test"}, AttentionLevel.ACTIVE)
        
        wm.set_attention(item_id, AttentionLevel.FOCUSED)
        item = wm.get_context(item_id)
        assert item.attention_level == AttentionLevel.FOCUSED
        assert item_id in wm.focused_items
    
    def test_focus_on(self):
        """Test focusing on items."""
        wm = WorkingMemory()
        
        item_id = wm.add_context({"task": "test"}, AttentionLevel.ACTIVE)
        
        wm.focus_on(item_id)
        assert item_id in wm.focused_items
        
        item = wm.get_context(item_id)
        assert item.attention_level == AttentionLevel.FOCUSED
    
    def test_unfocus(self):
        """Test unfocusing items."""
        wm = WorkingMemory()
        
        item_id = wm.add_context({"task": "test"}, AttentionLevel.FOCUSED)
        assert item_id in wm.focused_items
        
        wm.unfocus(item_id)
        assert item_id not in wm.focused_items
        
        item = wm.get_context(item_id)
        assert item.attention_level == AttentionLevel.ACTIVE
    
    def test_get_focused_items(self):
        """Test retrieving focused items."""
        wm = WorkingMemory()
        
        id1 = wm.add_context({"task": "1"}, AttentionLevel.FOCUSED)
        id2 = wm.add_context({"task": "2"}, AttentionLevel.ACTIVE)
        id3 = wm.add_context({"task": "3"}, AttentionLevel.FOCUSED)
        
        focused = wm.get_focused_items()
        assert len(focused) == 2
        
        focused_ids = {item.item_id for item in focused}
        assert id1 in focused_ids
        assert id3 in focused_ids
        assert id2 not in focused_ids
    
    def test_focus_capacity_management(self):
        """Test that focus capacity is managed correctly."""
        wm = WorkingMemory(max_focused_items=2)
        
        # Add 2 focused items
        id1 = wm.add_context({"task": "1"}, AttentionLevel.FOCUSED)
        time.sleep(0.01)
        id2 = wm.add_context({"task": "2"}, AttentionLevel.FOCUSED)
        
        assert len(wm.focused_items) == 2
        
        # Add 3rd focused item - should unfocus oldest
        time.sleep(0.01)
        id3 = wm.add_context({"task": "3"}, AttentionLevel.FOCUSED)
        
        assert len(wm.focused_items) == 2
        assert id1 not in wm.focused_items  # Oldest unfocused
        assert id2 in wm.focused_items
        assert id3 in wm.focused_items
        
        # Check that unfocused item is now active
        item1 = wm.get_context(id1)
        assert item1.attention_level == AttentionLevel.ACTIVE


class TestThreadManagement:
    """Test thought thread management."""
    
    def test_create_thread(self):
        """Test creating threads."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test thread")
        
        assert thread_id is not None
        assert len(wm.threads) == 1
        assert thread_id in wm.active_thread_ids
        
        thread = wm.get_thread(thread_id)
        assert thread is not None
        assert thread.name == "test thread"
        assert thread.status == ThreadStatus.ACTIVE
    
    def test_create_child_thread(self):
        """Test creating child threads."""
        wm = WorkingMemory()
        
        parent_id = wm.create_thread("parent")
        child_id = wm.create_thread("child", parent_thread=parent_id)
        
        parent = wm.get_thread(parent_id)
        child = wm.get_thread(child_id)
        
        assert child.parent_thread == parent_id
        assert child_id in parent.child_threads
    
    def test_update_thread(self):
        """Test updating threads."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test")
        
        wm.update_thread(thread_id, status=ThreadStatus.PAUSED, metadata={"paused": True})
        
        thread = wm.get_thread(thread_id)
        assert thread.status == ThreadStatus.PAUSED
        assert thread.metadata["paused"] is True
        assert thread_id not in wm.active_thread_ids
    
    def test_add_context_to_thread(self):
        """Test adding context to threads."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test")
        context_id = wm.add_context({"task": "test"})
        
        wm.add_context_to_thread(thread_id, context_id)
        
        thread = wm.get_thread(thread_id)
        assert context_id in thread.context_items
    
    def test_get_thread_context(self):
        """Test retrieving thread context."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test")
        
        id1 = wm.add_context({"task": "1"})
        id2 = wm.add_context({"task": "2"})
        
        wm.add_context_to_thread(thread_id, id1)
        wm.add_context_to_thread(thread_id, id2)
        
        context = wm.get_thread_context(thread_id)
        assert len(context) == 2
        
        context_ids = {item.item_id for item in context}
        assert id1 in context_ids
        assert id2 in context_ids
    
    def test_complete_thread(self):
        """Test completing threads."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test")
        assert thread_id in wm.active_thread_ids
        
        wm.complete_thread(thread_id, success=True)
        
        thread = wm.get_thread(thread_id)
        assert thread.status == ThreadStatus.COMPLETED
        assert thread.completed_at is not None
        assert thread_id not in wm.active_thread_ids
    
    def test_complete_thread_failure(self):
        """Test completing threads with failure."""
        wm = WorkingMemory()
        
        thread_id = wm.create_thread("test")
        wm.complete_thread(thread_id, success=False)
        
        thread = wm.get_thread(thread_id)
        assert thread.status == ThreadStatus.FAILED
    
    def test_get_active_threads(self):
        """Test retrieving active threads."""
        wm = WorkingMemory()
        
        id1 = wm.create_thread("thread1")
        id2 = wm.create_thread("thread2")
        id3 = wm.create_thread("thread3")
        
        wm.update_thread(id2, status=ThreadStatus.PAUSED)
        
        active = wm.get_active_threads()
        assert len(active) == 2
        
        active_ids = {thread.thread_id for thread in active}
        assert id1 in active_ids
        assert id3 in active_ids
        assert id2 not in active_ids
    
    def test_thread_capacity_management(self):
        """Test that thread capacity is managed correctly."""
        wm = WorkingMemory(max_active_threads=2)
        
        # Create 2 active threads
        id1 = wm.create_thread("thread1")
        time.sleep(0.01)
        id2 = wm.create_thread("thread2")
        
        assert len(wm.active_thread_ids) == 2
        
        # Create 3rd thread - should pause oldest
        time.sleep(0.01)
        id3 = wm.create_thread("thread3")
        
        assert len(wm.active_thread_ids) == 2
        assert id1 not in wm.active_thread_ids  # Oldest paused
        assert id2 in wm.active_thread_ids
        assert id3 in wm.active_thread_ids
        
        # Check that paused thread has correct status
        thread1 = wm.get_thread(id1)
        assert thread1.status == ThreadStatus.PAUSED


class TestInterruptionResume:
    """Test interruption and resume capabilities."""
    
    def test_create_checkpoint(self):
        """Test creating checkpoints."""
        wm = WorkingMemory()
        
        # Set up some state
        thread_id = wm.create_thread("test")
        item_id = wm.add_context({"task": "test"}, AttentionLevel.FOCUSED)
        wm.add_context_to_thread(thread_id, item_id)
        
        # Create checkpoint
        checkpoint_id = wm.create_checkpoint(metadata={"reason": "test"})
        
        assert checkpoint_id is not None
        assert len(wm.checkpoints) == 1
        
        checkpoint = wm.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert thread_id in checkpoint.active_threads
        assert item_id in checkpoint.focused_items
        assert checkpoint.metadata["reason"] == "test"
    
    def test_restore_checkpoint(self):
        """Test restoring from checkpoints."""
        wm = WorkingMemory()
        
        # Set up initial state
        thread_id = wm.create_thread("test")
        item_id = wm.add_context({"task": "test"}, AttentionLevel.FOCUSED)
        
        # Create checkpoint
        checkpoint_id = wm.create_checkpoint()
        
        # Modify state
        wm.complete_thread(thread_id)
        wm.unfocus(item_id)
        
        assert thread_id not in wm.active_thread_ids
        assert item_id not in wm.focused_items
        
        # Restore checkpoint
        success = wm.restore_checkpoint(checkpoint_id)
        assert success is True
        
        # Check state restored
        assert thread_id in wm.active_thread_ids
        assert item_id in wm.focused_items
        
        thread = wm.get_thread(thread_id)
        assert thread.status == ThreadStatus.ACTIVE
    
    def test_restore_nonexistent_checkpoint(self):
        """Test restoring from nonexistent checkpoint."""
        wm = WorkingMemory()
        
        success = wm.restore_checkpoint("nonexistent")
        assert success is False
    
    def test_delete_checkpoint(self):
        """Test deleting checkpoints."""
        wm = WorkingMemory()
        
        checkpoint_id = wm.create_checkpoint()
        assert len(wm.checkpoints) == 1
        
        wm.delete_checkpoint(checkpoint_id)
        assert len(wm.checkpoints) == 0
        assert wm.get_checkpoint(checkpoint_id) is None
    
    def test_checkpoint_preserves_attention_state(self):
        """Test that checkpoints preserve full attention state."""
        wm = WorkingMemory()
        
        # Create items with different attention levels
        id1 = wm.add_context({"task": "1"}, AttentionLevel.FOCUSED)
        id2 = wm.add_context({"task": "2"}, AttentionLevel.ACTIVE)
        id3 = wm.add_context({"task": "3"}, AttentionLevel.BACKGROUND)
        
        # Create checkpoint
        checkpoint_id = wm.create_checkpoint()
        
        # Modify attention
        wm.set_attention(id1, AttentionLevel.BACKGROUND)
        wm.set_attention(id2, AttentionLevel.FOCUSED)
        
        # Restore
        wm.restore_checkpoint(checkpoint_id)
        
        # Check attention restored
        item1 = wm.get_context(id1)
        item2 = wm.get_context(id2)
        item3 = wm.get_context(id3)
        
        assert item1.attention_level == AttentionLevel.FOCUSED
        assert item2.attention_level == AttentionLevel.ACTIVE
        assert item3.attention_level == AttentionLevel.BACKGROUND


class TestStatistics:
    """Test statistics and introspection."""
    
    def test_get_statistics(self):
        """Test getting working memory statistics."""
        wm = WorkingMemory(
            max_context_items=50,
            max_focused_items=5,
            max_active_threads=10,
        )
        
        # Add some items and threads
        wm.add_context({"task": "1"}, AttentionLevel.FOCUSED)
        wm.add_context({"task": "2"}, AttentionLevel.ACTIVE)
        wm.add_context({"task": "3"}, AttentionLevel.BACKGROUND)
        
        thread_id = wm.create_thread("test")
        wm.complete_thread(thread_id)
        
        wm.create_checkpoint()
        
        stats = wm.get_statistics()
        
        assert stats["context_items"]["total"] == 3
        assert stats["context_items"]["focused"] == 1
        assert stats["context_items"]["active"] == 1
        assert stats["context_items"]["background"] == 1
        
        assert stats["threads"]["total"] == 1
        assert stats["threads"]["active"] == 0
        assert stats["threads"]["completed"] == 1
        
        assert stats["checkpoints"] == 1
        
        assert stats["capacity"]["max_context_items"] == 50
        assert stats["capacity"]["max_focused_items"] == 5
        assert stats["capacity"]["max_active_threads"] == 10
    
    def test_clear(self):
        """Test clearing working memory."""
        wm = WorkingMemory()
        
        # Add some state
        wm.add_context({"task": "test"}, AttentionLevel.FOCUSED)
        wm.create_thread("test")
        wm.create_checkpoint()
        
        assert len(wm.context_items) > 0
        assert len(wm.threads) > 0
        assert len(wm.checkpoints) > 0
        
        # Clear
        wm.clear()
        
        assert len(wm.context_items) == 0
        assert len(wm.threads) == 0
        assert len(wm.checkpoints) == 0
        assert len(wm.focused_items) == 0
        assert len(wm.active_thread_ids) == 0


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_task_execution_with_interruption(self):
        """Test a realistic task execution with interruption and resume."""
        wm = WorkingMemory()
        
        # Start a task
        task_thread = wm.create_thread("main_task")
        
        # Add context for the task
        context1 = wm.add_context(
            {"step": "research", "data": "initial findings"},
            AttentionLevel.FOCUSED
        )
        wm.add_context_to_thread(task_thread, context1)
        
        # Add more context
        context2 = wm.add_context(
            {"step": "analysis", "data": "partial results"},
            AttentionLevel.ACTIVE
        )
        wm.add_context_to_thread(task_thread, context2)
        
        # Create checkpoint before interruption
        checkpoint = wm.create_checkpoint(metadata={"reason": "interruption"})
        
        # Simulate interruption - new urgent task
        urgent_thread = wm.create_thread("urgent_task")
        urgent_context = wm.add_context(
            {"task": "urgent", "priority": "high"},
            AttentionLevel.FOCUSED
        )
        wm.add_context_to_thread(urgent_thread, urgent_context)
        
        # Complete urgent task
        wm.complete_thread(urgent_thread)
        
        # Resume original task
        wm.restore_checkpoint(checkpoint)
        
        # Verify state restored
        assert task_thread in wm.active_thread_ids
        assert context1 in wm.focused_items
        
        thread = wm.get_thread(task_thread)
        assert thread.status == ThreadStatus.ACTIVE
        assert len(thread.context_items) == 2
    
    def test_multiple_parallel_threads(self):
        """Test managing multiple parallel threads of thought."""
        wm = WorkingMemory(max_active_threads=5)
        
        # Create multiple threads for different aspects of a problem
        research_thread = wm.create_thread("research")
        analysis_thread = wm.create_thread("analysis")
        planning_thread = wm.create_thread("planning")
        
        # Add context to each thread
        research_ctx = wm.add_context({"source": "paper1"}, AttentionLevel.ACTIVE)
        wm.add_context_to_thread(research_thread, research_ctx)
        
        analysis_ctx = wm.add_context({"data": "dataset1"}, AttentionLevel.ACTIVE)
        wm.add_context_to_thread(analysis_thread, analysis_ctx)
        
        planning_ctx = wm.add_context({"plan": "step1"}, AttentionLevel.FOCUSED)
        wm.add_context_to_thread(planning_thread, planning_ctx)
        
        # Verify all threads active
        active = wm.get_active_threads()
        assert len(active) == 3
        
        # Focus shifts to analysis
        wm.unfocus(planning_ctx)
        wm.focus_on(analysis_ctx)
        
        # Complete research thread
        wm.complete_thread(research_thread)
        
        # Verify state
        active = wm.get_active_threads()
        assert len(active) == 2
        assert research_thread not in wm.active_thread_ids
        
        focused = wm.get_focused_items()
        assert len(focused) == 1
        assert focused[0].item_id == analysis_ctx
    
    def test_hierarchical_thread_structure(self):
        """Test hierarchical thread structure with parent-child relationships."""
        wm = WorkingMemory()
        
        # Create parent thread
        parent = wm.create_thread("main_problem")
        
        # Create child threads for sub-problems
        child1 = wm.create_thread("sub_problem_1", parent_thread=parent)
        child2 = wm.create_thread("sub_problem_2", parent_thread=parent)
        
        # Create grandchild
        grandchild = wm.create_thread("detail_1", parent_thread=child1)
        
        # Verify structure
        parent_thread = wm.get_thread(parent)
        assert len(parent_thread.child_threads) == 2
        assert child1 in parent_thread.child_threads
        assert child2 in parent_thread.child_threads
        
        child1_thread = wm.get_thread(child1)
        assert child1_thread.parent_thread == parent
        assert len(child1_thread.child_threads) == 1
        assert grandchild in child1_thread.child_threads
        
        grandchild_thread = wm.get_thread(grandchild)
        assert grandchild_thread.parent_thread == child1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
