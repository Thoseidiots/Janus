"""
Integration tests for working memory with the unified memory system.

Tests how working memory integrates with episodic and semantic memory
for a complete memory architecture.
"""

import pytest
from datetime import datetime

from janus_reasoning_engine.memory import (
    UnifiedMemory,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    MemoryType,
    MemoryQuery,
    AttentionLevel,
    ThreadStatus,
    OutcomeType,
    SkillLevel,
)


class TestWorkingMemoryIntegration:
    """Test working memory integration with other memory systems."""
    
    @pytest.fixture
    def memory_system(self, tmp_path):
        """Create a complete memory system for testing."""
        # Initialize unified memory
        unified = UnifiedMemory(
            sqlite_path=str(tmp_path / "test.db"),
            artifacts_dir=str(tmp_path / "artifacts"),
            enable_hbm=False,  # Disable HBM for faster tests
        )
        unified.initialize()
        
        # Initialize episodic and semantic memory
        episodic = EpisodicMemory(unified)
        semantic = SemanticMemory(unified)
        
        # Initialize working memory
        working = WorkingMemory()
        
        yield {
            "unified": unified,
            "episodic": episodic,
            "semantic": semantic,
            "working": working,
        }
        
        # Cleanup
        unified.shutdown()
    
    def test_task_execution_with_memory_integration(self, memory_system):
        """
        Test a complete task execution flow using all memory systems.
        
        Scenario: Execute a task, store experience in episodic memory,
        update skills in semantic memory, and manage context in working memory.
        """
        unified = memory_system["unified"]
        episodic = memory_system["episodic"]
        semantic = memory_system["semantic"]
        working = memory_system["working"]
        
        # 1. Start a task in working memory
        task_thread = working.create_thread("data_analysis_task")
        
        # 2. Add context for the task
        task_context = working.add_context(
            {
                "task_type": "data_analysis",
                "dataset": "customer_data.csv",
                "goal": "identify trends",
            },
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(task_thread, task_context)
        
        # 3. Check if we have relevant skills in semantic memory
        skills = semantic.get_skills_by_domain("data_analysis")
        
        if not skills:
            # Add skill if not present
            skill_id = semantic.add_skill(
                name="Data Analysis",
                description="Analyzing datasets to identify trends",
                level=SkillLevel.INTERMEDIATE,
                domains=["data_analysis"],
            )
        else:
            skill_id = skills[0].skill_id
        
        # 4. Add progress context
        progress_context = working.add_context(
            {
                "step": "data_loading",
                "status": "in_progress",
                "progress": 0.3,
            },
            AttentionLevel.ACTIVE
        )
        working.add_context_to_thread(task_thread, progress_context)
        
        # 5. Simulate task completion
        working.update_context(
            progress_context,
            content={
                "step": "analysis_complete",
                "status": "completed",
                "progress": 1.0,
                "findings": "identified 3 key trends",
            }
        )
        
        # 6. Complete the thread
        working.complete_thread(task_thread, success=True)
        
        # 7. Store experience in episodic memory
        experience_id = episodic.store_experience(
            context={
                "task_type": "data_analysis",
                "dataset": "customer_data.csv",
            },
            action={
                "action_type": "analyze_data",
                "tools_used": ["pandas", "matplotlib"],
            },
            outcome={
                "result": "identified 3 key trends",
                "value": 100,
            },
            outcome_type=OutcomeType.SUCCESS,
            skills=[skill_id],
        )
        
        # 8. Update skill confidence based on success
        current_skill = semantic.get_skill(skill_id)
        new_confidence = min(1.0, current_skill.confidence + 0.1)
        semantic.update_skill(skill_id, confidence=new_confidence)
        
        # Verify the integration
        
        # Check working memory state
        thread = working.get_thread(task_thread)
        assert thread.status == ThreadStatus.COMPLETED
        assert len(thread.context_items) == 2
        
        # Check episodic memory - just verify it was stored
        stats = episodic.get_statistics()
        assert stats["total_experiences"] > 0
        assert stats["success_count"] > 0
        
        # Check semantic memory
        updated_skill = semantic.get_skill(skill_id)
        assert updated_skill.confidence > 0.5  # Confidence increased
    
    def test_interruption_and_context_preservation(self, memory_system):
        """
        Test interruption handling with context preservation across memory systems.
        
        Scenario: Start a task, get interrupted, save state, resume later.
        """
        unified = memory_system["unified"]
        episodic = memory_system["episodic"]
        semantic = memory_system["semantic"]
        working = memory_system["working"]
        
        # 1. Start main task
        main_thread = working.create_thread("main_task")
        main_context = working.add_context(
            {
                "task": "write_report",
                "section": "introduction",
                "progress": 0.4,
            },
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(main_thread, main_context)
        
        # 2. Add relevant knowledge from semantic memory
        from janus_reasoning_engine.memory import KnowledgeType
        
        knowledge_id = semantic.add_knowledge(
            knowledge_type=KnowledgeType.CONCEPT,
            name="report_writing",
            content={"best_practices": ["clear structure", "concise language"]},
        )
        
        knowledge_context = working.add_context(
            {
                "type": "knowledge_reference",
                "knowledge_id": knowledge_id,
                "name": "report_writing",
            },
            AttentionLevel.ACTIVE
        )
        working.add_context_to_thread(main_thread, knowledge_context)
        
        # 3. Create checkpoint before interruption
        checkpoint_id = working.create_checkpoint(
            metadata={"reason": "urgent_interruption"}
        )
        
        # 4. Handle interruption - urgent task
        urgent_thread = working.create_thread("urgent_task")
        urgent_context = working.add_context(
            {"task": "urgent_email", "priority": "high"},
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(urgent_thread, urgent_context)
        
        # 5. Complete urgent task
        working.complete_thread(urgent_thread, success=True)
        
        # Store urgent task experience
        episodic.store_experience(
            context={"task": "urgent_email"},
            action={"action_type": "send_email"},
            outcome={"result": "email sent"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        # 6. Resume main task
        working.restore_checkpoint(checkpoint_id)
        
        # Verify state restored
        assert main_thread in working.active_thread_ids
        assert main_context in working.focused_items
        
        thread = working.get_thread(main_thread)
        assert thread.status == ThreadStatus.ACTIVE
        assert len(thread.context_items) == 2
        
        # Verify we can still access semantic knowledge
        knowledge = semantic.get_knowledge(knowledge_id)
        assert knowledge is not None
        assert knowledge.name == "report_writing"
    
    def test_learning_from_experience_flow(self, memory_system):
        """
        Test learning flow: experience in working memory → episodic storage → skill update.
        
        Scenario: Try a new skill, succeed/fail, update skill inventory.
        """
        unified = memory_system["unified"]
        episodic = memory_system["episodic"]
        semantic = memory_system["semantic"]
        working = memory_system["working"]
        
        # 1. Identify skill gap (new skill needed)
        new_skill_id = semantic.add_skill(
            name="Machine Learning",
            description="Basic ML model training",
            level=SkillLevel.BEGINNER,
            confidence=0.3,
            domains=["ai"],
        )
        
        # 2. Start learning task in working memory
        learning_thread = working.create_thread("learn_ml")
        
        learning_context = working.add_context(
            {
                "activity": "learning",
                "skill": "machine_learning",
                "resource": "online_tutorial",
            },
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(learning_thread, learning_context)
        
        # 3. Practice task
        practice_thread = working.create_thread("practice_ml", parent_thread=learning_thread)
        
        practice_context = working.add_context(
            {
                "activity": "practice",
                "task": "train_simple_model",
                "dataset": "iris",
            },
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(practice_thread, practice_context)
        
        # 4. Complete practice (success)
        working.complete_thread(practice_thread, success=True)
        
        # 5. Store practice experience
        practice_exp_id = episodic.store_experience(
            context={"task": "train_simple_model", "dataset": "iris"},
            action={"action_type": "train_ml_model", "algorithm": "decision_tree"},
            outcome={
                "result": "model trained successfully",
                "accuracy": 0.95,
            },
            outcome_type=OutcomeType.SUCCESS,
            skills=[new_skill_id],
        )
        
        # 6. Update skill based on success
        semantic.update_skill(new_skill_id, level=SkillLevel.INTERMEDIATE, confidence=0.5)
        
        # 7. Complete learning thread
        working.complete_thread(learning_thread, success=True)
        
        # Verify learning progression
        
        # Check skill improved
        updated_skill = semantic.get_skill(new_skill_id)
        assert updated_skill.level == SkillLevel.INTERMEDIATE
        assert updated_skill.confidence == 0.5
        
        # Check experience stored
        stats = episodic.get_statistics()
        assert stats["total_experiences"] > 0
        assert stats["success_count"] > 0
        
        # Check thread hierarchy
        learning_thread_obj = working.get_thread(learning_thread)
        assert practice_thread in learning_thread_obj.child_threads
    
    def test_multi_thread_context_switching(self, memory_system):
        """
        Test managing multiple threads with context switching.
        
        Scenario: Work on multiple tasks, switch focus, maintain context.
        """
        unified = memory_system["unified"]
        working = memory_system["working"]
        semantic = memory_system["semantic"]
        
        # Create multiple threads for different tasks
        thread1 = working.create_thread("code_review")
        thread2 = working.create_thread("bug_fix")
        thread3 = working.create_thread("documentation")
        
        # Add context to each thread
        ctx1 = working.add_context(
            {"task": "review_pr", "file": "main.py"},
            AttentionLevel.FOCUSED
        )
        working.add_context_to_thread(thread1, ctx1)
        
        ctx2 = working.add_context(
            {"task": "fix_bug", "issue": "memory_leak"},
            AttentionLevel.ACTIVE
        )
        working.add_context_to_thread(thread2, ctx2)
        
        ctx3 = working.add_context(
            {"task": "write_docs", "section": "api"},
            AttentionLevel.BACKGROUND
        )
        working.add_context_to_thread(thread3, ctx3)
        
        # Verify initial state
        assert len(working.get_active_threads()) == 3
        focused = working.get_focused_items()
        assert len(focused) == 1
        assert focused[0].item_id == ctx1
        
        # Switch focus to bug fix
        working.unfocus(ctx1)
        working.focus_on(ctx2)
        
        # Verify focus switched
        focused = working.get_focused_items()
        assert len(focused) == 1
        assert focused[0].item_id == ctx2
        
        # Complete code review
        working.complete_thread(thread1, success=True)
        
        # Verify state
        assert len(working.get_active_threads()) == 2
        assert thread1 not in working.active_thread_ids
        
        # Get context for active threads
        thread2_context = working.get_thread_context(thread2)
        assert len(thread2_context) == 1
        assert thread2_context[0].content["task"] == "fix_bug"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
