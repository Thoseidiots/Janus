"""
Integration tests for Semantic and Episodic Memory interaction.

Tests how skills, procedures, and knowledge from semantic memory
integrate with experiences from episodic memory.
"""

import pytest
import tempfile
import shutil

from janus_reasoning_engine.memory import (
    UnifiedMemory,
    SemanticMemory,
    EpisodicMemory,
    SkillLevel,
    OutcomeType,
    KnowledgeType,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def unified_memory(temp_dir):
    """Create unified memory instance for testing."""
    memory = UnifiedMemory(
        hbm_dimension=1000,
        hbm_sparsity=0.1,
        sqlite_path=f"{temp_dir}/test_integration.db",
        artifacts_dir=f"{temp_dir}/artifacts",
        enable_hbm=True,
        enable_sqlite=True,
        enable_filesystem=True,
    )
    memory.initialize()
    yield memory
    memory.shutdown()


@pytest.fixture
def semantic_memory(unified_memory):
    """Create semantic memory instance."""
    return SemanticMemory(unified_memory)


@pytest.fixture
def episodic_memory(unified_memory):
    """Create episodic memory instance."""
    return EpisodicMemory(unified_memory)


class TestSemanticEpisodicIntegration:
    """Test integration between semantic and episodic memory."""
    
    def test_skill_learning_from_experience(self, semantic_memory, episodic_memory):
        """Test updating skills based on experiences."""
        # Add a skill
        python_id = semantic_memory.add_skill(
            name="Python",
            description="Python programming",
            level=SkillLevel.BEGINNER,
            confidence=0.4,
        )
        
        # Store successful experience using this skill
        episodic_memory.store_experience(
            context={"task": "Build web scraper", "platform": "Upwork"},
            action={"type": "code", "language": "python"},
            outcome={"success": True, "quality": "high"},
            skills=[python_id],
            outcome_type=OutcomeType.SUCCESS,
            earnings=150.0,
        )
        
        # Update skill based on successful experience
        semantic_memory.update_skill(
            python_id,
            level=SkillLevel.INTERMEDIATE,
            confidence=0.6,
            increment_use=True,
            increment_success=True,
        )
        
        # Verify skill was updated
        skill = semantic_memory.get_skill(python_id)
        assert skill.level == SkillLevel.INTERMEDIATE
        assert skill.confidence == 0.6
        assert skill.use_count == 1
        assert skill.success_count == 1
        
        # Verify experience was stored
        all_experiences = episodic_memory.retrieve_by_outcome(OutcomeType.SUCCESS)
        assert len(all_experiences) >= 1
    
    def test_procedure_execution_tracking(self, semantic_memory, episodic_memory):
        """Test tracking procedure execution through experiences."""
        # Add a procedure
        deploy_proc_id = semantic_memory.add_procedure(
            name="Deploy to Heroku",
            description="Deploy application to Heroku",
            steps=[
                {"action": "prepare", "description": "Prepare app"},
                {"action": "deploy", "description": "Deploy to Heroku"},
            ],
        )
        
        # Execute procedure and store experience
        episodic_memory.store_experience(
            context={"procedure": deploy_proc_id, "platform": "Heroku"},
            action={"type": "deployment", "procedure_id": deploy_proc_id},
            outcome={"success": True, "deployment_time": 120},
            outcome_type=OutcomeType.SUCCESS,
            time_spent=0.5,
        )
        
        # Update procedure usage
        semantic_memory.update_procedure_usage(deploy_proc_id, success=True)
        
        # Verify procedure statistics
        procedure = semantic_memory.get_procedure(deploy_proc_id)
        assert procedure.use_count == 1
        assert procedure.success_count == 1
    
    def test_skill_gap_identification(self, semantic_memory, episodic_memory):
        """Test identifying skill gaps from failed experiences."""
        # Add existing skill
        python_id = semantic_memory.add_skill(
            name="Python",
            description="Python programming",
            level=SkillLevel.INTERMEDIATE,
        )
        
        # Store failed experience requiring unknown skill
        episodic_memory.store_experience(
            context={"task": "Build ML model", "required_skills": ["python", "machine-learning"]},
            action={"type": "code", "attempted": True},
            outcome={"success": False, "reason": "Lack of ML knowledge"},
            skills=[python_id],
            outcome_type=OutcomeType.FAILURE,
        )
        
        # Identify that ML skill is missing
        # (In real system, this would trigger learning)
        ml_id = semantic_memory.add_skill(
            name="Machine Learning",
            description="Building ML models",
            level=SkillLevel.NOVICE,
            confidence=0.2,
        )
        
        # After learning, store successful experience
        episodic_memory.store_experience(
            context={"task": "Build ML model", "after_learning": True},
            action={"type": "code", "used_skills": ["python", "ml"]},
            outcome={"success": True, "model_accuracy": 0.85},
            skills=[python_id, ml_id],
            outcome_type=OutcomeType.SUCCESS,
            earnings=300.0,
            learning_value=0.9,
        )
        
        # Update ML skill after successful use
        semantic_memory.update_skill(
            ml_id,
            level=SkillLevel.BEGINNER,
            confidence=0.5,
            increment_use=True,
            increment_success=True,
        )
        
        # Verify learning progression
        ml_skill = semantic_memory.get_skill(ml_id)
        assert ml_skill.level == SkillLevel.BEGINNER
        assert ml_skill.confidence == 0.5
        
        # Verify experiences were stored
        all_experiences = episodic_memory.retrieve_by_outcome(OutcomeType.SUCCESS)
        assert len(all_experiences) >= 1
    
    def test_knowledge_application_in_experience(self, semantic_memory, episodic_memory):
        """Test applying knowledge from semantic memory in experiences."""
        # Add knowledge about best practices
        knowledge_id = semantic_memory.add_knowledge(
            knowledge_type=KnowledgeType.RULE,
            name="Use version control",
            content={
                "rule": "Always use Git for version control",
                "benefits": ["Track changes", "Collaboration", "Backup"],
            },
        )
        
        # Store experience applying this knowledge
        episodic_memory.store_experience(
            context={"task": "Software project", "applied_knowledge": [knowledge_id]},
            action={"type": "development", "used_git": True},
            outcome={"success": True, "collaboration_smooth": True},
            outcome_type=OutcomeType.SUCCESS,
            tags=["best-practices", "version-control"],
        )
        
        # Retrieve experiences with this tag
        experiences = episodic_memory.retrieve_by_outcome(OutcomeType.SUCCESS)
        best_practice_experiences = [
            exp for exp in experiences
            if "best-practices" in exp.tags
        ]
        assert len(best_practice_experiences) >= 1
    
    def test_complete_learning_cycle(self, semantic_memory, episodic_memory):
        """Test complete learning cycle: skill → experience → improvement."""
        # 1. Start with basic skill
        js_id = semantic_memory.add_skill(
            name="JavaScript",
            description="JavaScript programming",
            level=SkillLevel.BEGINNER,
            confidence=0.3,
        )
        
        # 2. First attempt - partial success
        episodic_memory.store_experience(
            context={"task": "Build React app", "first_attempt": True},
            action={"type": "code", "framework": "react"},
            outcome={"success": False, "completed": 0.6},
            skills=[js_id],
            outcome_type=OutcomeType.PARTIAL,
            time_spent=8.0,
        )
        
        # 3. Learn from failure, add React knowledge
        react_knowledge_id = semantic_memory.add_knowledge(
            knowledge_type=KnowledgeType.CONCEPT,
            name="React Hooks",
            content={
                "concept": "React Hooks for state management",
                "examples": ["useState", "useEffect", "useContext"],
            },
        )
        
        # 4. Second attempt - success
        episodic_memory.store_experience(
            context={"task": "Build React app", "second_attempt": True, "learned": True},
            action={"type": "code", "framework": "react", "used_hooks": True},
            outcome={"success": True, "quality": "good"},
            skills=[js_id],
            outcome_type=OutcomeType.SUCCESS,
            earnings=200.0,
            time_spent=6.0,
            learning_value=0.8,
        )
        
        # 5. Update skill based on success
        semantic_memory.update_skill(
            js_id,
            level=SkillLevel.INTERMEDIATE,
            confidence=0.6,
            increment_use=True,
            increment_success=True,
        )
        
        # 6. Verify improvement
        skill = semantic_memory.get_skill(js_id)
        assert skill.level == SkillLevel.INTERMEDIATE
        assert skill.confidence == 0.6
        
        # 7. Verify experiences were stored
        all_experiences = episodic_memory.retrieve_by_outcome(OutcomeType.SUCCESS)
        assert len(all_experiences) >= 1
        
        # 8. Get successful experiences for replay
        successful = episodic_memory.retrieve_successful_experiences()
        assert len(successful) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
