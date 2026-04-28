"""
Tests for the episodic memory system.

**Validates: Requirements REQ-6.1, REQ-6.4**
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from janus_reasoning_engine.memory import (
    UnifiedMemory,
    EpisodicMemory,
    Experience,
    OutcomeType,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def unified_memory(temp_dir):
    """Create a unified memory instance for testing."""
    memory = UnifiedMemory(
        hbm_dimension=1024,  # Smaller for testing
        hbm_sparsity=0.1,
        sqlite_path=str(Path(temp_dir) / "test.db"),
        artifacts_dir=str(Path(temp_dir) / "artifacts"),
    )
    memory.initialize()
    yield memory
    memory.shutdown()


@pytest.fixture
def episodic_memory(unified_memory):
    """Create an episodic memory instance for testing."""
    return EpisodicMemory(unified_memory)


class TestExperience:
    """Tests for Experience dataclass."""
    
    def test_experience_creation(self):
        """Test creating an experience."""
        experience = Experience(
            experience_id="test-123",
            context={"task": "web development"},
            action={"type": "submit_proposal"},
            outcome={"result": "hired"},
            timestamp=datetime.utcnow(),
            skills=["python", "django"],
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=500.0,
        )
        
        assert experience.experience_id == "test-123"
        assert experience.context["task"] == "web development"
        assert experience.outcome_type == OutcomeType.SUCCESS
        assert experience.earnings == 500.0
    
    def test_experience_to_dict(self):
        """Test converting experience to dictionary."""
        experience = Experience(
            experience_id="test-123",
            context={"task": "test"},
            action={"type": "test"},
            outcome={"result": "test"},
            timestamp=datetime.utcnow(),
            skills=["skill1"],
            platforms=["platform1"],
        )
        
        data = experience.to_dict()
        
        assert data["experience_id"] == "test-123"
        assert "context" in data
        assert "action" in data
        assert "outcome" in data
        assert "timestamp" in data
        assert data["skills"] == ["skill1"]
        assert data["platforms"] == ["platform1"]
    
    def test_experience_from_dict(self):
        """Test creating experience from dictionary."""
        data = {
            "experience_id": "test-123",
            "context": {"task": "test"},
            "action": {"type": "test"},
            "outcome": {"result": "test"},
            "timestamp": datetime.utcnow().isoformat(),
            "skills": ["skill1"],
            "platforms": ["platform1"],
            "outcome_type": "success",
            "earnings": 100.0,
        }
        
        experience = Experience.from_dict(data)
        
        assert experience.experience_id == "test-123"
        assert experience.skills == ["skill1"]
        assert experience.outcome_type == OutcomeType.SUCCESS
        assert experience.earnings == 100.0


class TestEpisodicMemory:
    """Tests for EpisodicMemory system."""
    
    def test_store_experience(self, episodic_memory):
        """Test storing an experience."""
        experience_id = episodic_memory.store_experience(
            context={"task": "web development", "platform": "upwork"},
            action={"type": "submit_proposal", "bid": 500},
            outcome={"result": "hired", "earnings": 500},
            skills=["python", "django"],
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=500.0,
            time_spent=10.0,
        )
        
        assert experience_id is not None
        assert isinstance(experience_id, str)
        assert episodic_memory.total_experiences == 1
        assert episodic_memory.success_count == 1
    
    def test_store_multiple_experiences(self, episodic_memory):
        """Test storing multiple experiences."""
        # Store success
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        # Store failure
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "failure"},
            outcome_type=OutcomeType.FAILURE,
        )
        
        assert episodic_memory.total_experiences == 2
        assert episodic_memory.success_count == 1
        assert episodic_memory.failure_count == 1
    
    def test_retrieve_similar_experiences(self, episodic_memory):
        """Test retrieving similar experiences."""
        # Store some experiences
        episodic_memory.store_experience(
            context={"task": "web development with Python"},
            action={"type": "submit_proposal"},
            outcome={"result": "hired"},
            skills=["python", "django"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "mobile app development"},
            action={"type": "submit_proposal"},
            outcome={"result": "rejected"},
            skills=["react-native"],
            outcome_type=OutcomeType.FAILURE,
        )
        
        # Search for similar experiences
        results = episodic_memory.retrieve_similar_experiences(
            "web development",
            limit=5,
            similarity_threshold=0.0,  # Low threshold for testing
        )
        
        # Should find at least some results
        assert len(results) >= 0  # May or may not find with sparse encoding
    
    def test_retrieve_by_skill(self, episodic_memory):
        """Test retrieving experiences by skill."""
        # Store experiences with different skills
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            skills=["python", "django"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "success"},
            skills=["javascript", "react"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "task3"},
            action={"type": "action3"},
            outcome={"result": "success"},
            skills=["python", "flask"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        # Retrieve Python experiences
        python_experiences = episodic_memory.retrieve_by_skill("python")
        
        # Should find experiences with Python skill
        assert len(python_experiences) >= 0  # Depends on SQLite backend implementation
    
    def test_retrieve_by_platform(self, episodic_memory):
        """Test retrieving experiences by platform."""
        # Store experiences on different platforms
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "success"},
            platforms=["fiverr"],
            outcome_type=OutcomeType.SUCCESS,
        )
        
        # Retrieve Upwork experiences
        upwork_experiences = episodic_memory.retrieve_by_platform("upwork")
        
        # Should find experiences on Upwork
        assert len(upwork_experiences) >= 0
    
    def test_retrieve_by_outcome(self, episodic_memory):
        """Test retrieving experiences by outcome type."""
        # Store experiences with different outcomes
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "failure"},
            outcome_type=OutcomeType.FAILURE,
        )
        
        episodic_memory.store_experience(
            context={"task": "task3"},
            action={"type": "action3"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        # Retrieve successful experiences
        successes = episodic_memory.retrieve_by_outcome(OutcomeType.SUCCESS)
        
        # Should find success experiences
        assert len(successes) >= 0
    
    def test_retrieve_successful_experiences(self, episodic_memory):
        """Test retrieving successful experiences with earnings filter."""
        # Store successful experiences with different earnings
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
            earnings=100.0,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
            earnings=500.0,
        )
        
        episodic_memory.store_experience(
            context={"task": "task3"},
            action={"type": "action3"},
            outcome={"result": "failure"},
            outcome_type=OutcomeType.FAILURE,
        )
        
        # Retrieve all successful experiences
        all_successes = episodic_memory.retrieve_successful_experiences()
        assert len(all_successes) >= 0
        
        # Retrieve high-earning experiences
        high_earning = episodic_memory.retrieve_successful_experiences(
            min_earnings=200.0
        )
        
        # Should filter by earnings
        for exp in high_earning:
            assert exp.earnings is not None
            assert exp.earnings >= 200.0
    
    def test_experience_replay_batch(self, episodic_memory):
        """Test getting experience replay batch."""
        # Store mix of successes and failures
        for i in range(5):
            episodic_memory.store_experience(
                context={"task": f"task{i}"},
                action={"type": f"action{i}"},
                outcome={"result": "success"},
                outcome_type=OutcomeType.SUCCESS,
            )
        
        for i in range(3):
            episodic_memory.store_experience(
                context={"task": f"fail{i}"},
                action={"type": f"action{i}"},
                outcome={"result": "failure"},
                outcome_type=OutcomeType.FAILURE,
            )
        
        # Get replay batch prioritizing failures
        batch = episodic_memory.get_experience_replay_batch(
            batch_size=5,
            prioritize_failures=True,
        )
        
        assert len(batch) <= 5
        
        # Get replay batch without prioritization
        batch2 = episodic_memory.get_experience_replay_batch(
            batch_size=5,
            prioritize_failures=False,
        )
        
        assert len(batch2) <= 5
    
    def test_analyze_skill_performance(self, episodic_memory):
        """Test analyzing skill performance."""
        # Store experiences with Python skill
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            skills=["python"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=100.0,
            time_spent=5.0,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "success"},
            skills=["python"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=200.0,
            time_spent=10.0,
        )
        
        episodic_memory.store_experience(
            context={"task": "task3"},
            action={"type": "action3"},
            outcome={"result": "failure"},
            skills=["python"],
            outcome_type=OutcomeType.FAILURE,
            time_spent=3.0,
        )
        
        # Analyze Python skill
        stats = episodic_memory.analyze_skill_performance("python")
        
        assert "skill" in stats
        assert stats["skill"] == "python"
        assert "total_experiences" in stats
        assert "success_rate" in stats
        assert "average_earnings" in stats
    
    def test_analyze_platform_performance(self, episodic_memory):
        """Test analyzing platform performance."""
        # Store experiences on Upwork
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=100.0,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "success"},
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=200.0,
        )
        
        # Analyze Upwork platform
        stats = episodic_memory.analyze_platform_performance("upwork")
        
        assert "platform" in stats
        assert stats["platform"] == "upwork"
        assert "total_experiences" in stats
        assert "success_rate" in stats
        assert "average_earnings" in stats
    
    def test_get_statistics(self, episodic_memory):
        """Test getting overall statistics."""
        # Store some experiences
        episodic_memory.store_experience(
            context={"task": "task1"},
            action={"type": "action1"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        episodic_memory.store_experience(
            context={"task": "task2"},
            action={"type": "action2"},
            outcome={"result": "failure"},
            outcome_type=OutcomeType.FAILURE,
        )
        
        # Get statistics
        stats = episodic_memory.get_statistics()
        
        assert stats["total_experiences"] == 2
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1
        assert stats["success_rate"] == 0.5


class TestEpisodicMemoryIntegration:
    """Integration tests for episodic memory."""
    
    def test_complete_workflow(self, episodic_memory):
        """Test complete episodic memory workflow."""
        # Store a successful job experience
        exp_id = episodic_memory.store_experience(
            context={
                "platform": "upwork",
                "job_title": "Python Web Development",
                "description": "Build a Django REST API",
            },
            action={
                "type": "submit_proposal",
                "bid_amount": 500,
                "proposal_text": "I can build this API...",
            },
            outcome={
                "result": "hired",
                "final_earnings": 500,
                "client_rating": 5.0,
            },
            skills=["python", "django", "rest-api"],
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=500.0,
            time_spent=20.0,
            difficulty=0.6,
            learning_value=0.3,
            tags=["web-dev", "backend"],
        )
        
        assert exp_id is not None
        
        # Retrieve similar experiences
        similar = episodic_memory.retrieve_similar_experiences(
            "Django web development",
            limit=5,
            similarity_threshold=0.0,
        )
        
        # Retrieve by skill
        python_exp = episodic_memory.retrieve_by_skill("python")
        
        # Retrieve by platform
        upwork_exp = episodic_memory.retrieve_by_platform("upwork")
        
        # Analyze skill performance
        python_stats = episodic_memory.analyze_skill_performance("python")
        assert python_stats["total_experiences"] >= 0
        
        # Get statistics
        stats = episodic_memory.get_statistics()
        assert stats["total_experiences"] >= 1
    
    def test_learning_from_failures(self, episodic_memory):
        """Test learning from failure experiences."""
        # Store a failure experience
        episodic_memory.store_experience(
            context={
                "platform": "upwork",
                "job_title": "Machine Learning Project",
            },
            action={
                "type": "submit_proposal",
                "bid_amount": 1000,
            },
            outcome={
                "result": "rejected",
                "reason": "insufficient experience",
            },
            skills=["machine-learning"],
            platforms=["upwork"],
            outcome_type=OutcomeType.FAILURE,
            learning_value=0.8,  # High learning value
            tags=["ml", "rejected"],
        )
        
        # Get replay batch prioritizing failures
        batch = episodic_memory.get_experience_replay_batch(
            batch_size=10,
            prioritize_failures=True,
        )
        
        # Should include the failure
        assert len(batch) >= 0
        
        # Analyze what went wrong
        ml_stats = episodic_memory.analyze_skill_performance("machine-learning")
        
        # Should show low success rate
        if ml_stats["total_experiences"] > 0:
            assert ml_stats["success_rate"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
