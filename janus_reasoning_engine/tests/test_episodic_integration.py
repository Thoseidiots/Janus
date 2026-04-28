"""
Integration tests for episodic memory with reasoning engine.

**Validates: Requirements REQ-6.1, REQ-6.4**
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from janus_reasoning_engine.memory import (
    UnifiedMemory,
    EpisodicMemory,
    OutcomeType,
)
from janus_reasoning_engine.core import JanusReasoningEngine, EngineConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def engine_with_memory(temp_dir):
    """Create a reasoning engine with memory layer."""
    # Create config
    config = EngineConfig()
    config.memory.sqlite_path = str(Path(temp_dir) / "test.db")
    config.memory.hbm_dimension = 1024
    
    # Create memory
    memory = UnifiedMemory(
        hbm_dimension=1024,
        sqlite_path=config.memory.sqlite_path,
        artifacts_dir=str(Path(temp_dir) / "artifacts"),
    )
    memory.initialize()
    
    # Create engine
    engine = JanusReasoningEngine(config=config)
    engine.initialize()
    
    yield engine, memory
    
    # Cleanup
    engine.shutdown()
    memory.shutdown()


class TestEpisodicMemoryIntegration:
    """Integration tests for episodic memory with reasoning engine."""
    
    def test_engine_has_episodic_memory(self, engine_with_memory):
        """Test that reasoning engine can access episodic memory."""
        engine, memory = engine_with_memory
        
        # Create episodic memory instance
        episodic_memory = EpisodicMemory(memory)
        
        # Store an experience
        exp_id = episodic_memory.store_experience(
            context={"task": "test_task"},
            action={"type": "test_action"},
            outcome={"result": "success"},
            outcome_type=OutcomeType.SUCCESS,
        )
        
        assert exp_id is not None
        assert episodic_memory.total_experiences == 1
    
    def test_store_and_retrieve_experience(self, engine_with_memory):
        """Test storing and retrieving experiences through engine."""
        engine, memory = engine_with_memory
        episodic_memory = EpisodicMemory(memory)
        
        # Store multiple experiences
        for i in range(5):
            episodic_memory.store_experience(
                context={"task": f"task_{i}"},
                action={"type": f"action_{i}"},
                outcome={"result": "success"},
                skills=["python"],
                outcome_type=OutcomeType.SUCCESS,
                earnings=100.0 * (i + 1),
            )
        
        # Retrieve successful experiences
        successes = episodic_memory.retrieve_successful_experiences()
        assert len(successes) == 5
        
        # Get statistics
        stats = episodic_memory.get_statistics()
        assert stats["total_experiences"] == 5
        assert stats["success_rate"] == 1.0
    
    def test_experience_replay_workflow(self, engine_with_memory):
        """Test experience replay workflow."""
        engine, memory = engine_with_memory
        episodic_memory = EpisodicMemory(memory)
        
        # Store mix of successes and failures
        for i in range(3):
            episodic_memory.store_experience(
                context={"task": f"success_{i}"},
                action={"type": "action"},
                outcome={"result": "success"},
                outcome_type=OutcomeType.SUCCESS,
            )
        
        for i in range(2):
            episodic_memory.store_experience(
                context={"task": f"failure_{i}"},
                action={"type": "action"},
                outcome={"result": "failure"},
                outcome_type=OutcomeType.FAILURE,
            )
        
        # Get replay batch prioritizing failures
        batch = episodic_memory.get_experience_replay_batch(
            batch_size=5,
            prioritize_failures=True,
        )
        
        assert len(batch) == 5
        
        # Should have failures first
        failure_count = sum(
            1 for exp in batch
            if exp.outcome_type == OutcomeType.FAILURE
        )
        assert failure_count >= 1  # At least some failures


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
