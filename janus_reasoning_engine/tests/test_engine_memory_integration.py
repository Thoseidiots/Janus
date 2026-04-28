"""
Integration tests for reasoning engine with memory layer.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from janus_reasoning_engine.core.engine import JanusReasoningEngine
from janus_reasoning_engine.core.config import EngineConfig
from janus_reasoning_engine.memory import UnifiedMemory, MemoryType


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
    config.memory.checkpoint_dir = str(Path(temp_dir) / "checkpoints")
    
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


class TestEngineMemoryIntegration:
    """Tests for reasoning engine with memory layer."""
    
    def test_engine_with_memory_initialization(self, engine_with_memory):
        """Test that engine and memory can be initialized together."""
        engine, memory = engine_with_memory
        
        assert engine.initialized
        assert memory.initialized
    
    def test_store_decision_in_memory(self, engine_with_memory):
        """Test storing engine decisions in memory."""
        engine, memory = engine_with_memory
        
        # Make a decision
        decision = engine.decide_next_action()
        
        # Store decision in episodic memory
        memory_id = memory.store(
            MemoryType.EPISODIC,
            {
                "decision_type": decision.decision_type,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
            },
            metadata={
                "source": "reasoning_engine",
                "action_count": engine.action_count,
            }
        )
        
        assert memory_id is not None
        
        # Retrieve decision
        results = memory.structured_query(
            MemoryType.EPISODIC,
            filters={"source": "reasoning_engine"},
            limit=10
        )
        
        assert len(results) > 0
        assert results[0].content["decision_type"] == decision.decision_type
    
    def test_store_reflection_in_memory(self, engine_with_memory):
        """Test storing engine reflections in memory."""
        engine, memory = engine_with_memory
        
        # Perform reflection
        insights = engine.reflect_on_recent_actions()
        
        # Store reflection in semantic memory
        memory_id = memory.store(
            MemoryType.SEMANTIC,
            {
                "reflection_type": "periodic",
                "actions_taken": insights["actions_taken"],
                "uptime": insights["uptime_seconds"],
            },
            metadata={
                "source": "reflection",
            }
        )
        
        assert memory_id is not None
        
        # Retrieve reflection
        results = memory.structured_query(
            MemoryType.SEMANTIC,
            filters={"source": "reflection"},
            limit=10
        )
        
        assert len(results) > 0
    
    def test_memory_statistics_with_engine(self, engine_with_memory):
        """Test getting memory statistics while engine is running."""
        engine, memory = engine_with_memory
        
        # Store some data
        memory.store(MemoryType.EPISODIC, {"test": "data1"})
        memory.store(MemoryType.SEMANTIC, {"test": "data2"})
        
        # Get statistics
        stats = memory.get_statistics()
        
        assert stats["initialized"]
        assert "backends" in stats
        
        # Get engine status
        engine_status = engine.get_status()
        
        assert engine_status["initialized"]
        assert engine_status["subsystems"]["goal_manager"] is False  # Not set yet
    
    def test_multiple_decisions_with_memory(self, engine_with_memory):
        """Test storing multiple decisions in memory."""
        engine, memory = engine_with_memory
        
        # Make multiple decisions
        for i in range(5):
            decision = engine.decide_next_action()
            
            memory.store(
                MemoryType.EPISODIC,
                {
                    "decision_type": decision.decision_type,
                    "iteration": i,
                },
                metadata={"source": "test"}
            )
        
        # Retrieve all decisions
        results = memory.structured_query(
            MemoryType.EPISODIC,
            filters={"source": "test"},
            limit=10
        )
        
        assert len(results) == 5
    
    def test_checkpoint_storage(self, engine_with_memory):
        """Test storing engine checkpoints."""
        engine, memory = engine_with_memory
        
        # Get engine status
        status = engine.get_status()
        
        # Store as checkpoint
        import json
        checkpoint_data = json.dumps(status, indent=2)
        
        memory_id = memory.store(
            MemoryType.ARTIFACT,
            {
                "filename": "engine_checkpoint.json",
                "data": checkpoint_data,
            },
            metadata={
                "type": "checkpoint",
                "action_count": engine.action_count,
            }
        )
        
        assert memory_id is not None
        
        # Retrieve checkpoint
        from janus_reasoning_engine.memory import MemoryQuery
        
        results = memory.retrieve(
            MemoryQuery(
                query_type=MemoryType.ARTIFACT,
                filters={"type": "checkpoint"},
                limit=10
            )
        )
        
        assert len(results) > 0
        assert "storage_path" in results[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
