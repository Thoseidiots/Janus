"""
Tests for core reasoning engine architecture.

Validates that the core interfaces, configuration, and engine work correctly.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import json

from janus_reasoning_engine.core.interfaces import (
    ReasoningEngine,
    GoalManager,
    StrategyPlanner,
    ExecutionMonitor,
    Goal,
    Strategy,
    GoalStatus,
    StrategyStatus,
)
from janus_reasoning_engine.core.config import (
    EngineConfig,
    MemoryConfig,
    ReasoningConfig,
    SafetyConfig,
    get_default_config,
)
from janus_reasoning_engine.core.engine import JanusReasoningEngine
from janus_reasoning_engine.utils.logging import setup_logging
from janus_reasoning_engine.utils.telemetry import TelemetryCollector


class TestCoreInterfaces:
    """Test core interface definitions."""
    
    def test_goal_creation(self):
        """Test Goal dataclass creation."""
        goal = Goal(
            id="goal-1",
            description="Test goal",
            priority=0.8,
            expected_value=100.0,
            feasibility=0.9,
            status=GoalStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        
        assert goal.id == "goal-1"
        assert goal.description == "Test goal"
        assert goal.priority == 0.8
        assert goal.status == GoalStatus.ACTIVE
        assert goal.metadata == {}
    
    def test_strategy_creation(self):
        """Test Strategy dataclass creation."""
        strategy = Strategy(
            id="strategy-1",
            goal_id="goal-1",
            description="Test strategy",
            expected_value=50.0,
            time_estimate=2.0,
            success_probability=0.7,
            resource_requirements={"compute": "low"},
            status=StrategyStatus.PROPOSED,
            created_at=datetime.utcnow(),
        )
        
        assert strategy.id == "strategy-1"
        assert strategy.goal_id == "goal-1"
        assert strategy.success_probability == 0.7
        assert strategy.steps == []
        assert strategy.metadata == {}


class TestEngineConfig:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        
        assert config.memory.hbm_dimension == 10000
        assert config.reasoning.decision_timeout == 30.0
        assert config.safety.max_spending_per_action == 100.0
        assert config.logging.log_level == "INFO"
    
    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = EngineConfig()
        config_dict = config.to_dict()
        
        assert "memory" in config_dict
        assert "reasoning" in config_dict
        assert "safety" in config_dict
        assert config_dict["memory"]["hbm_dimension"] == 10000
    
    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        config_dict = {
            "memory": {"hbm_dimension": 5000},
            "reasoning": {"decision_timeout": 60.0},
            "safety": {"max_spending_per_action": 50.0},
        }
        
        config = EngineConfig.from_dict(config_dict)
        
        assert config.memory.hbm_dimension == 5000
        assert config.reasoning.decision_timeout == 60.0
        assert config.safety.max_spending_per_action == 50.0
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Create and save config
            config = EngineConfig()
            config.memory.hbm_dimension = 8000
            config.save(str(config_path))
            
            # Load config
            loaded_config = EngineConfig.load(str(config_path))
            
            assert loaded_config.memory.hbm_dimension == 8000


class TestLogging:
    """Test logging infrastructure."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(log_level="DEBUG", enable_console=False)
        
        assert logger is not None
        assert logger.name == "janus_reasoning_engine"
    
    def test_logging_with_file(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(
                log_level="INFO",
                log_file=str(log_file),
                enable_console=False
            )
            
            logger.info("Test message")
            
            # Close handlers to release file lock on Windows
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content


class TestTelemetry:
    """Test telemetry system."""
    
    def test_telemetry_creation(self):
        """Test telemetry collector creation."""
        telemetry = TelemetryCollector(enable_telemetry=True)
        
        assert telemetry.enabled
        assert len(telemetry.metrics) == 0
    
    def test_record_metric(self):
        """Test metric recording."""
        telemetry = TelemetryCollector(enable_telemetry=True)
        
        telemetry.record_metric("test_metric", 42.0)
        
        assert len(telemetry.metrics) == 1
        assert telemetry.metrics[0].name == "test_metric"
        assert telemetry.metrics[0].value == 42.0
    
    def test_increment_counter(self):
        """Test counter increment."""
        telemetry = TelemetryCollector(enable_telemetry=True)
        
        telemetry.increment_counter("test_counter")
        telemetry.increment_counter("test_counter", 5)
        
        assert telemetry.counters["test_counter"] == 6
    
    def test_timer(self):
        """Test timer functionality."""
        import time
        
        telemetry = TelemetryCollector(enable_telemetry=True)
        
        telemetry.start_timer("test_timer")
        time.sleep(0.1)
        duration = telemetry.stop_timer("test_timer")
        
        assert duration is not None
        assert duration >= 0.1
        assert len(telemetry.metrics) == 1
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        telemetry = TelemetryCollector(enable_telemetry=True)
        
        telemetry.record_metric("metric1", 10.0)
        telemetry.increment_counter("counter1")
        telemetry.record_decision_latency(0.5)
        
        stats = telemetry.get_statistics()
        
        assert stats["total_metrics"] == 2
        assert stats["counters"]["counter1"] == 1
        assert "decision_latency" in stats


class TestReasoningEngine:
    """Test main reasoning engine."""
    
    def test_engine_creation(self):
        """Test engine creation."""
        engine = JanusReasoningEngine()
        
        assert engine is not None
        assert not engine.initialized
        assert engine.action_count == 0
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = JanusReasoningEngine()
        engine.initialize()
        
        assert engine.initialized
        assert engine.start_time is not None
    
    def test_engine_shutdown(self):
        """Test engine shutdown."""
        engine = JanusReasoningEngine()
        engine.initialize()
        engine.shutdown()
        
        assert not engine.initialized
    
    def test_decide_next_action(self):
        """Test decision making."""
        engine = JanusReasoningEngine()
        engine.initialize()
        
        decision = engine.decide_next_action()
        
        assert decision is not None
        assert decision.decision_type == "assess_state"
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.alternatives_considered) > 0
        assert engine.action_count == 1
    
    def test_reflect_on_actions(self):
        """Test reflection."""
        engine = JanusReasoningEngine()
        engine.initialize()
        
        # Make some decisions
        engine.decide_next_action()
        engine.decide_next_action()
        
        # Reflect
        insights = engine.reflect_on_recent_actions()
        
        assert insights is not None
        assert insights["actions_taken"] == 2
        assert "uptime_seconds" in insights
    
    def test_get_status(self):
        """Test status retrieval."""
        engine = JanusReasoningEngine()
        engine.initialize()
        
        status = engine.get_status()
        
        assert status["initialized"]
        assert status["action_count"] == 0
        assert "subsystems" in status
        assert "telemetry" in status
    
    def test_engine_without_initialization(self):
        """Test that engine requires initialization."""
        engine = JanusReasoningEngine()
        
        with pytest.raises(RuntimeError):
            engine.decide_next_action()
    
    def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        config = EngineConfig()
        config.reasoning.decision_timeout = 60.0
        config.safety.max_spending_per_action = 50.0
        
        engine = JanusReasoningEngine(config=config)
        
        assert engine.config.reasoning.decision_timeout == 60.0
        assert engine.config.safety.max_spending_per_action == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
