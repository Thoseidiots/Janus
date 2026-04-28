# Janus Autonomous Reasoning Engine

The brain that orchestrates all Janus systems to achieve true autonomy. Provides goal-directed reasoning, strategic planning, and adaptive decision-making.

## Overview

The Janus Reasoning Engine is the central orchestrator that enables Janus to operate autonomously like a human with a computer:
- Wake up and decide what to do
- Discover opportunities on the internet
- Learn new skills when needed
- Take on work that's interesting and profitable
- Adapt strategies based on results
- Manage time and priorities
- Earn money autonomously

**Not a bot following scripts. A digital entity with genuine agency.**

## Architecture

### Core Components

1. **ReasoningEngine** - Top-level orchestrator
   - Coordinates all subsystems
   - Makes high-level decisions
   - Reflects on performance
   - Manages autonomous operation

2. **GoalManager** - Goal management system
   - Creates and tracks goals
   - Decomposes high-level goals into sub-goals
   - Prioritizes goals by value and feasibility
   - Monitors progress

3. **StrategyPlanner** - Strategy generation and planning
   - Generates multiple strategies per goal
   - Evaluates strategies by expected value
   - Creates detailed execution plans
   - Adapts strategies based on feedback

4. **ExecutionMonitor** - Execution tracking
   - Monitors strategy execution
   - Detects stuck/blocked states
   - Triggers adaptations when needed
   - Records outcomes for learning

### Supporting Infrastructure

- **Configuration System** - Centralized parameter management
- **Logging** - Structured logging with decision and reflection tracking
- **Telemetry** - Performance metrics and system health monitoring

## Installation

The reasoning engine is part of the Janus ecosystem. It integrates with:
- JanusGPT (LLM reasoning)
- Holographic Brain Memory (associative memory)
- Autonomous Worker (work execution)
- Computer Use Engine (UI automation)
- Wallet (financial tracking)
- Screen Recorder (observation)

## Quick Start

```python
from janus_reasoning_engine import JanusReasoningEngine, EngineConfig

# Create engine with default config
engine = JanusReasoningEngine()

# Or with custom config
config = EngineConfig()
config.reasoning.exploration_rate = 0.2
config.safety.max_spending_per_action = 50.0
engine = JanusReasoningEngine(config=config)

# Initialize
engine.initialize()

# Make decisions
decision = engine.decide_next_action()
print(f"Decision: {decision.decision_type}")
print(f"Rationale: {decision.rationale}")
print(f"Confidence: {decision.confidence}")

# Reflect on actions
insights = engine.reflect_on_recent_actions()
print(f"Actions taken: {insights['actions_taken']}")

# Get status
status = engine.get_status()
print(f"Initialized: {status['initialized']}")
print(f"Uptime: {status['uptime_seconds']}s")

# Shutdown
engine.shutdown()
```

## Configuration

Configuration is managed through the `EngineConfig` class with subsections:

### Memory Configuration
```python
config.memory.hbm_dimension = 10000
config.memory.sqlite_path = "janus_reasoning.db"
config.memory.max_episodic_memories = 10000
```

### Reasoning Configuration
```python
config.reasoning.decision_timeout = 30.0  # seconds
config.reasoning.max_strategies_per_goal = 5
config.reasoning.exploration_rate = 0.1
```

### Safety Configuration
```python
config.safety.max_spending_per_action = 100.0  # dollars
config.safety.require_approval_threshold = 100.0
config.safety.enable_ethical_filter = True
```

### Integration Configuration
```python
config.integration.janus_gpt_model = "gpt-4"
config.integration.enable_computer_use = True
config.integration.enable_hbm = True
```

### Logging Configuration
```python
config.logging.log_level = "INFO"
config.logging.log_file = "janus_reasoning_engine.log"
config.logging.enable_telemetry = True
```

## Configuration Files

Save and load configuration:
```python
# Save
config.save("config.json")

# Load
config = EngineConfig.load("config.json")

# From environment variables
config = EngineConfig.from_env()
```

Environment variables (prefix with `JANUS_REASONING_`):
- `JANUS_REASONING_HBM_DIMENSION`
- `JANUS_REASONING_SQLITE_PATH`
- `JANUS_REASONING_EXPLORATION_RATE`
- `JANUS_REASONING_MAX_SPENDING`
- `JANUS_REASONING_GPT_MODEL`
- `JANUS_REASONING_LOG_LEVEL`
- `JANUS_REASONING_WORKSPACE`
- `JANUS_REASONING_AUTONOMOUS_MODE`

## Logging

The engine provides structured logging with specialized loggers:

### Decision Logging
Decisions are logged to `janus_decisions.jsonl`:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "type": "select_opportunity",
  "rationale": "High earning potential with good skill match",
  "confidence": 0.85,
  "alternatives": ["learn_new_skill", "continue_current_work"],
  "metadata": {}
}
```

### Reflection Logging
Reflections are logged to `janus_reflections.jsonl`:
```json
{
  "timestamp": "2024-01-15T11:00:00",
  "type": "periodic",
  "insights": "Completed 10 actions with 80% success rate",
  "actions_reviewed": 10,
  "metadata": {}
}
```

## Telemetry

The engine collects performance metrics:

```python
# Get statistics
stats = engine.telemetry.get_statistics()

# Example output:
{
  "total_metrics": 150,
  "counters": {
    "decisions_made": 25,
    "reflections_performed": 3
  },
  "decision_latency": {
    "mean": 0.5,
    "min": 0.2,
    "max": 1.2,
    "count": 25
  }
}
```

Telemetry is written to `janus_telemetry.jsonl`.

## Testing

Run tests:
```bash
pytest janus_reasoning_engine/tests/ -v
```

## Development Status

### ✅ Completed (Task 1)
- Core architecture and interfaces
- Configuration system
- Logging infrastructure
- Telemetry system
- Main reasoning engine implementation
- Comprehensive tests

### 🚧 In Progress
- Memory layer integration (Task 2)
- Goal management system (Task 3)
- Opportunity discovery (Task 5)
- Learning system (Task 6)
- Planning and execution (Task 7)

### 📋 Planned
- Multi-agent collaboration
- Payment and revenue integration
- Client and market integration
- Infrastructure integration
- Advanced reasoning capabilities

## Requirements

This implementation satisfies:
- **REQ-1.1**: Goal setting and management
- **REQ-1.2**: Strategy formation
- **REQ-5.1**: Causal reasoning foundation
- **REQ-9.3**: Transparency through logging and telemetry

## Integration Points

The reasoning engine is designed to integrate with:
- `janus_gpt.py` - LLM-based reasoning
- `holographic_brain_memory/` - Associative memory
- `janus_autonomous_worker.py` - Work execution
- `janus_computer_use.py` - UI automation
- `janus_wallet.py` - Financial tracking
- `janus_checkpoint.py` - State persistence
- `janus_system_orchestrator.py` - System coordination

## License

Part of the Janus ecosystem.

## Contributing

This is the foundational task (Task 1) of the Janus Reasoning Engine implementation. Future tasks will build upon this core architecture to add:
- Memory systems
- Goal decomposition
- Opportunity discovery
- Learning capabilities
- Execution monitoring
- Multi-agent collaboration
- Full autonomous operation

See `.kiro/specs/janus-reasoning-engine/tasks.md` for the complete implementation plan.
