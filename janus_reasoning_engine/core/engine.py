"""
Main reasoning engine implementation.

Orchestrates all subsystems to provide autonomous reasoning capabilities.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from janus_reasoning_engine.core.interfaces import (
    ReasoningEngine,
    GoalManager,
    StrategyPlanner,
    ExecutionMonitor,
    ReasoningDecision,
)
from janus_reasoning_engine.core.config import EngineConfig
from janus_reasoning_engine.utils.logging import setup_logging, DecisionLogger, ReflectionLogger
from janus_reasoning_engine.utils.telemetry import TelemetryCollector


class JanusReasoningEngine(ReasoningEngine):
    """
    Main implementation of the Janus Reasoning Engine.
    
    Coordinates goal management, strategy planning, and execution monitoring
    to enable autonomous operation.
    """
    
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        goal_manager: Optional[GoalManager] = None,
        strategy_planner: Optional[StrategyPlanner] = None,
        execution_monitor: Optional[ExecutionMonitor] = None,
    ):
        """
        Initialize the reasoning engine.
        
        Args:
            config: Engine configuration
            goal_manager: Goal management subsystem
            strategy_planner: Strategy planning subsystem
            execution_monitor: Execution monitoring subsystem
        """
        self.config = config or EngineConfig()
        
        # Set up logging
        self.logger = setup_logging(
            log_level=self.config.logging.log_level,
            log_file=self.config.logging.log_file,
            enable_console=True
        )
        
        # Set up specialized loggers
        self.decision_logger = DecisionLogger(self.logger)
        self.reflection_logger = ReflectionLogger(self.logger)
        
        # Set up telemetry
        self.telemetry = TelemetryCollector(
            enable_telemetry=self.config.logging.enable_telemetry,
            output_file="janus_telemetry.jsonl"
        )
        
        # Subsystems (will be injected or created)
        self.goal_manager = goal_manager
        self.strategy_planner = strategy_planner
        self.execution_monitor = execution_monitor
        
        # Engine state
        self.initialized = False
        self.action_count = 0
        self.start_time: Optional[datetime] = None
        
        self.logger.info("Janus Reasoning Engine created")
    
    def initialize(self) -> None:
        """Initialize the reasoning engine and all subsystems."""
        if self.initialized:
            self.logger.warning("Engine already initialized")
            return
        
        self.logger.info("Initializing Janus Reasoning Engine...")
        
        # Validate subsystems
        if self.goal_manager is None:
            self.logger.warning("No GoalManager provided - goal management will be limited")
        
        if self.strategy_planner is None:
            self.logger.warning("No StrategyPlanner provided - strategy planning will be limited")
        
        if self.execution_monitor is None:
            self.logger.warning("No ExecutionMonitor provided - execution monitoring will be limited")
        
        # Initialize telemetry
        self.telemetry.increment_counter("engine_initializations")
        
        # Mark as initialized
        self.initialized = True
        self.start_time = datetime.utcnow()
        
        self.logger.info("Janus Reasoning Engine initialized successfully")
    
    def shutdown(self) -> None:
        """Gracefully shutdown the reasoning engine."""
        if not self.initialized:
            self.logger.warning("Engine not initialized, nothing to shutdown")
            return
        
        self.logger.info("Shutting down Janus Reasoning Engine...")
        
        # Flush telemetry
        self.telemetry.flush()
        
        # Log final statistics
        stats = self.telemetry.get_statistics()
        self.logger.info(f"Final statistics: {stats}")
        
        # Mark as shutdown
        self.initialized = False
        
        self.logger.info("Janus Reasoning Engine shutdown complete")
    
    def decide_next_action(self) -> ReasoningDecision:
        """
        Decide what action to take next.
        
        This is the core decision-making method that determines what Janus
        should do at any given moment.
        
        Returns:
            ReasoningDecision with the chosen action and rationale
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        self.logger.info("Deciding next action...")
        self.telemetry.start_timer("decision_making")
        
        try:
            # For now, return a placeholder decision
            # This will be implemented in later tasks with actual reasoning logic
            decision = ReasoningDecision(
                decision_type="assess_state",
                rationale="Initial implementation - assessing current state",
                confidence=0.5,
                alternatives_considered=["discover_opportunities", "continue_work", "learn_skill"],
                timestamp=datetime.utcnow(),
                metadata={
                    "action_count": self.action_count,
                    "engine_uptime": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                }
            )
            
            # Log decision
            if self.config.logging.log_decisions:
                self.decision_logger.log_decision(
                    decision_type=decision.decision_type,
                    rationale=decision.rationale,
                    confidence=decision.confidence,
                    alternatives=decision.alternatives_considered,
                    metadata=decision.metadata
                )
            
            # Record telemetry
            duration = self.telemetry.stop_timer("decision_making")
            if duration:
                self.telemetry.record_decision_latency(duration)
            
            self.action_count += 1
            self.telemetry.increment_counter("decisions_made")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error during decision making: {e}", exc_info=True)
            self.telemetry.increment_counter("decision_errors")
            raise
    
    def reflect_on_recent_actions(self) -> Dict[str, Any]:
        """
        Reflect on recent actions and outcomes.
        
        Implements metacognition - thinking about thinking.
        
        Returns:
            Dictionary with reflection insights
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        self.logger.info("Reflecting on recent actions...")
        self.telemetry.start_timer("reflection")
        
        try:
            # Get telemetry statistics
            stats = self.telemetry.get_statistics()
            
            # Generate reflection insights
            insights = {
                "actions_taken": self.action_count,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                "telemetry_stats": stats,
                "reflection_timestamp": datetime.utcnow().isoformat(),
            }
            
            # Log reflection
            if self.config.logging.log_reflections:
                self.reflection_logger.log_reflection(
                    reflection_type="periodic",
                    insights=f"Completed {self.action_count} actions",
                    actions_reviewed=self.action_count,
                    metadata=insights
                )
            
            # Record telemetry
            self.telemetry.stop_timer("reflection")
            self.telemetry.increment_counter("reflections_performed")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error during reflection: {e}", exc_info=True)
            self.telemetry.increment_counter("reflection_errors")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the reasoning engine.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "initialized": self.initialized,
            "action_count": self.action_count,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": {
                "autonomous_mode": self.config.enable_autonomous_mode,
                "workspace_dir": self.config.workspace_dir,
            },
            "subsystems": {
                "goal_manager": self.goal_manager is not None,
                "strategy_planner": self.strategy_planner is not None,
                "execution_monitor": self.execution_monitor is not None,
            },
            "telemetry": self.telemetry.get_statistics(),
        }
        
        return status
    
    def set_goal_manager(self, goal_manager: GoalManager) -> None:
        """
        Set the goal manager subsystem.
        
        Args:
            goal_manager: GoalManager instance
        """
        self.goal_manager = goal_manager
        self.logger.info("Goal manager set")
    
    def set_strategy_planner(self, strategy_planner: StrategyPlanner) -> None:
        """
        Set the strategy planner subsystem.
        
        Args:
            strategy_planner: StrategyPlanner instance
        """
        self.strategy_planner = strategy_planner
        self.logger.info("Strategy planner set")
    
    def set_execution_monitor(self, execution_monitor: ExecutionMonitor) -> None:
        """
        Set the execution monitor subsystem.
        
        Args:
            execution_monitor: ExecutionMonitor instance
        """
        self.execution_monitor = execution_monitor
        self.logger.info("Execution monitor set")
