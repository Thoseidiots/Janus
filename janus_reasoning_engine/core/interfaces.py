"""
Core interfaces for the Janus Reasoning Engine.

Defines the abstract base classes that all reasoning components must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class GoalStatus(Enum):
    """Status of a goal in the system."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrategyStatus(Enum):
    """Status of a strategy."""
    PROPOSED = "proposed"
    ACTIVE = "active"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Represents a goal in the reasoning system."""
    id: str
    description: str
    priority: float  # 0.0 to 1.0
    expected_value: float  # Expected value (money, skills, reputation)
    feasibility: float  # 0.0 to 1.0
    status: GoalStatus
    created_at: datetime
    updated_at: datetime
    parent_goal_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Strategy:
    """Represents a strategy for achieving a goal."""
    id: str
    goal_id: str
    description: str
    expected_value: float
    time_estimate: float  # hours
    success_probability: float  # 0.0 to 1.0
    resource_requirements: Dict[str, Any]
    status: StrategyStatus
    created_at: datetime
    steps: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionContext:
    """Context for executing a strategy."""
    strategy_id: str
    goal_id: str
    current_step: int
    total_steps: int
    start_time: datetime
    state: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReasoningDecision:
    """Represents a decision made by the reasoning engine."""
    decision_type: str
    rationale: str
    confidence: float  # 0.0 to 1.0
    alternatives_considered: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReasoningEngine(ABC):
    """
    Abstract base class for the reasoning engine.
    
    The reasoning engine is the top-level orchestrator that coordinates
    goal management, strategy planning, and execution monitoring.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the reasoning engine and all subsystems."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shutdown the reasoning engine."""
        pass
    
    @abstractmethod
    def decide_next_action(self) -> ReasoningDecision:
        """
        Decide what action to take next.
        
        Returns:
            ReasoningDecision with the chosen action and rationale
        """
        pass
    
    @abstractmethod
    def reflect_on_recent_actions(self) -> Dict[str, Any]:
        """
        Reflect on recent actions and outcomes.
        
        Returns:
            Dictionary with reflection insights
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the reasoning engine.
        
        Returns:
            Dictionary with status information
        """
        pass


class GoalManager(ABC):
    """
    Abstract base class for goal management.
    
    Handles goal creation, decomposition, prioritization, and progress tracking.
    """
    
    @abstractmethod
    def create_goal(
        self,
        description: str,
        priority: float,
        expected_value: float,
        parent_goal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            description: Human-readable goal description
            priority: Priority level (0.0 to 1.0)
            expected_value: Expected value of achieving the goal
            parent_goal_id: Optional parent goal for hierarchical goals
            metadata: Optional additional metadata
            
        Returns:
            Created Goal object
        """
        pass
    
    @abstractmethod
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Retrieve a goal by ID.
        
        Args:
            goal_id: Unique goal identifier
            
        Returns:
            Goal object or None if not found
        """
        pass
    
    @abstractmethod
    def get_active_goals(self) -> List[Goal]:
        """
        Get all active goals, sorted by priority.
        
        Returns:
            List of active Goal objects
        """
        pass
    
    @abstractmethod
    def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        """
        Update the status of a goal.
        
        Args:
            goal_id: Unique goal identifier
            status: New status
        """
        pass
    
    @abstractmethod
    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """
        Update progress toward a goal.
        
        Args:
            goal_id: Unique goal identifier
            progress: Progress value (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """
        Break down a high-level goal into sub-goals.
        
        Args:
            goal_id: Goal to decompose
            
        Returns:
            List of sub-goals
        """
        pass


class StrategyPlanner(ABC):
    """
    Abstract base class for strategy planning.
    
    Generates and evaluates strategies for achieving goals.
    """
    
    @abstractmethod
    def generate_strategies(self, goal: Goal) -> List[Strategy]:
        """
        Generate multiple strategies for achieving a goal.
        
        Args:
            goal: Goal to generate strategies for
            
        Returns:
            List of Strategy objects
        """
        pass
    
    @abstractmethod
    def evaluate_strategy(self, strategy: Strategy) -> float:
        """
        Evaluate a strategy's expected utility.
        
        Args:
            strategy: Strategy to evaluate
            
        Returns:
            Utility score (higher is better)
        """
        pass
    
    @abstractmethod
    def select_best_strategy(self, strategies: List[Strategy]) -> Optional[Strategy]:
        """
        Select the best strategy from a list.
        
        Args:
            strategies: List of strategies to choose from
            
        Returns:
            Best strategy or None if no viable strategies
        """
        pass
    
    @abstractmethod
    def create_execution_plan(self, strategy: Strategy) -> List[Dict[str, Any]]:
        """
        Create a detailed execution plan for a strategy.
        
        Args:
            strategy: Strategy to plan
            
        Returns:
            List of execution steps
        """
        pass
    
    @abstractmethod
    def adapt_strategy(
        self,
        strategy: Strategy,
        feedback: Dict[str, Any]
    ) -> Strategy:
        """
        Adapt a strategy based on execution feedback.
        
        Args:
            strategy: Current strategy
            feedback: Execution feedback
            
        Returns:
            Adapted strategy
        """
        pass


class ExecutionMonitor(ABC):
    """
    Abstract base class for execution monitoring.
    
    Tracks execution progress, detects issues, and triggers adaptations.
    """
    
    @abstractmethod
    def start_execution(self, strategy: Strategy) -> ExecutionContext:
        """
        Start executing a strategy.
        
        Args:
            strategy: Strategy to execute
            
        Returns:
            ExecutionContext for tracking
        """
        pass
    
    @abstractmethod
    def update_execution_progress(
        self,
        context: ExecutionContext,
        step_result: Dict[str, Any]
    ) -> None:
        """
        Update execution progress with step result.
        
        Args:
            context: Execution context
            step_result: Result of the current step
        """
        pass
    
    @abstractmethod
    def check_execution_health(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Check if execution is progressing normally.
        
        Args:
            context: Execution context
            
        Returns:
            Health status dictionary
        """
        pass
    
    @abstractmethod
    def detect_stuck_state(self, context: ExecutionContext) -> bool:
        """
        Detect if execution is stuck or blocked.
        
        Args:
            context: Execution context
            
        Returns:
            True if stuck, False otherwise
        """
        pass
    
    @abstractmethod
    def complete_execution(
        self,
        context: ExecutionContext,
        success: bool,
        outcome: Dict[str, Any]
    ) -> None:
        """
        Mark execution as complete and record outcome.
        
        Args:
            context: Execution context
            success: Whether execution succeeded
            outcome: Execution outcome data
        """
        pass
    
    @abstractmethod
    def get_execution_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current execution status for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Status dictionary or None if not executing
        """
        pass
