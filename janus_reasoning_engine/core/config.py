"""
Configuration system for the Janus Reasoning Engine.

Provides centralized configuration management for all engine parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    hbm_dimension: int = 10000
    hbm_sparsity: float = 0.1
    sqlite_path: str = "janus_reasoning.db"
    checkpoint_dir: str = "checkpoints"
    max_episodic_memories: int = 10000
    max_working_memory_items: int = 50


@dataclass
class ReasoningConfig:
    """Configuration for reasoning parameters."""
    decision_timeout: float = 30.0  # seconds
    planning_timeout: float = 300.0  # seconds
    max_strategies_per_goal: int = 5
    min_strategy_confidence: float = 0.3
    exploration_rate: float = 0.1  # epsilon for exploration vs exploitation
    reflection_interval: int = 10  # reflect every N actions


@dataclass
class ExecutionConfig:
    """Configuration for execution monitoring."""
    max_execution_time: float = 3600.0  # seconds
    stuck_detection_threshold: int = 3  # failed attempts before considering stuck
    checkpoint_interval: int = 300  # seconds between checkpoints
    max_retries: int = 3


@dataclass
class SafetyConfig:
    """Configuration for safety guardrails."""
    max_spending_per_action: float = 100.0  # dollars
    require_approval_threshold: float = 100.0  # dollars
    enable_ethical_filter: bool = True
    enable_credential_protection: bool = True
    max_daily_spending: float = 500.0  # dollars


@dataclass
class IntegrationConfig:
    """Configuration for external system integrations."""
    janus_gpt_model: str = "gpt-4"
    janus_gpt_temperature: float = 0.7
    enable_computer_use: bool = True
    enable_autonomous_worker: bool = True
    enable_wallet: bool = True
    enable_screen_recorder: bool = True
    enable_hbm: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging and telemetry."""
    log_level: str = "INFO"
    log_file: str = "janus_reasoning_engine.log"
    enable_telemetry: bool = True
    telemetry_interval: int = 60  # seconds
    log_decisions: bool = True
    log_reflections: bool = True


@dataclass
class EngineConfig:
    """
    Main configuration class for the Janus Reasoning Engine.
    
    Aggregates all configuration subsystems and provides load/save functionality.
    """
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    workspace_dir: str = "janus_workspace"
    enable_autonomous_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "memory": {
                "hbm_dimension": self.memory.hbm_dimension,
                "hbm_sparsity": self.memory.hbm_sparsity,
                "sqlite_path": self.memory.sqlite_path,
                "checkpoint_dir": self.memory.checkpoint_dir,
                "max_episodic_memories": self.memory.max_episodic_memories,
                "max_working_memory_items": self.memory.max_working_memory_items,
            },
            "reasoning": {
                "decision_timeout": self.reasoning.decision_timeout,
                "planning_timeout": self.reasoning.planning_timeout,
                "max_strategies_per_goal": self.reasoning.max_strategies_per_goal,
                "min_strategy_confidence": self.reasoning.min_strategy_confidence,
                "exploration_rate": self.reasoning.exploration_rate,
                "reflection_interval": self.reasoning.reflection_interval,
            },
            "execution": {
                "max_execution_time": self.execution.max_execution_time,
                "stuck_detection_threshold": self.execution.stuck_detection_threshold,
                "checkpoint_interval": self.execution.checkpoint_interval,
                "max_retries": self.execution.max_retries,
            },
            "safety": {
                "max_spending_per_action": self.safety.max_spending_per_action,
                "require_approval_threshold": self.safety.require_approval_threshold,
                "enable_ethical_filter": self.safety.enable_ethical_filter,
                "enable_credential_protection": self.safety.enable_credential_protection,
                "max_daily_spending": self.safety.max_daily_spending,
            },
            "integration": {
                "janus_gpt_model": self.integration.janus_gpt_model,
                "janus_gpt_temperature": self.integration.janus_gpt_temperature,
                "enable_computer_use": self.integration.enable_computer_use,
                "enable_autonomous_worker": self.integration.enable_autonomous_worker,
                "enable_wallet": self.integration.enable_wallet,
                "enable_screen_recorder": self.integration.enable_screen_recorder,
                "enable_hbm": self.integration.enable_hbm,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_file": self.logging.log_file,
                "enable_telemetry": self.logging.enable_telemetry,
                "telemetry_interval": self.logging.telemetry_interval,
                "log_decisions": self.logging.log_decisions,
                "log_reflections": self.logging.log_reflections,
            },
            "workspace_dir": self.workspace_dir,
            "enable_autonomous_mode": self.enable_autonomous_mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Memory config
        if "memory" in data:
            mem = data["memory"]
            config.memory = MemoryConfig(
                hbm_dimension=mem.get("hbm_dimension", 10000),
                hbm_sparsity=mem.get("hbm_sparsity", 0.1),
                sqlite_path=mem.get("sqlite_path", "janus_reasoning.db"),
                checkpoint_dir=mem.get("checkpoint_dir", "checkpoints"),
                max_episodic_memories=mem.get("max_episodic_memories", 10000),
                max_working_memory_items=mem.get("max_working_memory_items", 50),
            )
        
        # Reasoning config
        if "reasoning" in data:
            reas = data["reasoning"]
            config.reasoning = ReasoningConfig(
                decision_timeout=reas.get("decision_timeout", 30.0),
                planning_timeout=reas.get("planning_timeout", 300.0),
                max_strategies_per_goal=reas.get("max_strategies_per_goal", 5),
                min_strategy_confidence=reas.get("min_strategy_confidence", 0.3),
                exploration_rate=reas.get("exploration_rate", 0.1),
                reflection_interval=reas.get("reflection_interval", 10),
            )
        
        # Execution config
        if "execution" in data:
            exec_cfg = data["execution"]
            config.execution = ExecutionConfig(
                max_execution_time=exec_cfg.get("max_execution_time", 3600.0),
                stuck_detection_threshold=exec_cfg.get("stuck_detection_threshold", 3),
                checkpoint_interval=exec_cfg.get("checkpoint_interval", 300),
                max_retries=exec_cfg.get("max_retries", 3),
            )
        
        # Safety config
        if "safety" in data:
            safe = data["safety"]
            config.safety = SafetyConfig(
                max_spending_per_action=safe.get("max_spending_per_action", 100.0),
                require_approval_threshold=safe.get("require_approval_threshold", 100.0),
                enable_ethical_filter=safe.get("enable_ethical_filter", True),
                enable_credential_protection=safe.get("enable_credential_protection", True),
                max_daily_spending=safe.get("max_daily_spending", 500.0),
            )
        
        # Integration config
        if "integration" in data:
            integ = data["integration"]
            config.integration = IntegrationConfig(
                janus_gpt_model=integ.get("janus_gpt_model", "gpt-4"),
                janus_gpt_temperature=integ.get("janus_gpt_temperature", 0.7),
                enable_computer_use=integ.get("enable_computer_use", True),
                enable_autonomous_worker=integ.get("enable_autonomous_worker", True),
                enable_wallet=integ.get("enable_wallet", True),
                enable_screen_recorder=integ.get("enable_screen_recorder", True),
                enable_hbm=integ.get("enable_hbm", True),
            )
        
        # Logging config
        if "logging" in data:
            log = data["logging"]
            config.logging = LoggingConfig(
                log_level=log.get("log_level", "INFO"),
                log_file=log.get("log_file", "janus_reasoning_engine.log"),
                enable_telemetry=log.get("enable_telemetry", True),
                telemetry_interval=log.get("telemetry_interval", 60),
                log_decisions=log.get("log_decisions", True),
                log_reflections=log.get("log_reflections", True),
            )
        
        # Global settings
        config.workspace_dir = data.get("workspace_dir", "janus_workspace")
        config.enable_autonomous_mode = data.get("enable_autonomous_mode", False)
        
        return config
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration file
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "EngineConfig":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            EngineConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> "EngineConfig":
        """
        Create configuration from environment variables.
        
        Environment variables should be prefixed with JANUS_REASONING_
        
        Returns:
            EngineConfig instance
        """
        config = cls()
        
        # Memory settings
        if os.getenv("JANUS_REASONING_HBM_DIMENSION"):
            config.memory.hbm_dimension = int(os.getenv("JANUS_REASONING_HBM_DIMENSION"))
        if os.getenv("JANUS_REASONING_SQLITE_PATH"):
            config.memory.sqlite_path = os.getenv("JANUS_REASONING_SQLITE_PATH")
        
        # Reasoning settings
        if os.getenv("JANUS_REASONING_EXPLORATION_RATE"):
            config.reasoning.exploration_rate = float(os.getenv("JANUS_REASONING_EXPLORATION_RATE"))
        
        # Safety settings
        if os.getenv("JANUS_REASONING_MAX_SPENDING"):
            config.safety.max_spending_per_action = float(os.getenv("JANUS_REASONING_MAX_SPENDING"))
        
        # Integration settings
        if os.getenv("JANUS_REASONING_GPT_MODEL"):
            config.integration.janus_gpt_model = os.getenv("JANUS_REASONING_GPT_MODEL")
        
        # Logging settings
        if os.getenv("JANUS_REASONING_LOG_LEVEL"):
            config.logging.log_level = os.getenv("JANUS_REASONING_LOG_LEVEL")
        
        # Global settings
        if os.getenv("JANUS_REASONING_WORKSPACE"):
            config.workspace_dir = os.getenv("JANUS_REASONING_WORKSPACE")
        if os.getenv("JANUS_REASONING_AUTONOMOUS_MODE"):
            config.enable_autonomous_mode = os.getenv("JANUS_REASONING_AUTONOMOUS_MODE").lower() == "true"
        
        return config


def get_default_config() -> EngineConfig:
    """
    Get default configuration for the reasoning engine.
    
    Returns:
        EngineConfig with default values
    """
    return EngineConfig()
