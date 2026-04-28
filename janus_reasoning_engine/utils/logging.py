"""
Logging infrastructure for the Janus Reasoning Engine.

Provides structured logging with decision tracking and reflection logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class DecisionLogger:
    """Logger for reasoning decisions with structured output."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.decisions_file = Path("janus_decisions.jsonl")
    
    def log_decision(
        self,
        decision_type: str,
        rationale: str,
        confidence: float,
        alternatives: list,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log a reasoning decision.
        
        Args:
            decision_type: Type of decision made
            rationale: Explanation of the decision
            confidence: Confidence level (0.0 to 1.0)
            alternatives: Alternative options considered
            metadata: Additional metadata
        """
        decision_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": decision_type,
            "rationale": rationale,
            "confidence": confidence,
            "alternatives": alternatives,
            "metadata": metadata or {}
        }
        
        # Log to standard logger
        self.logger.info(
            f"Decision: {decision_type} (confidence: {confidence:.2f}) - {rationale}"
        )
        
        # Append to decisions file
        with open(self.decisions_file, 'a') as f:
            f.write(json.dumps(decision_record) + '\n')


class ReflectionLogger:
    """Logger for reflection and metacognition."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.reflections_file = Path("janus_reflections.jsonl")
    
    def log_reflection(
        self,
        reflection_type: str,
        insights: str,
        actions_reviewed: int,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log a reflection session.
        
        Args:
            reflection_type: Type of reflection
            insights: Key insights from reflection
            actions_reviewed: Number of actions reviewed
            metadata: Additional metadata
        """
        reflection_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": reflection_type,
            "insights": insights,
            "actions_reviewed": actions_reviewed,
            "metadata": metadata or {}
        }
        
        # Log to standard logger
        self.logger.info(
            f"Reflection: {reflection_type} - {insights} ({actions_reviewed} actions reviewed)"
        )
        
        # Append to reflections file
        with open(self.reflections_file, 'a') as f:
            f.write(json.dumps(reflection_record) + '\n')


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up logging for the reasoning engine.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("janus_reasoning_engine")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"janus_reasoning_engine.{name}")
