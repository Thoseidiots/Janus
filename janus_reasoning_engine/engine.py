"""
engine.py — Main entry point for the Janus Autonomous Reasoning Engine.

Instantiates and wires all subsystems, exposes a clean public API:
  - initialize()   — set up all components
  - run_cycle()    — delegate to AutonomousLoop
  - get_status()   — return status of all subsystems
  - shutdown()     — graceful shutdown

Requirements: All requirements
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from janus_reasoning_engine.core.config import EngineConfig
from janus_reasoning_engine.core.autonomous_loop import AutonomousLoop, CycleOutcome
from janus_reasoning_engine.utils.logging import setup_logging

logger = logging.getLogger("janus.engine")

# Optional: real desktop control engine
try:
    from janus_computer_use import ComputerUseEngine as _ComputerUseEngine  # type: ignore
    _HAS_COMPUTER_USE = True
    logger.info("janus_computer_use available — real desktop control enabled")
except Exception:
    _HAS_COMPUTER_USE = False
    _ComputerUseEngine = None  # type: ignore


class JanusReasoningEngine:
    """
    Top-level orchestrator for the Janus Autonomous Reasoning Engine.

    Wires together all subsystems (memory, goals, discovery, planning,
    reasoning, learning, finance, market, infrastructure, agents, safety,
    advanced capabilities) and exposes a simple lifecycle API.

    All subsystems are optional — the engine degrades gracefully when
    individual components are unavailable.

    Usage::

        engine = JanusReasoningEngine()
        engine.initialize()
        outcome = engine.run_cycle()
        status = engine.get_status()
        engine.shutdown()
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._initialized = False
        self._start_time: Optional[datetime] = None
        self._cycle_count = 0

        # Subsystem references (populated in initialize())
        self._loop: Optional[AutonomousLoop] = None
        self._subsystems: Dict[str, Any] = {}
        self._computer_use_engine: Any = None  # shared ComputerUseEngine instance

        # Set up logging early
        setup_logging(
            log_level=self.config.logging.log_level,
            log_file=self.config.logging.log_file,
            enable_console=True,
        )
        logger.info("JanusReasoningEngine created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up all subsystems and wire them together."""
        if self._initialized:
            logger.warning("Engine already initialized")
            return

        logger.info("Initializing JanusReasoningEngine …")

        # ── ComputerUseEngine (shared desktop control instance) ───────
        # NOTE: We do NOT instantiate ComputerUseEngine here at startup.
        # It is created on-demand inside the tool handlers (as an async
        # context manager) so it never touches the desktop until Janus
        # actually needs to execute a step.
        if _HAS_COMPUTER_USE:
            self._subsystems["computer_use_engine"] = "available"
            logger.info("ComputerUseEngine available — will activate on first desktop task")
        else:
            self._subsystems["computer_use_engine"] = None

        # ── Memory ────────────────────────────────────────────────────
        unified_memory = self._try_init("unified_memory", self._init_unified_memory)
        episodic_memory = self._try_init("episodic_memory", self._init_episodic_memory)
        working_memory = self._try_init("working_memory", self._init_working_memory)

        # ── Goals ─────────────────────────────────────────────────────
        goal_manager = self._try_init("goal_manager", self._init_goal_manager)
        progress_tracker = self._try_init("progress_tracker", self._init_progress_tracker)

        # ── Discovery ─────────────────────────────────────────────────
        opportunity_monitor = self._try_init("opportunity_monitor", self._init_opportunity_monitor)
        opportunity_scorer = self._try_init("opportunity_scorer", self._init_opportunity_scorer)
        exploration_strategy = self._try_init("exploration_strategy", self._init_exploration_strategy)

        # ── Planning ──────────────────────────────────────────────────
        multi_step_planner = self._try_init("multi_step_planner", self._init_multi_step_planner)
        execution_monitor = self._try_init("execution_monitor", self._init_execution_monitor)

        # ── Reasoning ─────────────────────────────────────────────────
        metacognition = self._try_init("metacognition", self._init_metacognition)

        # ── Learning ──────────────────────────────────────────────────
        skill_inventory = self._try_init("skill_inventory", self._init_skill_inventory)

        # ── Finance ───────────────────────────────────────────────────
        financial_manager = self._try_init("financial_manager", self._init_financial_manager)

        # ── Safety ────────────────────────────────────────────────────
        safety_guardrails = self._try_init("safety_guardrails", self._init_safety_guardrails)
        transparency_logger = self._try_init("transparency_logger", self._init_transparency_logger)

        # ── Advanced ──────────────────────────────────────────────────
        causal_horizon = self._try_init("causal_horizon", self._init_causal_horizon)
        advanced_concepts = self._try_init("advanced_concepts", self._init_advanced_concepts)
        human_capabilities = self._try_init("human_capabilities", self._init_human_capabilities)

        # ── Wire the autonomous loop ───────────────────────────────────
        self._loop = AutonomousLoop(
            opportunity_monitor=opportunity_monitor,
            opportunity_scorer=opportunity_scorer,
            exploration_strategy=exploration_strategy,
            multi_step_planner=multi_step_planner,
            execution_monitor=execution_monitor,
            metacognition=metacognition,
            progress_tracker=progress_tracker,
            skill_inventory=skill_inventory,
            episodic_memory=episodic_memory,
            working_memory=working_memory,
            financial_manager=financial_manager,
        )

        self._initialized = True
        self._start_time = datetime.utcnow()
        logger.info("JanusReasoningEngine initialized — %d subsystems active",
                    sum(1 for v in self._subsystems.values() if v is not None))

    def run_cycle(self) -> CycleOutcome:
        """
        Execute one full autonomous loop cycle.

        Returns:
            CycleOutcome describing what happened.

        Raises:
            RuntimeError: If the engine has not been initialized.
        """
        if not self._initialized or self._loop is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._cycle_count += 1
        outcome = self._loop.run_cycle()

        # Log via transparency logger if available
        tl = self._subsystems.get("transparency_logger")
        if tl is not None:
            try:
                tl.log_activity("cycle_complete", {
                    "cycle_num": outcome.cycle_num,
                    "action": outcome.action_taken,
                    "success": outcome.success,
                    "earnings": outcome.earnings,
                })
            except Exception:
                pass

        return outcome

    def get_status(self) -> Dict[str, Any]:
        """
        Return a status snapshot of all subsystems.

        Returns:
            Dict with engine metadata and per-subsystem availability.
        """
        uptime = (
            (datetime.utcnow() - self._start_time).total_seconds()
            if self._start_time else 0.0
        )
        return {
            "initialized": self._initialized,
            "cycle_count": self._cycle_count,
            "uptime_seconds": uptime,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "config": {
                "autonomous_mode": self.config.enable_autonomous_mode,
                "workspace_dir": self.config.workspace_dir,
            },
            "subsystems": {
                name: (value is not None)
                for name, value in self._subsystems.items()
            },
        }

    def set_goal(self, description: str, priority: float = 0.8) -> Optional[str]:
        """
        Set a new high-level goal.

        Args:
            description: Human-readable goal description.
            priority: Priority 0–1 (default 0.8).

        Returns:
            Goal ID string, or None if goal manager unavailable.
        """
        gm = self._subsystems.get("goal_manager")
        if gm is None:
            logger.warning("Goal manager not available — cannot set goal")
            return None
        try:
            goal = gm.create_goal(description=description, priority=priority, expected_value=0.0)
            logger.info("Goal set: %s (id=%s)", description, goal.id)
            return goal.id
        except Exception as exc:
            logger.error("Failed to set goal: %s", exc)
            return None

    def shutdown(self) -> None:
        """Gracefully shut down the engine and all subsystems."""
        if not self._initialized:
            logger.warning("Engine not initialized — nothing to shut down")
            return

        logger.info("Shutting down JanusReasoningEngine …")

        # Shut down goal manager if it has a shutdown method
        gm = self._subsystems.get("goal_manager")
        if gm is not None and hasattr(gm, "shutdown"):
            try:
                gm.shutdown()
            except Exception as exc:
                logger.debug("Goal manager shutdown error: %s", exc)

        self._initialized = False
        self._loop = None
        logger.info("JanusReasoningEngine shutdown complete")

    # ------------------------------------------------------------------
    # Private subsystem initialisers
    # ------------------------------------------------------------------

    def _try_init(self, name: str, init_fn) -> Any:
        """Run init_fn, store result in _subsystems, return it (or None on error)."""
        try:
            instance = init_fn()
            self._subsystems[name] = instance
            if instance is not None:
                logger.debug("Subsystem '%s' initialized", name)
            return instance
        except Exception as exc:
            logger.warning("Subsystem '%s' failed to initialize: %s", name, exc)
            self._subsystems[name] = None
            return None

    def _init_unified_memory(self):
        from janus_reasoning_engine.memory.unified_memory import UnifiedMemory
        return UnifiedMemory(db_path=self.config.memory.sqlite_path)

    def _init_episodic_memory(self):
        from janus_reasoning_engine.memory.episodic_memory import EpisodicMemory
        return EpisodicMemory(db_path=self.config.memory.sqlite_path)

    def _init_working_memory(self):
        from janus_reasoning_engine.memory.working_memory import WorkingMemory
        return WorkingMemory(max_items=self.config.memory.max_working_memory_items)

    def _init_goal_manager(self):
        from janus_reasoning_engine.goals.goal_manager import GoalManagerImpl
        gm = GoalManagerImpl(db_path=self.config.memory.sqlite_path)
        gm.initialize()
        return gm

    def _init_progress_tracker(self):
        from janus_reasoning_engine.goals.progress_tracker import ProgressTracker
        return ProgressTracker()

    def _init_opportunity_monitor(self):
        from janus_reasoning_engine.discovery.opportunity_monitor import OpportunityMonitor
        # engine=None — ComputerUseEngine is created on-demand inside each
        # tool handler only when Janus actually needs to browse
        return OpportunityMonitor(engine=None)

    def _init_opportunity_scorer(self):
        from janus_reasoning_engine.discovery.opportunity_scorer import OpportunityScorer
        # Seed with skills from inventory if available
        return OpportunityScorer()

    def _init_exploration_strategy(self):
        from janus_reasoning_engine.discovery.opportunity_scorer import ExplorationStrategy
        return ExplorationStrategy(
            exploration_rate=self.config.reasoning.exploration_rate
        )

    def _init_multi_step_planner(self):
        from janus_reasoning_engine.planning.multi_step_planner import MultiStepPlanner
        return MultiStepPlanner()

    def _init_execution_monitor(self):
        from janus_reasoning_engine.planning.execution_monitor import ExecutionMonitor
        from janus_reasoning_engine.planning.tool_orchestrator import ToolOrchestrator, ToolRegistry
        # ToolRegistry auto-registers real computer_use/browser handlers
        # when janus_computer_use is installed (no extra wiring needed)
        registry = ToolRegistry()
        orchestrator = ToolOrchestrator(registry=registry)
        return ExecutionMonitor(orchestrator=orchestrator)

    def _init_metacognition(self):
        from janus_reasoning_engine.reasoning.metacognition import Metacognition
        return Metacognition()

    def _init_skill_inventory(self):
        from janus_reasoning_engine.learning.skill_inventory import SkillInventory
        return SkillInventory()

    def _init_financial_manager(self):
        from janus_reasoning_engine.finance.financial_manager import FinancialManager
        return FinancialManager()

    def _init_safety_guardrails(self):
        from janus_reasoning_engine.safety.guardrails import SafetyGuardrails
        return SafetyGuardrails(
            budget_limit=self.config.safety.max_spending_per_action
        )

    def _init_transparency_logger(self):
        from janus_reasoning_engine.safety.transparency_logger import TransparencyLogger
        return TransparencyLogger()

    def _init_causal_horizon(self):
        from janus_reasoning_engine.advanced.causal_horizon_bridge import CausalHorizonBridge
        return CausalHorizonBridge()

    def _init_advanced_concepts(self):
        from janus_reasoning_engine.advanced.advanced_concepts import AdvancedConcepts
        return AdvancedConcepts()

    def _init_human_capabilities(self):
        from janus_reasoning_engine.advanced.human_capabilities import HumanCapabilities
        return HumanCapabilities()
