"""
tool_orchestrator.py
====================
Tool selection and orchestration for the Janus Reasoning Engine.

Maintains a registry of available tools, selects the right tool for a task
based on keyword matching, and dispatches plan steps to the appropriate tool.
All tool calls are wrapped in try/except so a single tool failure never
crashes the orchestrator.

Requirements: REQ-4.2, REQ-8.1, REQ-8.2
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from janus_reasoning_engine.planning.multi_step_planner import PlanStep, StepStatus

logger = logging.getLogger(__name__)

# ── Optional: janus_computer_use (real desktop/browser control) ───────────────
try:
    from janus_computer_use import ComputerUseEngine, BrowserComputerUse  # type: ignore
    _HAS_COMPUTER_USE = True
    logger.info("janus_computer_use available — real desktop control enabled")
except Exception:
    _HAS_COMPUTER_USE = False
    ComputerUseEngine = None  # type: ignore
    BrowserComputerUse = None  # type: ignore
    logger.debug("janus_computer_use not available — computer_use tool will use stub")

# ── Optional: janus_autonomous_worker ────────────────────────────────────────
try:
    from janus_autonomous_worker import AutonomousWorker as _AutonomousWorker  # type: ignore
    _HAS_AUTONOMOUS_WORKER = True
except Exception:
    _HAS_AUTONOMOUS_WORKER = False
    _AutonomousWorker = None


def _run_async(coro) -> Any:
    """Run an async coroutine synchronously, reusing an existing loop if present."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop (e.g. Jupyter) — use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of executing a single plan step."""
    step_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ── Keyword → tool mapping ────────────────────────────────────────────────────

_TOOL_KEYWORDS: Dict[str, List[str]] = {
    "browser": [
        "browse", "search", "web", "url", "http", "website", "internet",
        "google", "find online", "research", "scrape", "navigate",
    ],
    "computer_use": [
        "click", "type", "screenshot", "desktop", "ui", "interface",
        "application", "app", "window", "mouse", "keyboard", "screen",
    ],
    "code_execution": [
        "code", "script", "run", "execute", "python", "program", "compile",
        "test", "debug", "function", "algorithm", "compute", "calculate",
    ],
    "file_manipulation": [
        "file", "folder", "directory", "read", "write", "save", "load",
        "document", "csv", "json", "txt", "pdf", "copy", "move", "delete",
    ],
    "autonomous_worker": [
        "job", "work", "task", "complete", "deliver", "submit", "earn",
        "freelance", "client", "project", "assignment",
    ],
}


# ── ToolRegistry ──────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Registry of available tools.

    Each tool is a callable that accepts a PlanStep and returns a StepResult.
    The five built-in tools are registered by default; additional tools can be
    registered at runtime via :meth:`register`.
    """

    BUILTIN_TOOLS = [
        "computer_use",
        "browser",
        "code_execution",
        "file_manipulation",
        "autonomous_worker",
    ]

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[[PlanStep], StepResult]] = {}
        self._register_defaults()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, handler: Callable[[PlanStep], StepResult]) -> None:
        """Register a tool handler under *name*."""
        self._tools[name] = handler
        logger.debug("Registered tool: %s", name)

    def available_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_handler(self, name: str) -> Optional[Callable[[PlanStep], StepResult]]:
        """Return the handler for *name*, or None if not registered."""
        return self._tools.get(name)

    # ── Default handlers ──────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        """Register handlers for all built-in tools.

        computer_use and browser use the real ComputerUseEngine when
        janus_computer_use is installed; otherwise fall back to stubs.
        """
        # Start with stubs for everything
        for tool_name in self.BUILTIN_TOOLS:
            self.register(tool_name, self._make_stub(tool_name))

        # Override with real implementations where available
        if _HAS_COMPUTER_USE:
            self.register("computer_use", self._computer_use_handler)
            self.register("browser", self._browser_handler)
            logger.info("Real computer_use and browser handlers registered")

        if _HAS_AUTONOMOUS_WORKER:
            self.register("autonomous_worker", self._autonomous_worker_handler)

    @staticmethod
    def _make_stub(tool_name: str) -> Callable[[PlanStep], StepResult]:
        """Create a stub handler that logs and returns success."""
        def _stub(step: PlanStep) -> StepResult:
            logger.info("[%s] Executing step %s: %s", tool_name, step.id, step.description)
            return StepResult(
                step_id=step.id,
                success=True,
                output=f"{tool_name} executed: {step.description}",
                metadata={"tool": tool_name, "stub": True},
            )
        _stub.__name__ = f"_stub_{tool_name}"
        return _stub

    @staticmethod
    def _computer_use_handler(step: PlanStep) -> StepResult:
        """
        Execute a desktop UI action via ComputerUseEngine.

        The step description is passed to the engine's run_goal() method,
        which uses the OBSERVE → PLAN → ACT loop to carry out the task.
        """
        async def _run():
            async with ComputerUseEngine() as engine:
                result = await engine.run_goal(step.description)
                return result

        try:
            logger.info("[computer_use] Running goal: %s", step.description)
            result = _run_async(_run())
            success = getattr(result, "success", True)
            output = getattr(result, "data", str(result))
            return StepResult(
                step_id=step.id,
                success=bool(success),
                output=output,
                metadata={"tool": "computer_use"},
            )
        except Exception as exc:
            logger.warning("computer_use handler failed: %s", exc)
            return StepResult(
                step_id=step.id,
                success=False,
                error=str(exc),
                metadata={"tool": "computer_use"},
            )

    @staticmethod
    def _browser_handler(step: PlanStep) -> StepResult:
        """
        Execute a browser task via BrowserComputerUse inside ComputerUseEngine.

        Handles search, navigation, login, job application, and work submission
        by delegating to the high-level BrowserComputerUse helper.
        """
        desc_lower = step.description.lower()

        async def _run():
            async with ComputerUseEngine() as engine:
                browser = BrowserComputerUse(engine)

                # Route to the most appropriate browser method
                if any(k in desc_lower for k in ("search", "find", "look up", "google")):
                    query = step.description
                    result = await browser.search(query)
                elif any(k in desc_lower for k in ("open", "navigate", "go to", "visit")):
                    # Extract URL or site name from description
                    result = await engine.run_goal(step.description)
                elif any(k in desc_lower for k in ("apply", "submit proposal", "bid")):
                    result = await engine.run_goal(step.description)
                elif any(k in desc_lower for k in ("login", "log in", "sign in")):
                    result = await engine.run_goal(step.description)
                else:
                    # General browser task — let the engine figure it out
                    result = await engine.run_goal(step.description)

                return result

        try:
            logger.info("[browser] Running: %s", step.description)
            result = _run_async(_run())
            success = getattr(result, "success", True)
            output = getattr(result, "data", str(result))
            return StepResult(
                step_id=step.id,
                success=bool(success),
                output=output,
                metadata={"tool": "browser"},
            )
        except Exception as exc:
            logger.warning("browser handler failed: %s", exc)
            return StepResult(
                step_id=step.id,
                success=False,
                error=str(exc),
                metadata={"tool": "browser"},
            )

    @staticmethod
    def _autonomous_worker_handler(step: PlanStep) -> StepResult:
        """
        Delegate to JanusOrchestrator (OBSERVE→PLAN→ACT→EXECUTE→REVIEW) when
        available, falling back to janus_autonomous_worker.

        This is the primary execution path for goal-directed tasks — the
        orchestrator uses ScreenInterpreter + AvusBrain + SkillExecutor to
        actually carry out the work.
        """
        # Try JanusOrchestrator first (real intelligence loop)
        try:
            from janus_reasoning_engine.agents.orchestrator_bridge import OrchestratorBridge
            bridge = OrchestratorBridge()
            if bridge._orchestrator is not None:
                result = bridge.run_goal(step.description, max_cycles=20)
                return StepResult(
                    step_id=step.id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    metadata={"tool": "janus_orchestrator", "cycles": result.cycle_num},
                )
        except Exception as exc:
            logger.warning("JanusOrchestrator delegation failed: %s", exc)

        # Fall back to janus_autonomous_worker
        if _HAS_AUTONOMOUS_WORKER:
            try:
                worker = _AutonomousWorker()
                result = worker.execute_task(step.description)
                return StepResult(
                    step_id=step.id,
                    success=True,
                    output=result,
                    metadata={"tool": "autonomous_worker", "delegated": True},
                )
            except Exception as exc:
                logger.warning("autonomous_worker execution failed: %s", exc)
                return StepResult(
                    step_id=step.id,
                    success=False,
                    error=str(exc),
                    metadata={"tool": "autonomous_worker"},
                )

        return StepResult(
            step_id=step.id,
            success=False,
            error="No execution backend available (JanusOrchestrator and autonomous_worker both unavailable)",
            metadata={"tool": "autonomous_worker"},
        )


# ── ToolOrchestrator ──────────────────────────────────────────────────────────

class ToolOrchestrator:
    """
    Selects and dispatches tools for plan steps.

    Tool selection is keyword-based: the step description and tool_type hint
    are matched against :data:`_TOOL_KEYWORDS`.  If no match is found the
    step's ``tool_type`` field is used directly, falling back to ``"browser"``.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or ToolRegistry()

    # ── Public API ────────────────────────────────────────────────────────────

    def select_tool(self, task_description: str) -> str:
        """
        Select the most appropriate tool for *task_description*.

        Keyword matching is performed against the description (case-insensitive).
        The tool with the most keyword hits wins; ties are broken by the order
        in :data:`_TOOL_KEYWORDS`.

        Args:
            task_description: Natural-language description of the task.

        Returns:
            Tool name string (always a registered tool).
        """
        desc_lower = task_description.lower()
        best_tool = "browser"
        best_score = 0

        for tool, keywords in _TOOL_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_tool = tool

        # Ensure the selected tool is actually registered
        if best_tool not in self.registry.available_tools():
            best_tool = "browser"

        logger.debug("Selected tool '%s' for task: %s", best_tool, task_description)
        return best_tool

    def execute_step(
        self,
        step: PlanStep,
        tools: Optional[List[str]] = None,
    ) -> StepResult:
        """
        Execute *step* using the appropriate registered tool.

        The tool is chosen from *tools* (if provided) or via :meth:`select_tool`.
        All exceptions are caught and returned as a failed :class:`StepResult`.

        Args:
            step:  The PlanStep to execute.
            tools: Optional whitelist of tool names to consider.

        Returns:
            StepResult indicating success or failure.
        """
        # Determine which tool to use
        tool_name = self._resolve_tool(step, tools)

        handler = self.registry.get_handler(tool_name)
        if handler is None:
            msg = f"No handler registered for tool '{tool_name}'"
            logger.error(msg)
            return StepResult(step_id=step.id, success=False, error=msg)

        try:
            result = handler(step)
            if result.success:
                step.status = StepStatus.COMPLETED
            else:
                step.status = StepStatus.FAILED
            return result
        except Exception as exc:
            logger.error("Tool '%s' raised an exception for step %s: %s", tool_name, step.id, exc)
            step.status = StepStatus.FAILED
            return StepResult(
                step_id=step.id,
                success=False,
                error=str(exc),
                metadata={"tool": tool_name},
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_tool(
        self,
        step: PlanStep,
        tools: Optional[List[str]],
    ) -> str:
        """Determine the tool name to use for *step*."""
        available = self.registry.available_tools()

        # If a whitelist is given, restrict to those tools
        if tools:
            available = [t for t in tools if t in available]

        # Prefer the step's explicit tool_type if it's available
        if step.tool_type and step.tool_type in available:
            return step.tool_type

        # Fall back to keyword-based selection
        selected = self.select_tool(step.description)
        if selected in available:
            return selected

        # Last resort: first available tool
        return available[0] if available else "browser"
