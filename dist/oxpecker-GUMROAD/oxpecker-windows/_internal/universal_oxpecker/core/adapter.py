from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .events import DebugEvent, Issue, StackFrame, Variable


class LanguageAdapter(ABC):
    """
    Every language family must implement this contract.
    The adapter is responsible for:
      - detecting whether it owns a given target (file / entry-point / language tag)
      - starting / initialising a debug session
      - controlling execution (breakpoints, continue, pause)
      - reading runtime state (stack, variables)
      - normalising errors into the universal Issue model
      - static analysis / syntax checking (optional but encouraged)
    """

    # subclasses MUST set this
    language: str = "unknown"

    # ------------------------------------------------------------------ #
    # Event Factory (DRIFT-EVT-004)                                        #
    # ------------------------------------------------------------------ #
    def create_issue(
        self,
        severity: str,
        message: str,
        source_path: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        fix_suggestion: Optional[str] = None,
        frame: Optional[StackFrame] = None,
    ) -> Issue:
        """Centralized factory for creating Issue events."""
        return Issue(
            severity=severity,
            language=self.language,
            message=message,
            source_path=source_path,
            line=line,
            column=column,
            fix_suggestion=fix_suggestion,
            frame=frame,
        )

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def detect(self, target: str) -> bool:
        """Return True if this adapter should handle *target*."""

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a debug session and return a session-info dict."""

    # ------------------------------------------------------------------ #
    # Execution control                                                    #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def set_breakpoint(self, source_path: str, line: int, column: Optional[int] = None) -> bool:
        """Set a breakpoint. Returns True on success."""

    @abstractmethod
    def continue_execution(self) -> DebugEvent:
        """Resume execution; returns an event describing what happened next."""

    @abstractmethod
    def pause_execution(self) -> DebugEvent:
        """Pause a running process."""

    # ------------------------------------------------------------------ #
    # State inspection                                                     #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def read_stack(self) -> List[StackFrame]:
        """Return the current call stack."""

    @abstractmethod
    def read_variables(self, frame_id: str) -> List[Variable]:
        """Return variables in scope for *frame_id*."""

    # ------------------------------------------------------------------ #
    # Error normalisation                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def normalize_error(self, error: Exception) -> Issue:
        """Convert a raw exception / error object to a universal Issue."""

    # ------------------------------------------------------------------ #
    # Optional static analysis (default: no-op)                           #
    # ------------------------------------------------------------------ #
    def analyze_file(self, source_path: str) -> List[Issue]:
        """
        Run static / syntax analysis on *source_path* and return Issues.
        Override in adapters that support static analysis.
        """
        return []

    def analyze_code(self, code: str, filename: str = "<code>") -> List[Issue]:
        """
        Run static / syntax analysis on an in-memory code string.
        Override in adapters that support static analysis.
        """
        return []
