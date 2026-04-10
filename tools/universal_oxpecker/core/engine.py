from __future__ import annotations
from typing import Dict, List, Optional
from .adapter import LanguageAdapter
from .events import DebugEvent, Issue
from .scanner import ProjectScanner, ScanResult


class DebugEngine:
    """
    Central orchestrator.  Maintains a registry of adapters and the
    current active session.
    """

    def __init__(self) -> None:
        self._adapters: List[LanguageAdapter] = []
        self._session: Dict[str, object] = {}
        self._scanner = ProjectScanner(self)

    # ------------------------------------------------------------------ #
    # Adapter registry                                                     #
    # ------------------------------------------------------------------ #
    def register_adapter(self, adapter: LanguageAdapter) -> None:
        self._adapters.append(adapter)

    def registered_languages(self) -> List[str]:
        return [a.language for a in self._adapters]

    def detect_adapter(self, target: str) -> LanguageAdapter:
        for adapter in self._adapters:
            if adapter.detect(target):
                return adapter
        raise ValueError(
            f"No adapter found for target: '{target}'.\n"
            f"Registered languages: {self.registered_languages()}"
        )

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #
    def start(self, target: str, options: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        adapter = self.detect_adapter(target)
        self._session = adapter.initialize(target, options)
        self._session["adapter"] = adapter
        self._session["target"] = target
        return self._session

    def active_adapter(self) -> LanguageAdapter:
        if "adapter" not in self._session:
            raise RuntimeError("No active session.  Call start() first.")
        return self._session["adapter"]  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Execution control (delegate to active adapter)                      #
    # ------------------------------------------------------------------ #
    def set_breakpoint(self, source_path: str, line: int,
                       column: Optional[int] = None) -> bool:
        return self.active_adapter().set_breakpoint(source_path, line, column)

    def continue_execution(self) -> DebugEvent:
        return self.active_adapter().continue_execution()

    def pause_execution(self) -> DebugEvent:
        return self.active_adapter().pause_execution()

    def read_stack(self):
        return self.active_adapter().read_stack()

    def read_variables(self, frame_id: str):
        return self.active_adapter().read_variables(frame_id)

    # ------------------------------------------------------------------ #
    # Error / analysis (delegate to active adapter)                       #
    # ------------------------------------------------------------------ #
    def analyze_error(self, error: Exception) -> Issue:
        return self.active_adapter().normalize_error(error)

    def analyze_file(self, source_path: str) -> List[Issue]:
        # Use the adapter for the file's language
        adapter = self.detect_adapter(source_path)
        return adapter.analyze_file(source_path)

    def analyze_code(self, code: str, language_hint: str = "",
                     filename: str = "<code>") -> List[Issue]:
        target = language_hint or filename
        adapter = self.detect_adapter(target)
        return adapter.analyze_code(code, filename)

    def scan_path(self, target: str, recursive: bool = True,
                  workers: int = 4) -> List[ScanResult]:
        return self._scanner.scan(target, recursive=recursive, workers=workers)

    # ------------------------------------------------------------------ #
    # Convenience: scan a whole file for issues and suggest fixes         #
    # ------------------------------------------------------------------ #
    def debug_file(self, source_path: str) -> List[Issue]:
        """
        Full pipeline for a source file:
          1. detect language adapter
          2. run static / syntax analysis
          3. attach fix suggestions via the analysis layer
        """
        try:
            from ..analysis.fix_suggester import FixSuggester  # lazy import
        except ImportError:  # pragma: no cover - top-level package fallback
            from analysis.fix_suggester import FixSuggester

        adapter = self.detect_adapter(source_path)
        issues = adapter.analyze_file(source_path)
        suggester = FixSuggester()
        suggester.enrich(issues)
        return issues

    def repair_file(self, source_path: str, output_dir: Optional[str] = None,
                    max_rounds: int = 10):
        try:
            from ..analysis.repair_workflow import AutomatedProgramRepair  # lazy import
        except ImportError:  # pragma: no cover - top-level package fallback
            from analysis.repair_workflow import AutomatedProgramRepair
        repairer = AutomatedProgramRepair(self)
        return repairer.repair_file(source_path, output_dir=output_dir, max_rounds=max_rounds)

    def repair_project(self, target: str, recursive: bool = True, workers: int = 4,
                       output_dir: Optional[str] = None, max_rounds: int = 10):
        try:
            from ..analysis.repair_workflow import AutomatedProgramRepair  # lazy import
        except ImportError:  # pragma: no cover - top-level package fallback
            from analysis.repair_workflow import AutomatedProgramRepair
        repairer = AutomatedProgramRepair(self)
        return repairer.repair_project(
            target,
            recursive=recursive,
            workers=workers,
            output_dir=output_dir,
            max_rounds=max_rounds,
        )

    def rollback_file(self, working_path: str, steps: int = 1):
        try:
            from ..analysis.repair_workflow import AutomatedProgramRepair  # lazy import
        except ImportError:  # pragma: no cover - top-level package fallback
            from analysis.repair_workflow import AutomatedProgramRepair
        repairer = AutomatedProgramRepair(self)
        return repairer.rollback_last(working_path, steps=steps)

    def rollback_history(self, working_path: str):
        try:
            from ..analysis.repair_workflow import AutomatedProgramRepair  # lazy import
        except ImportError:  # pragma: no cover - top-level package fallback
            from analysis.repair_workflow import AutomatedProgramRepair
        repairer = AutomatedProgramRepair(self)
        return repairer.list_rollback_history(working_path)
