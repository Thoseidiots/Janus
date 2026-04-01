from __future__ import annotations
from typing import Dict, List, Optional, Any
import importlib
from .engine import DebugEngine
from .scanner import ScanResult

class OxpeckerOrchestrator:
    """
    Facade for the Universal Oxpecker engine.
    Decouples the CLI from core engine, scanner, and repair logic.
    Implements lazy loading for adapters (PERF-GRAPH-008).
    """
    def __init__(self) -> None:
        self.engine = DebugEngine()
        self._adapters_loaded = False

    def _ensure_adapters(self) -> None:
        if not self._adapters_loaded:
            # Lazy import and register adapters
            adapters_mod = importlib.import_module("universal_oxpecker.adapters")
            for adapter in adapters_mod.ALL_ADAPTERS:
                self.engine.register_adapter(adapter)
            self._adapters_loaded = True

    def get_registered_languages(self) -> List[str]:
        self._ensure_adapters()
        return self.engine.registered_languages()

    def debug_file(self, source_path: str) -> List[Any]:
        self._ensure_adapters()
        return self.engine.debug_file(source_path)

    def scan_path(self, target: str, recursive: bool = True, workers: int = 4) -> List[ScanResult]:
        self._ensure_adapters()
        return self.engine.scan_path(target, recursive=recursive, workers=workers)

    def repair_file(self, source_path: str, output_dir: Optional[str] = None, max_rounds: int = 10) -> Any:
        self._ensure_adapters()
        return self.engine.repair_file(source_path, output_dir=output_dir, max_rounds=max_rounds)

    def repair_project(self, target: str, recursive: bool = True, workers: int = 4, 
                       output_dir: Optional[str] = None, max_rounds: int = 10) -> List[Any]:
        self._ensure_adapters()
        return self.engine.repair_project(target, recursive=recursive, workers=workers, 
                                          output_dir=output_dir, max_rounds=max_rounds)

    def rollback_file(self, working_path: str, steps: int = 1) -> Any:
        self._ensure_adapters()
        return self.engine.rollback_file(working_path, steps=steps)
