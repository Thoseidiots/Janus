from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

try:
    from .events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from core.events import Issue


@dataclass
class ScanResult:
    source_path: str
    language: str
    issues: List[Issue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "info")


class ProjectScanner:
    """
    Scans one file or a whole directory tree using the engine's adapter registry.
    Multiple worker threads can be used to let more "Oxpeckers" inspect files.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    def scan(self, target: str, recursive: bool = True, workers: int = 4) -> List[ScanResult]:
        paths = self._collect_paths(target, recursive=recursive)
        if not paths:
            return []

        if workers <= 1 or len(paths) == 1:
            results = [self._scan_one(path) for path in paths]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                future_map = {pool.submit(self._scan_one, path): path for path in paths}
                for future in as_completed(future_map):
                    results.append(future.result())

        return sorted(results, key=lambda item: item.source_path)

    def _collect_paths(self, target: str, recursive: bool) -> List[str]:
        if os.path.isfile(target):
            return [target] if self._supports(target) else []

        if not os.path.isdir(target):
            return []

        discovered: List[str] = []
        if recursive:
            for root, _, files in os.walk(target):
                for name in files:
                    path = os.path.join(root, name)
                    if self._supports(path):
                        discovered.append(path)
        else:
            for name in os.listdir(target):
                path = os.path.join(target, name)
                if os.path.isfile(path) and self._supports(path):
                    discovered.append(path)
        return sorted(discovered)

    def _supports(self, path: str) -> bool:
        try:
            self.engine.detect_adapter(path)
            return True
        except ValueError:
            return False

    def _scan_one(self, path: str) -> ScanResult:
        adapter = self.engine.detect_adapter(path)
        issues = self.engine.debug_file(path)
        return ScanResult(source_path=path, language=adapter.language, issues=issues)
