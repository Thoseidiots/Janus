"""
Universal Oxpecker analysis strategy.

Delegates static analysis to the universal_oxpecker DebugEngine, which
supports 13 language families through its adapter registry.  For each
source file found in an application's installation directory the engine
runs syntax checking and fix-suggestion, then the results are translated
into Capability objects that describe the languages and error patterns
the application works with.

This strategy is most useful for:
- Applications that ship source files (scripts, plugins, config DSLs)
- Janus internal tools that need their own code analysed
- Any app whose install directory contains readable source code
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType


logger = logging.getLogger(__name__)

# Extensions the oxpecker adapters cover (mirrors manifest.json language_families)
_OXPECKER_EXTENSIONS = {
    ".py", ".py3", ".pyw",                          # Python
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",  # JavaScript / TypeScript
    ".java", ".kt", ".kts",                         # Java / Kotlin
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",     # C / C++
    ".rs",                                           # Rust
    ".go",                                           # Go
    ".cs",                                           # C#
    ".rb",                                           # Ruby
    ".php",                                          # PHP
    ".lua",                                          # Lua
    ".swift",                                        # Swift
    ".zig",                                          # Zig
}

# Map language names (from adapter.language) to CapabilityCategory
_LANG_CATEGORY = {
    "python":              CapabilityCategory.DEVELOPMENT_TOOLS,
    "javascript":          CapabilityCategory.DEVELOPMENT_TOOLS,
    "typescript":          CapabilityCategory.DEVELOPMENT_TOOLS,
    "java":                CapabilityCategory.DEVELOPMENT_TOOLS,
    "kotlin":              CapabilityCategory.DEVELOPMENT_TOOLS,
    "c":                   CapabilityCategory.DEVELOPMENT_TOOLS,
    "c++":                 CapabilityCategory.DEVELOPMENT_TOOLS,
    "rust":                CapabilityCategory.DEVELOPMENT_TOOLS,
    "go":                  CapabilityCategory.DEVELOPMENT_TOOLS,
    "csharp":              CapabilityCategory.DEVELOPMENT_TOOLS,
    "ruby":                CapabilityCategory.DEVELOPMENT_TOOLS,
    "php":                 CapabilityCategory.DEVELOPMENT_TOOLS,
    "lua":                 CapabilityCategory.DEVELOPMENT_TOOLS,
    "swift":               CapabilityCategory.DEVELOPMENT_TOOLS,
    "zig":                 CapabilityCategory.DEVELOPMENT_TOOLS,
}

# Maximum source files to scan per application (keeps analysis fast)
_MAX_FILES = 30
# Maximum issues to surface per file (avoids noise from very broken files)
_MAX_ISSUES_PER_FILE = 10


class OxpeckerStrategy(BaseAnalysisStrategy):
    """
    Capability analysis strategy backed by the universal_oxpecker DebugEngine.

    For each application the strategy:
    1. Finds source files in the install directory (up to _MAX_FILES)
    2. Runs the oxpecker engine on each file to detect language and issues
    3. Produces one capability per language family found, describing the
       syntax-checking and repair services available for that language
    4. Attaches issue counts and severity summaries to the capability description
    """

    def __init__(self):
        super().__init__("oxpecker_analysis", confidence_factor=0.9)
        self._engine = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    # Strategy interface                                                   #
    # ------------------------------------------------------------------ #

    def can_analyze(self, app: Application) -> bool:
        """True if the install directory contains any oxpecker-supported source files."""
        if not app.is_accessible:
            return False
        source_files = self._collect_source_files(app.installation_path, limit=1)
        return len(source_files) > 0

    def extract_capabilities(self, app: Application) -> List[Capability]:
        engine = self._get_engine()
        if engine is None:
            return []

        source_files = self._collect_source_files(app.installation_path, limit=_MAX_FILES)
        if not source_files:
            return []

        # language -> {files, errors, warnings}
        lang_stats: dict = {}

        for file_path in source_files:
            try:
                adapter = engine.detect_adapter(str(file_path))
                language = adapter.language

                if language not in lang_stats:
                    lang_stats[language] = {"files": 0, "errors": 0, "warnings": 0, "infos": 0}

                lang_stats[language]["files"] += 1

                issues = engine.debug_file(str(file_path))
                for issue in issues[:_MAX_ISSUES_PER_FILE]:
                    sev = getattr(issue, "severity", "info").lower()
                    if sev == "error":
                        lang_stats[language]["errors"] += 1
                    elif sev == "warning":
                        lang_stats[language]["warnings"] += 1
                    else:
                        lang_stats[language]["infos"] += 1

            except ValueError:
                # No adapter for this file — skip silently
                continue
            except Exception as exc:
                self.logger.debug(f"Oxpecker error on {file_path}: {exc}")
                continue

        return [
            self._build_capability(app, language, stats)
            for language, stats in lang_stats.items()
        ]

    # ------------------------------------------------------------------ #
    # Engine bootstrap                                                     #
    # ------------------------------------------------------------------ #

    def _get_engine(self):
        """
        Lazy-load and configure the DebugEngine with all available adapters.
        Returns None if the oxpecker package cannot be imported.
        """
        if self._engine is not None:
            return self._engine

        try:
            # Add the Janus-repo root to sys.path so the import resolves
            janus_tools = Path(__file__).resolve().parents[3] / "Janus-repo" / "tools"
            if str(janus_tools) not in sys.path:
                sys.path.insert(0, str(janus_tools))

            from universal_oxpecker.core.engine import DebugEngine
            from universal_oxpecker.adapters.python_adapter import PythonAdapter
            from universal_oxpecker.adapters.javascript_typescript_adapter import JavaScriptTypeScriptAdapter
            from universal_oxpecker.adapters.java_kotlin_adapter import JavaKotlinAdapter
            from universal_oxpecker.adapters.c_cpp_adapter import CCppAdapter
            from universal_oxpecker.adapters.rust_adapter import RustAdapter
            from universal_oxpecker.adapters.go_adapter import GoAdapter
            from universal_oxpecker.adapters.csharp_adapter import CSharpAdapter
            from universal_oxpecker.adapters.ruby_adapter import RubyAdapter
            from universal_oxpecker.adapters.php_adapter import PhpAdapter
            from universal_oxpecker.adapters.lua_adapter import LuaAdapter
            from universal_oxpecker.adapters.swift_adapter import SwiftAdapter
            from universal_oxpecker.adapters.zig_adapter import ZigAdapter

            engine = DebugEngine()
            for adapter_cls in [
                PythonAdapter, JavaScriptTypeScriptAdapter, JavaKotlinAdapter,
                CCppAdapter, RustAdapter, GoAdapter, CSharpAdapter,
                RubyAdapter, PhpAdapter, LuaAdapter, SwiftAdapter, ZigAdapter,
            ]:
                engine.register_adapter(adapter_cls())

            self._engine = engine
            self.logger.info(
                f"Oxpecker engine ready — "
                f"languages: {engine.registered_languages()}"
            )
            return self._engine

        except ImportError as exc:
            self.logger.warning(
                f"universal_oxpecker not available, skipping oxpecker strategy: {exc}"
            )
            return None

    # ------------------------------------------------------------------ #
    # File collection                                                      #
    # ------------------------------------------------------------------ #

    def _collect_source_files(self, directory: Path, limit: int) -> List[Path]:
        """Walk the directory and return up to `limit` supported source files."""
        if not directory.exists():
            return []

        found: List[Path] = []
        try:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() in _OXPECKER_EXTENSIONS:
                    found.append(path)
                    if len(found) >= limit:
                        break
        except (PermissionError, OSError):
            pass

        return found

    # ------------------------------------------------------------------ #
    # Capability builder                                                   #
    # ------------------------------------------------------------------ #

    def _build_capability(
        self, app: Application, language: str, stats: dict
    ) -> Capability:
        """Translate per-language scan stats into a Capability object."""
        files = stats["files"]
        errors = stats["errors"]
        warnings = stats["warnings"]

        category = _LANG_CATEGORY.get(language.lower(), CapabilityCategory.DEVELOPMENT_TOOLS)

        # Confidence is high when we found files and low when there are many errors
        # (many errors may mean the adapter is misidentifying the language)
        error_ratio = errors / max(files, 1)
        confidence = max(0.5, 0.95 - error_ratio * 0.1)

        description = (
            f"Contains {files} {language} source file(s). "
            f"Oxpecker static analysis: {errors} error(s), {warnings} warning(s). "
            f"Supports syntax checking, fix suggestions, and automated repair "
            f"via the universal_oxpecker engine."
        )

        return self._create_capability(
            app=app,
            name=f"{language.title()} Source Analysis",
            category=category,
            interface_type=InterfaceType.LIBRARY,
            description=description,
            confidence=confidence,
            supported_formats=[
                ext for ext, _ in [
                    (e, None) for e in _OXPECKER_EXTENSIONS
                    if language.lower() in e or e.lstrip(".") in language.lower()
                ]
            ],
        )
