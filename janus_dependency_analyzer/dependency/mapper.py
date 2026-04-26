"""
Dependency Mapper for the Janus Dependency Analyzer.

This module scans the Janus codebase for external application invocations
using AST parsing (for Python) and regex patterns (for all languages).
It also identifies potentially beneficial applications not currently used.
"""

import ast
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Applications that are Python builtins or internal references — skip these
SKIP_LIST: Set[str] = {
    "python", "python3", "pip", "pip3", "sys", "os", "self", "cls",
    "true", "false", "null", "none",
}

# File extensions to scan, grouped by language
PYTHON_EXTENSIONS = {".py"}
RUST_EXTENSIONS = {".rs"}
TS_JS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}

ALL_EXTENSIONS = PYTHON_EXTENSIONS | RUST_EXTENSIONS | TS_JS_EXTENSIONS

# Directories to skip entirely
SKIP_DIRS: Set[str] = {
    ".git", "__pycache__", "node_modules", ".hypothesis",
}

# Known valid invocation methods
VALID_INVOCATION_METHODS = {
    "subprocess", "os.system", "os.popen", "shell_command", "api_call",
}

# Common tool names to look for in string literals
COMMON_TOOLS = {
    "git", "ffmpeg", "curl", "wget", "docker", "npm", "node",
    "cargo", "rustc", "gcc", "make", "cmake",
}

# Regex patterns for each language
PYTHON_SUBPROCESS_PATTERN = re.compile(
    r'subprocess\.(?:run|call|check_output|Popen)\s*\(\s*[\[\'"](\w[\w.\-]*)',
    re.MULTILINE,
)
PYTHON_OS_SYSTEM_PATTERN = re.compile(
    r'os\.system\s*\(\s*[\'"](\w[\w.\-]*)',
    re.MULTILINE,
)
PYTHON_OS_POPEN_PATTERN = re.compile(
    r'os\.popen\s*\(\s*[\'"](\w[\w.\-]*)',
    re.MULTILINE,
)
RUST_COMMAND_PATTERN = re.compile(
    r'Command::new\s*\(\s*"(\w[\w.\-]*)"',
    re.MULTILINE,
)
TS_EXEC_PATTERN = re.compile(
    r'exec\s*\(\s*[\'"`](\w[\w.\-]*)',
    re.MULTILINE,
)
TS_SPAWN_PATTERN = re.compile(
    r'spawn\s*\(\s*[\'"`](\w[\w.\-]*)',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExternalInvocation:
    """Represents a single detected external application invocation."""
    application_name: str       # e.g. "git", "ffmpeg", "curl"
    invocation_method: str      # "subprocess", "os.system", "shell_command", "api_call"
    source_file: str            # relative path to the source file
    line_number: int            # line where the invocation occurs
    context: str                # surrounding code snippet (up to 3 lines)
    arguments: List[str]        # detected arguments if available
    language: str               # "python", "rust", "typescript", "javascript", "unknown"


@dataclass
class DependencyMapping:
    """Aggregated dependency information for an external application."""
    application_name: str
    invocation_count: int
    invocations: List[ExternalInvocation]
    first_seen: str             # source file where first detected
    languages_used_from: List[str]  # which languages invoke this app


# ---------------------------------------------------------------------------
# DependencyMapper
# ---------------------------------------------------------------------------

class DependencyMapper:
    """
    Scans a codebase for external application invocations.

    Uses AST parsing for Python files and regex patterns for all languages.
    Parallel file processing is used for performance.
    """

    def __init__(self, codebase_path: Path) -> None:
        self.codebase_path = codebase_path
        self._mappings: Dict[str, DependencyMapping] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_codebase(self, max_workers: int = 4) -> Dict[str, DependencyMapping]:
        """
        Scan the entire codebase for external application invocations.

        Returns dict mapping application_name -> DependencyMapping.
        Uses parallel file processing for performance.
        """
        source_files = list(self._collect_source_files())
        all_invocations: List[ExternalInvocation] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.scan_file, f): f for f in source_files
            }
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    invocations = future.result()
                    all_invocations.extend(invocations)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error scanning %s: %s", file_path, exc)

        self._mappings = self._aggregate_invocations(all_invocations)
        return self._mappings

    def scan_file(self, file_path: Path) -> List[ExternalInvocation]:
        """Scan a single file for external invocations."""
        suffix = file_path.suffix.lower()
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return []

        relative_path = self._relative_path(file_path)

        if suffix in PYTHON_EXTENSIONS:
            return self._scan_python_file(content, relative_path)
        elif suffix in RUST_EXTENSIONS:
            return self._scan_rust_file(content, relative_path)
        elif suffix in TS_JS_EXTENSIONS:
            return self._scan_ts_js_file(content, relative_path, suffix)
        return []

    def get_dependency_summary(self) -> List[DependencyMapping]:
        """Return all mappings sorted by invocation_count descending."""
        return sorted(
            self._mappings.values(),
            key=lambda m: m.invocation_count,
            reverse=True,
        )

    def get_top_dependencies(self, n: int = 10) -> List[DependencyMapping]:
        """Return the top N most-used external dependencies."""
        return self.get_dependency_summary()[:n]

    # ------------------------------------------------------------------
    # Internal helpers — file collection
    # ------------------------------------------------------------------

    def _collect_source_files(self) -> List[Path]:
        """Walk the codebase and collect all scannable source files."""
        result: List[Path] = []
        try:
            for path in self.codebase_path.rglob("*"):
                # Skip directories in the skip list
                if any(skip in path.parts for skip in SKIP_DIRS):
                    continue
                if path.is_file() and path.suffix.lower() in ALL_EXTENSIONS:
                    result.append(path)
        except OSError as exc:
            logger.warning("Error walking codebase: %s", exc)
        return result

    def _relative_path(self, file_path: Path) -> str:
        """Return a relative path string from the codebase root."""
        try:
            return str(file_path.relative_to(self.codebase_path))
        except ValueError:
            return str(file_path)

    # ------------------------------------------------------------------
    # Internal helpers — Python scanning
    # ------------------------------------------------------------------

    def _scan_python_file(
        self, content: str, relative_path: str
    ) -> List[ExternalInvocation]:
        """Scan a Python file using AST + regex fallback."""
        invocations: List[ExternalInvocation] = []
        lines = content.splitlines()

        # Try AST-based detection first
        try:
            tree = ast.parse(content)
            invocations.extend(
                self._ast_scan_python(tree, lines, relative_path)
            )
        except SyntaxError:
            pass  # Fall through to regex

        # Regex-based detection (catches things AST might miss, e.g. shell=True strings)
        invocations.extend(
            self._regex_scan_python(content, lines, relative_path)
        )

        return self._deduplicate_invocations(invocations)

    def _ast_scan_python(
        self,
        tree: ast.AST,
        lines: List[str],
        relative_path: str,
    ) -> List[ExternalInvocation]:
        """Walk the AST to find subprocess / os.system / os.popen calls."""
        invocations: List[ExternalInvocation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            method_name, app_name, args_list = self._extract_python_call(node, func)
            if method_name is None or app_name is None:
                continue
            if app_name.lower() in SKIP_LIST:
                continue

            line_no = getattr(node, "lineno", 1)
            context = self._get_context(lines, line_no - 1)

            invocations.append(
                ExternalInvocation(
                    application_name=app_name.lower(),
                    invocation_method=method_name,
                    source_file=relative_path,
                    line_number=line_no,
                    context=context,
                    arguments=args_list,
                    language="python",
                )
            )

        return invocations

    def _extract_python_call(
        self, node: ast.Call, func: ast.expr
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Determine if a Call node is a subprocess/os.system/os.popen call.
        Returns (method_name, app_name, args_list) or (None, None, []).
        """
        # subprocess.run / subprocess.call / subprocess.check_output / subprocess.Popen
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr in ("run", "call", "check_output", "Popen")
        ):
            if node.args:
                first_arg = node.args[0]
                # subprocess.run(["git", ...])
                if isinstance(first_arg, ast.List) and first_arg.elts:
                    app_name = self._extract_string_value(first_arg.elts[0])
                    rest = [
                        self._extract_string_value(e)
                        for e in first_arg.elts[1:]
                        if self._extract_string_value(e)
                    ]
                    if app_name:
                        return "subprocess", app_name, rest
                # subprocess.run("git status", shell=True)
                elif isinstance(first_arg, (ast.Constant, ast.Str)):
                    cmd_str = self._extract_string_value(first_arg)
                    if cmd_str:
                        parts = cmd_str.split()
                        if parts:
                            return "subprocess", parts[0], parts[1:]

        # os.system("git clone ...")
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr == "system"
        ):
            if node.args:
                cmd_str = self._extract_string_value(node.args[0])
                if cmd_str:
                    parts = cmd_str.split()
                    if parts:
                        return "os.system", parts[0], parts[1:]

        # os.popen("ffmpeg ...")
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr == "popen"
        ):
            if node.args:
                cmd_str = self._extract_string_value(node.args[0])
                if cmd_str:
                    parts = cmd_str.split()
                    if parts:
                        return "os.popen", parts[0], parts[1:]

        return None, None, []

    @staticmethod
    def _extract_string_value(node: ast.expr) -> Optional[str]:
        """Extract a string value from an AST node (Constant or Str)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # Python < 3.8 compatibility
        if isinstance(node, ast.Str):  # type: ignore[attr-defined]
            return node.s  # type: ignore[attr-defined]
        return None

    def _regex_scan_python(
        self, content: str, lines: List[str], relative_path: str
    ) -> List[ExternalInvocation]:
        """Regex-based fallback for Python files."""
        invocations: List[ExternalInvocation] = []

        patterns = [
            (PYTHON_SUBPROCESS_PATTERN, "subprocess"),
            (PYTHON_OS_SYSTEM_PATTERN, "os.system"),
            (PYTHON_OS_POPEN_PATTERN, "os.popen"),
        ]

        for pattern, method in patterns:
            for match in pattern.finditer(content):
                app_name = match.group(1).lower()
                if app_name in SKIP_LIST:
                    continue
                line_no = content[: match.start()].count("\n") + 1
                context = self._get_context(lines, line_no - 1)
                invocations.append(
                    ExternalInvocation(
                        application_name=app_name,
                        invocation_method=method,
                        source_file=relative_path,
                        line_number=line_no,
                        context=context,
                        arguments=[],
                        language="python",
                    )
                )

        return invocations

    # ------------------------------------------------------------------
    # Internal helpers — Rust scanning
    # ------------------------------------------------------------------

    def _scan_rust_file(
        self, content: str, relative_path: str
    ) -> List[ExternalInvocation]:
        """Scan a Rust file using regex patterns."""
        invocations: List[ExternalInvocation] = []
        lines = content.splitlines()

        for match in RUST_COMMAND_PATTERN.finditer(content):
            app_name = match.group(1).lower()
            if app_name in SKIP_LIST:
                continue
            line_no = content[: match.start()].count("\n") + 1
            context = self._get_context(lines, line_no - 1)
            invocations.append(
                ExternalInvocation(
                    application_name=app_name,
                    invocation_method="shell_command",
                    source_file=relative_path,
                    line_number=line_no,
                    context=context,
                    arguments=[],
                    language="rust",
                )
            )

        return invocations

    # ------------------------------------------------------------------
    # Internal helpers — TypeScript/JavaScript scanning
    # ------------------------------------------------------------------

    def _scan_ts_js_file(
        self, content: str, relative_path: str, suffix: str
    ) -> List[ExternalInvocation]:
        """Scan a TypeScript/JavaScript file using regex patterns."""
        invocations: List[ExternalInvocation] = []
        lines = content.splitlines()
        language = "typescript" if suffix in {".ts", ".tsx"} else "javascript"

        patterns = [
            (TS_EXEC_PATTERN, "shell_command"),
            (TS_SPAWN_PATTERN, "shell_command"),
        ]

        for pattern, method in patterns:
            for match in pattern.finditer(content):
                app_name = match.group(1).lower()
                if app_name in SKIP_LIST:
                    continue
                line_no = content[: match.start()].count("\n") + 1
                context = self._get_context(lines, line_no - 1)
                invocations.append(
                    ExternalInvocation(
                        application_name=app_name,
                        invocation_method=method,
                        source_file=relative_path,
                        line_number=line_no,
                        context=context,
                        arguments=[],
                        language=language,
                    )
                )

        return invocations

    # ------------------------------------------------------------------
    # Internal helpers — aggregation
    # ------------------------------------------------------------------

    def _aggregate_invocations(
        self, invocations: List[ExternalInvocation]
    ) -> Dict[str, DependencyMapping]:
        """Aggregate a flat list of invocations into DependencyMapping objects."""
        mappings: Dict[str, DependencyMapping] = {}

        for inv in invocations:
            name = inv.application_name
            if name not in mappings:
                mappings[name] = DependencyMapping(
                    application_name=name,
                    invocation_count=0,
                    invocations=[],
                    first_seen=inv.source_file,
                    languages_used_from=[],
                )
            mapping = mappings[name]
            mapping.invocations.append(inv)
            mapping.invocation_count += 1
            if inv.language not in mapping.languages_used_from:
                mapping.languages_used_from.append(inv.language)

        return mappings

    @staticmethod
    def _deduplicate_invocations(
        invocations: List[ExternalInvocation],
    ) -> List[ExternalInvocation]:
        """Remove duplicate invocations (same app, method, file, line)."""
        seen: Set[Tuple[str, str, str, int]] = set()
        result: List[ExternalInvocation] = []
        for inv in invocations:
            key = (inv.application_name, inv.invocation_method, inv.source_file, inv.line_number)
            if key not in seen:
                seen.add(key)
                result.append(inv)
        return result

    @staticmethod
    def _get_context(lines: List[str], line_index: int, window: int = 1) -> str:
        """Return up to `window` lines before and after the target line."""
        start = max(0, line_index - window)
        end = min(len(lines), line_index + window + 1)
        return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# FunctionalityGap
# ---------------------------------------------------------------------------

@dataclass
class FunctionalityGap:
    """A functionality gap in Janus that could be filled by an external application."""
    gap_name: str               # e.g. "Video Processing", "Audio Conversion"
    description: str
    suggested_applications: List[str]  # apps that could fill this gap
    related_code_patterns: List[str]   # code patterns that indicate this gap


# ---------------------------------------------------------------------------
# PotentialApplicationIdentifier
# ---------------------------------------------------------------------------

class PotentialApplicationIdentifier:
    """
    Identifies beneficial external applications that Janus could leverage
    but doesn't currently use, and maps capabilities to functionality gaps.
    """

    # Known functionality gaps in AI systems like Janus
    KNOWN_GAPS: List[FunctionalityGap] = [
        FunctionalityGap(
            gap_name="Video Processing",
            description="Video encoding, decoding, and manipulation",
            suggested_applications=["ffmpeg", "vlc", "handbrake"],
            related_code_patterns=["video", "mp4", "encode", "decode", "frame", "codec"],
        ),
        FunctionalityGap(
            gap_name="Audio Processing",
            description="Audio conversion, synthesis, and analysis",
            suggested_applications=["ffmpeg", "sox", "audacity"],
            related_code_patterns=["audio", "wav", "mp3", "tts", "speech", "voice"],
        ),
        FunctionalityGap(
            gap_name="Image Processing",
            description="Image manipulation, conversion, and analysis",
            suggested_applications=["imagemagick", "ffmpeg", "gimp"],
            related_code_patterns=["image", "png", "jpg", "resize", "crop", "pixel"],
        ),
        FunctionalityGap(
            gap_name="Version Control",
            description="Source code version control and collaboration",
            suggested_applications=["git", "svn", "mercurial"],
            related_code_patterns=["commit", "branch", "merge", "repository", "clone"],
        ),
        FunctionalityGap(
            gap_name="Container Management",
            description="Application containerization and orchestration",
            suggested_applications=["docker", "podman", "kubectl"],
            related_code_patterns=["container", "image", "dockerfile", "deploy", "kubernetes"],
        ),
        FunctionalityGap(
            gap_name="Network Operations",
            description="HTTP requests, file downloads, and API calls",
            suggested_applications=["curl", "wget", "httpie"],
            related_code_patterns=["http", "https", "download", "request", "api", "endpoint"],
        ),
        FunctionalityGap(
            gap_name="Text Processing",
            description="Advanced text search, transformation, and analysis",
            suggested_applications=["grep", "sed", "awk", "ripgrep"],
            related_code_patterns=["search", "replace", "pattern", "regex", "parse", "extract"],
        ),
        FunctionalityGap(
            gap_name="Database Operations",
            description="Database management and query execution",
            suggested_applications=["sqlite3", "psql", "mysql"],
            related_code_patterns=["database", "sql", "query", "table", "schema", "migrate"],
        ),
        FunctionalityGap(
            gap_name="Build and Compilation",
            description="Code compilation and build automation",
            suggested_applications=["make", "cmake", "cargo", "gcc", "clang"],
            related_code_patterns=["compile", "build", "link", "binary", "executable", "makefile"],
        ),
        FunctionalityGap(
            gap_name="Package Management",
            description="Software package installation and management",
            suggested_applications=["pip", "npm", "cargo", "apt", "brew"],
            related_code_patterns=["install", "package", "dependency", "requirements", "module"],
        ),
    ]

    def __init__(self, dependency_mappings: Dict[str, DependencyMapping]) -> None:
        self.dependency_mappings = dependency_mappings
        self._current_apps: Set[str] = set(dependency_mappings.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_gaps(
        self, source_files_content: Dict[str, str]
    ) -> List[FunctionalityGap]:
        """
        Identify functionality gaps by scanning source file content for
        patterns that suggest a need for external tools.

        Returns gaps where the suggested apps are NOT already in use.
        """
        combined_content = "\n".join(source_files_content.values()).lower()
        gaps: List[FunctionalityGap] = []

        for gap in self.KNOWN_GAPS:
            # Check if any related pattern appears in the source
            pattern_found = any(
                pattern in combined_content
                for pattern in gap.related_code_patterns
            )
            if not pattern_found:
                continue

            # Only include the gap if at least one suggested app is NOT already used
            unused_suggestions = [
                app for app in gap.suggested_applications
                if app not in self._current_apps
            ]
            if unused_suggestions:
                # Return a gap with only the unused suggestions
                gaps.append(
                    FunctionalityGap(
                        gap_name=gap.gap_name,
                        description=gap.description,
                        suggested_applications=unused_suggestions,
                        related_code_patterns=gap.related_code_patterns,
                    )
                )

        return gaps

    def get_unused_beneficial_apps(self) -> List[str]:
        """
        Return a list of application names that would be beneficial
        but are not currently detected in the codebase.
        """
        all_suggested: Set[str] = set()
        for gap in self.KNOWN_GAPS:
            all_suggested.update(gap.suggested_applications)

        # Remove apps that are already in use and skip-listed builtins
        return sorted(all_suggested - self._current_apps - SKIP_LIST)

    def get_dependency_relationships(self) -> Dict[str, List[str]]:
        """
        Track which applications are often used together.

        Returns dict: app_name -> [related_app_names].
        Based on co-occurrence in the same source files.
        """
        # Build a map: source_file -> set of app names found there
        file_to_apps: Dict[str, Set[str]] = {}
        for app_name, mapping in self.dependency_mappings.items():
            for inv in mapping.invocations:
                file_to_apps.setdefault(inv.source_file, set()).add(app_name)

        # Build co-occurrence relationships
        relationships: Dict[str, Set[str]] = {
            app: set() for app in self.dependency_mappings
        }
        for apps_in_file in file_to_apps.values():
            apps_list = list(apps_in_file)
            for i, app_a in enumerate(apps_list):
                for app_b in apps_list[i + 1 :]:
                    relationships[app_a].add(app_b)
                    relationships[app_b].add(app_a)

        return {app: sorted(related) for app, related in relationships.items()}
