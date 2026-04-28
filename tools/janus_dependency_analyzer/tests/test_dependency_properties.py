"""
Property-based tests for the Dependency Mapper.

This module implements property-based tests using Hypothesis to verify
universal correctness properties of the DependencyMapper and
PotentialApplicationIdentifier components.

Feature: janus-dependency-analyzer
"""

import tempfile
import pytest
from pathlib import Path
from typing import Dict, List

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from ..dependency.mapper import (
    DependencyMapper,
    DependencyMapping,
    ExternalInvocation,
    FunctionalityGap,
    PotentialApplicationIdentifier,
    VALID_INVOCATION_METHODS,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Simple identifier-like names (tool names)
tool_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=2,
    max_size=20,
).filter(lambda s: s[0].isalpha())

# Strategy for a count of tool occurrences (1..5)
occurrence_count_strategy = st.integers(min_value=1, max_value=5)

# Skip list for filtering
SKIP_SET = {"python", "python3", "pip", "pip3", "sys", "os",
            "self", "cls", "true", "false", "null", "none"}


# ---------------------------------------------------------------------------
# Property 5: Comprehensive Dependency Detection
# Feature: janus-dependency-analyzer, Property 5: Comprehensive Dependency Detection
# Validates: Requirements 3.1, 3.2, 3.3
# ---------------------------------------------------------------------------

class TestComprehensiveDependencyDetection:
    """
    **Validates: Requirements 3.1, 3.2, 3.3**

    Property 5: Comprehensive Dependency Detection
    For any Janus codebase, the Dependency_Mapper SHALL identify all external
    application invocations (subprocess calls, system commands, API calls) and
    record their context and frequency of usage.
    """

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_subprocess_calls_detected_in_python(self, tool: str) -> None:
        """
        Given Python source code containing subprocess.run(["<tool>", ...]),
        scan_file detects the tool as an invocation.
        """
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            py_file = tmp_path / "test_subprocess.py"
            py_file.write_text(
                f'import subprocess\nsubprocess.run(["{tool}", "--version"])\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            invocations = mapper.scan_file(py_file)

        detected_names = [inv.application_name for inv in invocations]
        assert tool in detected_names, (
            f"Expected '{tool}' to be detected in subprocess.run call, "
            f"but got: {detected_names}"
        )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_os_system_calls_detected(self, tool: str) -> None:
        """
        Given Python source with os.system("<tool> ..."), the tool is detected.
        """
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            py_file = tmp_path / "test_os_system.py"
            py_file.write_text(
                f'import os\nos.system("{tool} --help")\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            invocations = mapper.scan_file(py_file)

        detected_names = [inv.application_name for inv in invocations]
        assert tool in detected_names, (
            f"Expected '{tool}' to be detected in os.system call, "
            f"but got: {detected_names}"
        )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_rust_command_new_detected(self, tool: str) -> None:
        """
        Given Rust source with Command::new("<tool>"), the tool is detected.
        """
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rs_file = tmp_path / "test_command.rs"
            rs_file.write_text(
                f'use std::process::Command;\nCommand::new("{tool}").arg("--version").output();\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            invocations = mapper.scan_file(rs_file)

        detected_names = [inv.application_name for inv in invocations]
        assert tool in detected_names, (
            f"Expected '{tool}' to be detected in Rust Command::new call, "
            f"but got: {detected_names}"
        )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_typescript_exec_detected(self, tool: str) -> None:
        """
        Given TypeScript source with exec("<tool> install"), the tool is detected.
        """
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ts_file = tmp_path / "test_exec.ts"
            ts_file.write_text(
                f'import {{ exec }} from "child_process";\nexec("{tool} install");\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            invocations = mapper.scan_file(ts_file)

        detected_names = [inv.application_name for inv in invocations]
        assert tool in detected_names, (
            f"Expected '{tool}' to be detected in TypeScript exec call, "
            f"but got: {detected_names}"
        )

    @given(
        tool=tool_name_strategy,
        n=occurrence_count_strategy,
    )
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_invocation_count_matches_occurrences(self, tool: str, n: int) -> None:
        """
        If a tool appears N times in source code, the DependencyMapping has
        invocation_count >= N.
        """
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            lines = [f'subprocess.run(["{tool}", "--step{i}"])' for i in range(n)]
            py_file = tmp_path / "test_count.py"
            py_file.write_text(
                "import subprocess\n" + "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            mappings = mapper.scan_codebase()

        assert tool in mappings, (
            f"Expected '{tool}' to appear in dependency mappings after {n} occurrences"
        )
        assert mappings[tool].invocation_count >= n, (
            f"Expected invocation_count >= {n} for '{tool}', "
            f"got {mappings[tool].invocation_count}"
        )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_all_invocations_have_source_file(self, tool: str) -> None:
        """Every ExternalInvocation has a non-empty source_file."""
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            py_file = tmp_path / "test_source.py"
            py_file.write_text(
                f'import subprocess\nsubprocess.run(["{tool}"])\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            mappings = mapper.scan_codebase()

        for app_name, mapping in mappings.items():
            for inv in mapping.invocations:
                assert inv.source_file, (
                    f"Invocation of '{app_name}' has an empty source_file"
                )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_all_invocations_have_valid_method(self, tool: str) -> None:
        """Every invocation_method is one of the known valid methods."""
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            py_file = tmp_path / "test_method.py"
            py_file.write_text(
                f'import subprocess\nimport os\n'
                f'subprocess.run(["{tool}"])\n'
                f'os.system("{tool} --help")\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            mappings = mapper.scan_codebase()

        for app_name, mapping in mappings.items():
            for inv in mapping.invocations:
                assert inv.invocation_method in VALID_INVOCATION_METHODS, (
                    f"Invocation of '{app_name}' has unknown method: "
                    f"'{inv.invocation_method}'. "
                    f"Valid methods: {VALID_INVOCATION_METHODS}"
                )

    @given(tool=tool_name_strategy)
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_scan_result_is_dict_of_dependency_mappings(self, tool: str) -> None:
        """scan_codebase() returns a dict where all values are DependencyMapping instances."""
        assume(tool not in SKIP_SET)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            py_file = tmp_path / "test_dict.py"
            py_file.write_text(
                f'import subprocess\nsubprocess.run(["{tool}"])\n',
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            result = mapper.scan_codebase()

        assert isinstance(result, dict), (
            f"scan_codebase() should return a dict, got {type(result)}"
        )
        for key, value in result.items():
            assert isinstance(key, str), (
                f"Dict key should be str, got {type(key)}"
            )
            assert isinstance(value, DependencyMapping), (
                f"Dict value for '{key}' should be DependencyMapping, got {type(value)}"
            )


# ---------------------------------------------------------------------------
# Property 6: Intelligent Dependency Analysis
# Feature: janus-dependency-analyzer, Property 6: Intelligent Dependency Analysis
# Validates: Requirements 3.4, 3.5, 3.6
# ---------------------------------------------------------------------------

class TestIntelligentDependencyAnalysis:
    """
    **Validates: Requirements 3.4, 3.5, 3.6**

    Property 6: Intelligent Dependency Analysis
    For any system with external applications, the Dependency_Mapper SHALL
    identify potentially beneficial unused applications, map capabilities to
    Janus functionality gaps, and track dependency relationships between
    applications.
    """

    @given(
        extra_content=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz _\n",
            min_size=0,
            max_size=50,
        )
    )
    @settings(max_examples=25)
    def test_gaps_identified_from_video_patterns(self, extra_content: str) -> None:
        """
        Source code containing "video", "mp4", "encode" triggers the
        Video Processing gap.
        """
        # No current apps that cover video processing
        dependency_mappings: Dict[str, DependencyMapping] = {}
        identifier = PotentialApplicationIdentifier(dependency_mappings)

        # Source content that clearly contains video-related patterns
        source_content = f"video mp4 encode {extra_content}"
        gaps = identifier.identify_gaps({"main.py": source_content})

        gap_names = [g.gap_name for g in gaps]
        assert "Video Processing" in gap_names, (
            f"Expected 'Video Processing' gap to be identified when source "
            f"contains video patterns, but got gaps: {gap_names}"
        )

    @given(
        current_apps=st.lists(
            tool_name_strategy,
            min_size=0,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=25)
    def test_unused_apps_not_in_current_deps(self, current_apps: List[str]) -> None:
        """
        get_unused_beneficial_apps() never returns apps that are already in
        dependency_mappings.
        """
        # Build fake dependency mappings for the current apps
        dependency_mappings: Dict[str, DependencyMapping] = {
            app: DependencyMapping(
                application_name=app,
                invocation_count=1,
                invocations=[],
                first_seen="fake_file.py",
                languages_used_from=["python"],
            )
            for app in current_apps
        }

        identifier = PotentialApplicationIdentifier(dependency_mappings)
        unused = identifier.get_unused_beneficial_apps()

        current_set = set(current_apps)
        for app in unused:
            assert app not in current_set, (
                f"get_unused_beneficial_apps() returned '{app}' which is already "
                f"in the current dependency mappings"
            )

    @given(
        apps=st.lists(
            tool_name_strategy,
            min_size=2,
            max_size=6,
            unique=True,
        )
    )
    @settings(
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_dependency_relationships_are_symmetric_or_empty(
        self, apps: List[str]
    ) -> None:
        """
        If app A is related to app B, then B is related to A
        (or neither has relationships).
        """
        valid_apps = [a for a in apps if a not in SKIP_SET]
        assume(len(valid_apps) >= 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a Python file that uses all apps together (co-occurrence)
            lines = [f'subprocess.run(["{app}"])' for app in valid_apps]
            py_file = tmp_path / "test_cooccurrence.py"
            py_file.write_text(
                "import subprocess\n" + "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            mapper = DependencyMapper(tmp_path)
            mappings = mapper.scan_codebase()

        identifier = PotentialApplicationIdentifier(mappings)
        relationships = identifier.get_dependency_relationships()

        # Check symmetry: if A -> B then B -> A
        for app_a, related in relationships.items():
            for app_b in related:
                assert app_b in relationships, (
                    f"App '{app_b}' is in relationships of '{app_a}' "
                    f"but has no entry in the relationships dict"
                )
                assert app_a in relationships[app_b], (
                    f"Relationship is not symmetric: '{app_a}' -> '{app_b}' "
                    f"but '{app_b}' does not list '{app_a}'"
                )

    def test_gaps_suggest_valid_applications(self) -> None:
        """Every gap's suggested_applications list is non-empty."""
        dependency_mappings: Dict[str, DependencyMapping] = {}
        identifier = PotentialApplicationIdentifier(dependency_mappings)

        for gap in identifier.KNOWN_GAPS:
            assert len(gap.suggested_applications) > 0, (
                f"Gap '{gap.gap_name}' has an empty suggested_applications list"
            )

    @given(
        content=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _\n./",
            min_size=0,
            max_size=200,
        )
    )
    @settings(max_examples=25)
    def test_identify_gaps_returns_list(self, content: str) -> None:
        """identify_gaps() always returns a list (never raises)."""
        dependency_mappings: Dict[str, DependencyMapping] = {}
        identifier = PotentialApplicationIdentifier(dependency_mappings)

        try:
            result = identifier.identify_gaps({"source.py": content})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"identify_gaps() raised an unexpected exception: {exc!r}"
            )

        assert isinstance(result, list), (
            f"identify_gaps() should return a list, got {type(result)}"
        )
