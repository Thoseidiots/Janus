"""
autonomous_ide.py
=================
Janus Autonomous IDE — write, analyse, test, execute, and deploy code
without external API dependencies.

Components:
    CodeProject       — dataclass representing a managed project
    ExecutionResult   — dataclass for sandbox execution output
    CodeAnalyzer      — static analysis (safety, complexity, imports)
    SandboxExecutor   — isolated subprocess execution (Python / JS)
    TestRunner        — auto-generates and runs basic test suites
    DeploymentManager — packages projects and creates run scripts
    JanusIDE          — main orchestrator
"""

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CodeProject:
    """Represents a code project managed by Janus."""
    project_id:      str
    name:            str
    description:     str
    language:        str
    created_at:      str
    modified_at:     str
    files:           Dict[str, str]       # filepath -> content
    dependencies:    List[str]
    entry_point:     str
    status:          str                  # draft | testing | ready | deployed
    test_results:    Optional[Dict]  = None
    deployment_info: Optional[Dict]  = None


@dataclass
class ExecutionResult:
    """Result of a sandbox code execution."""
    success:        bool
    output:         str
    errors:         str
    exit_code:      int
    execution_time: float
    memory_used:    Optional[int]  = None
    metadata:       Optional[Dict] = None


# ── Code analyser ─────────────────────────────────────────────────────────────

class CodeAnalyzer:
    """Analyses code for safety, quality, and dependencies."""

    DANGEROUS_TOKENS = {
        "os.system", "subprocess.Popen", "eval", "exec",
        "compile", "__import__",
    }

    def analyze_python(self, code: str) -> Dict[str, Any]:
        """Run static analysis on Python source code."""
        result: Dict[str, Any] = {
            "valid_syntax":       False,
            "safety_score":       0.0,
            "complexity_score":   0.0,
            "detected_imports":   [],
            "security_warnings":  [],
            "suggestions":        [],
            "estimated_resources": {},
        }

        try:
            tree = ast.parse(code)
            result["valid_syntax"] = True
        except SyntaxError as e:
            result["security_warnings"].append(f"SyntaxError: {e}")
            return result

        # Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["detected_imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result["detected_imports"].append(node.module)

        # Safety scoring
        safety = 100.0
        for token in self.DANGEROUS_TOKENS:
            if token in code:
                result["security_warnings"].append(f"Dangerous token: {token}")
                safety -= 20
        if "open(" in code:
            result["security_warnings"].append("File operations detected")
            safety -= 10
        if any(t in code.lower() for t in ("socket", "http", "requests", "urllib")):
            result["security_warnings"].append("Network operations detected")
            safety -= 15
        result["safety_score"] = max(0.0, safety)

        # Complexity
        n_funcs   = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        n_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        n_loops   = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
        result["complexity_score"] = min(10.0, (n_funcs * 2 + n_classes * 3 + n_loops) / 10)
        result["estimated_resources"] = {
            "functions": n_funcs,
            "classes":   n_classes,
            "loops":     n_loops,
            "estimated_memory_mb": 10 + n_classes * 5 + n_funcs * 2,
            "estimated_runtime":   "fast" if n_loops < 3 else "medium",
        }

        # Suggestions
        if n_funcs > 10:
            result["suggestions"].append("Consider splitting into multiple modules")
        if len(code.splitlines()) > 500:
            result["suggestions"].append("Large file — consider splitting")

        return result

    def analyze_javascript(self, code: str) -> Dict[str, Any]:
        """Basic safety analysis for JavaScript source code."""
        result: Dict[str, Any] = {
            "valid_syntax":      True,
            "safety_score":      100.0,
            "detected_imports":  [],
            "security_warnings": [],
            "suggestions":       [],
        }
        for pattern in ("eval(", "innerHTML", "document.write", "Function("):
            if pattern in code:
                result["security_warnings"].append(f"Dangerous pattern: {pattern}")
                result["safety_score"] = max(0.0, result["safety_score"] - 20)

        result["detected_imports"] = re.findall(
            r'(?:require|import)\s*\([\'"]([^\'"]+)[\'"]\)', code
        )
        return result


# ── Sandbox executor ──────────────────────────────────────────────────────────

class SandboxExecutor:
    """Executes code in isolated temporary directories."""

    def __init__(self, workspace_dir: str = "/tmp/janus_sandbox") -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def _sandbox(self, project_id: str) -> Path:
        p = self.workspace_dir / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def execute_python(
        self, code: str, project_id: str, timeout: int = 30
    ) -> ExecutionResult:
        """Execute Python code in a sandbox subprocess."""
        sandbox = self._sandbox(project_id)
        code_file = sandbox / "main.py"
        code_file.write_text(code, encoding="utf-8")
        t0 = datetime.now()
        try:
            r = subprocess.run(
                ["python3", str(code_file)],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(sandbox),
            )
            elapsed = (datetime.now() - t0).total_seconds()
            return ExecutionResult(
                success=r.returncode == 0,
                output=r.stdout, errors=r.stderr,
                exit_code=r.returncode, execution_time=elapsed,
                metadata={"sandbox": str(sandbox)},
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, output="",
                errors=f"Timeout after {timeout}s",
                exit_code=-1, execution_time=float(timeout),
            )
        except Exception as e:
            return ExecutionResult(
                success=False, output="", errors=str(e),
                exit_code=-1, execution_time=0.0,
            )

    def execute_javascript(
        self, code: str, project_id: str, timeout: int = 30
    ) -> ExecutionResult:
        """Execute JavaScript code via Node.js in a sandbox subprocess."""
        sandbox = self._sandbox(project_id)
        code_file = sandbox / "main.js"
        code_file.write_text(code, encoding="utf-8")
        t0 = datetime.now()
        try:
            r = subprocess.run(
                ["node", str(code_file)],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(sandbox),
            )
            elapsed = (datetime.now() - t0).total_seconds()
            return ExecutionResult(
                success=r.returncode == 0,
                output=r.stdout, errors=r.stderr,
                exit_code=r.returncode, execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, output="",
                errors=f"Timeout after {timeout}s",
                exit_code=-1, execution_time=float(timeout),
            )
        except Exception as e:
            return ExecutionResult(
                success=False, output="", errors=str(e),
                exit_code=-1, execution_time=0.0,
            )

    def cleanup(self, project_id: str) -> None:
        """Remove the sandbox directory for a project."""
        sandbox = self.workspace_dir / project_id
        if sandbox.exists():
            shutil.rmtree(sandbox)


# ── Test runner ───────────────────────────────────────────────────────────────

class TestRunner:
    """Auto-generates and runs basic test suites."""

    _EXT = {"python": "py", "javascript": "js"}

    def run_tests(self, project: CodeProject, executor: SandboxExecutor) -> Dict:
        """Generate and run tests for a project."""
        if project.language == "python":
            test_code = self._python_tests()
            result = executor.execute_python(test_code, project.project_id)
        elif project.language == "javascript":
            test_code = self._js_tests()
            result = executor.execute_javascript(test_code, project.project_id)
        else:
            return {"success": False, "message": f"Unsupported language: {project.language}"}

        return {
            "success":        result.success,
            "output":         result.output,
            "errors":         result.errors,
            "execution_time": result.execution_time,
        }

    def _python_tests(self) -> str:
        return (
            "import unittest\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "try:\n"
            "    import main\n"
            "except ImportError:\n"
            "    main = None\n"
            "\n"
            "class BasicTests(unittest.TestCase):\n"
            "    def test_import(self):\n"
            "        self.assertIsNotNone(main)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

    def _js_tests(self) -> str:
        return (
            "const assert = require('assert');\n"
            "try {\n"
            "    require('./main.js');\n"
            "    assert.ok(true);\n"
            "} catch(e) {\n"
            "    assert.fail('Import failed: ' + e.message);\n"
            "}\n"
        )


# ── Deployment manager ────────────────────────────────────────────────────────

class DeploymentManager:
    """Packages projects and creates run scripts."""

    def __init__(self, deployment_dir: str = "/tmp/janus_deployments") -> None:
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

    def package_project(self, project: CodeProject) -> Tuple[bool, str]:
        """Write all project files to a deployment directory."""
        pkg_dir = self.deployment_dir / project.project_id
        pkg_dir.mkdir(parents=True, exist_ok=True)
        try:
            for filepath, content in project.files.items():
                dest = pkg_dir / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            readme = (
                f"# {project.name}\n\n{project.description}\n\n"
                f"- Language: {project.language}\n"
                f"- Created: {project.created_at}\n"
                f"- Entry: {project.entry_point}\n"
            )
            (pkg_dir / "README.md").write_text(readme, encoding="utf-8")

            manifest = {
                "project":      asdict(project),
                "packaged_at":  datetime.now().isoformat(),
                "version":      "1.0.0",
            }
            (pkg_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return True, str(pkg_dir)
        except Exception as e:
            return False, str(e)

    def create_run_script(self, project: CodeProject) -> Tuple[bool, str]:
        """Create a shell run script for Python projects."""
        if project.language != "python":
            return False, "Run script only supported for Python"
        ok, pkg_path = self.package_project(project)
        if not ok:
            return False, pkg_path
        script = Path(pkg_path) / "run.sh"
        script.write_text(
            f'#!/bin/bash\ncd "$(dirname "$0")"\npython3 {project.entry_point}\n',
            encoding="utf-8",
        )
        os.chmod(script, 0o755)
        return True, str(script)


# ── JanusIDE ──────────────────────────────────────────────────────────────────

class JanusIDE:
    """Main IDE orchestrator for autonomous code management."""

    def __init__(self, workspace_dir: str = "/tmp/janus_ide") -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._projects_file = self.workspace_dir / "projects.json"
        self.projects: Dict[str, CodeProject] = self._load_projects()
        self.analyzer   = CodeAnalyzer()
        self.executor   = SandboxExecutor()
        self.test_runner = TestRunner()
        self.deployer   = DeploymentManager()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_projects(self) -> Dict[str, CodeProject]:
        if self._projects_file.exists():
            try:
                data = json.loads(self._projects_file.read_text(encoding="utf-8"))
                return {pid: CodeProject(**p) for pid, p in data.items()}
            except Exception:
                pass
        return {}

    def _save_projects(self) -> None:
        self._projects_file.write_text(
            json.dumps({pid: asdict(p) for pid, p in self.projects.items()}, indent=2),
            encoding="utf-8",
        )

    # ── project management ────────────────────────────────────────────────────

    def create_project(
        self, name: str, description: str, language: str = "python"
    ) -> str:
        """Create a new project and return its ID."""
        project_id = hashlib.sha256(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        ext = "py" if language == "python" else "js"
        project = CodeProject(
            project_id=project_id,
            name=name,
            description=description,
            language=language,
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
            files={},
            dependencies=[],
            entry_point=f"main.{ext}",
            status="draft",
        )
        self.projects[project_id] = project
        self._save_projects()
        return project_id

    def add_file(self, project_id: str, filepath: str, content: str) -> None:
        """Add or update a file in a project."""
        project = self._get(project_id)
        project.files[filepath] = content
        project.modified_at = datetime.now().isoformat()
        self._save_projects()

    def analyze_project(self, project_id: str) -> Dict:
        """Run static analysis on all project files."""
        project = self._get(project_id)
        analyses = {}
        for fp, content in project.files.items():
            if project.language == "python":
                analyses[fp] = self.analyzer.analyze_python(content)
            elif project.language == "javascript":
                analyses[fp] = self.analyzer.analyze_javascript(content)
        scores = [a["safety_score"] for a in analyses.values()]
        return {
            "files_analyzed":      len(analyses),
            "total_warnings":      sum(len(a["security_warnings"]) for a in analyses.values()),
            "average_safety_score": sum(scores) / max(len(scores), 1),
            "file_analyses":       analyses,
        }

    def execute_project(self, project_id: str, timeout: int = 30) -> ExecutionResult:
        """Execute the project entry point in a sandbox."""
        project = self._get(project_id)
        code = project.files.get(project.entry_point, "")
        if not code:
            return ExecutionResult(
                success=False, output="", errors="Entry point not found",
                exit_code=-1, execution_time=0.0,
            )
        if project.language == "python":
            return self.executor.execute_python(code, project_id, timeout)
        if project.language == "javascript":
            return self.executor.execute_javascript(code, project_id, timeout)
        return ExecutionResult(
            success=False, output="",
            errors=f"Unsupported language: {project.language}",
            exit_code=-1, execution_time=0.0,
        )

    def test_project(self, project_id: str) -> Dict:
        """Run auto-generated tests on the project."""
        project = self._get(project_id)
        results = self.test_runner.run_tests(project, self.executor)
        project.test_results = results
        project.status = "tested" if results["success"] else "failed"
        self._save_projects()
        return results

    def deploy_project(self, project_id: str) -> Dict:
        """Package and deploy the project."""
        project = self._get(project_id)
        ok, pkg_path = self.deployer.package_project(project)
        if not ok:
            return {"success": False, "error": pkg_path}
        script_ok, script_path = self.deployer.create_run_script(project)
        info = {
            "success":         True,
            "package_path":    pkg_path,
            "run_script":      script_path if script_ok else None,
            "deployed_at":     datetime.now().isoformat(),
        }
        project.deployment_info = info
        project.status = "deployed"
        self._save_projects()
        return info

    def get_status(self, project_id: str) -> Dict:
        """Return a status snapshot for a project."""
        project = self._get(project_id)
        return {
            "project":      asdict(project),
            "file_count":   len(project.files),
            "total_lines":  sum(len(c.splitlines()) for c in project.files.values()),
            "has_tests":    project.test_results is not None,
            "is_deployed":  project.deployment_info is not None,
        }

    def list_projects(self) -> List[Dict]:
        """List all projects with summary info."""
        return [
            {
                "project_id": pid,
                "name":       p.name,
                "language":   p.language,
                "status":     p.status,
                "created_at": p.created_at,
            }
            for pid, p in self.projects.items()
        ]

    def _get(self, project_id: str) -> CodeProject:
        if project_id not in self.projects:
            raise ValueError(f"Project '{project_id}' not found")
        return self.projects[project_id]
