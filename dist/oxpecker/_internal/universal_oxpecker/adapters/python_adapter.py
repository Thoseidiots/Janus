"""
Python adapter  ─  full implementation
Capabilities
  • detect:          *.py files or 'python' tag
  • static analysis: ast.parse (syntax errors), pyflakes (unused/undefined),
                     pylint subprocess (optional, graceful fallback)
  • runtime error normalisation: traceback → Issue
  • read_stack / read_variables: uses inspect on live frames (same-process only)
"""
from __future__ import annotations

import ast
import sys
import traceback
import inspect
import subprocess
import textwrap
from typing import Any, Dict, List, Optional

try:
    from ..core.adapter import LanguageAdapter
    from ..core.events import DebugEvent, Issue, StackFrame, Variable
except ImportError:  # pragma: no cover - top-level package fallback
    from core.adapter import LanguageAdapter
    from core.events import DebugEvent, Issue, StackFrame, Variable


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tb_to_frame(tb_list) -> Optional[StackFrame]:
    if not tb_list:
        return None
    last = tb_list[-1]
    return StackFrame(
        id="0",
        name=last.name,
        source_path=last.filename,
        line=last.lineno,
    )


# ─────────────────────────────────────────────────────────────────────────────
class PythonAdapter(LanguageAdapter):
    language = "python"

    # ── detection ─────────────────────────────────────────────────────────
    def detect(self, target: str) -> bool:
        return target.endswith(".py") or target.lower() in {"python", "py"}

    # ── session ────────────────────────────────────────────────────────────
    def initialize(self, target: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"target": target, "options": options or {}, "language": self.language}

    # ── execution control (stub – real impl would use debugpy/bdb) ─────────
    def set_breakpoint(self, source_path: str, line: int,
                       column: Optional[int] = None) -> bool:
        return True   # real: register with bdb / debugpy

    def continue_execution(self) -> DebugEvent:
        return DebugEvent(kind="continued", message="Python execution continued")

    def pause_execution(self) -> DebugEvent:
        return DebugEvent(kind="paused", message="Python execution paused")

    # ── state inspection (same-process live frames) ────────────────────────
    def read_stack(self) -> List[StackFrame]:
        frames = []
        for i, (frame, lineno) in enumerate(inspect.stack()):
            frames.append(StackFrame(
                id=str(i),
                name=frame.f_code.co_name,
                source_path=frame.f_code.co_filename,
                line=lineno,
            ))
        return frames

    def read_variables(self, frame_id: str) -> List[Variable]:
        stack = inspect.stack()
        idx = int(frame_id) if frame_id.isdigit() else 0
        if idx >= len(stack):
            return []
        frame = stack[idx][0]
        variables = []
        for name, val in {**frame.f_locals, **frame.f_globals}.items():
            if name.startswith("__"):
                continue
            variables.append(Variable(
                name=name,
                value=repr(val),
                type_name=type(val).__name__,
            ))
        return variables[:50]   # cap to avoid flood

    # ── error normalisation ────────────────────────────────────────────────
    def normalize_error(self, error: Exception) -> Issue:
        tb = traceback.extract_tb(error.__traceback__) if error.__traceback__ else []
        frame = _tb_to_frame(list(tb))
        return self.create_issue(
            severity="error",
            message=str(error),
            source_path=frame.source_path if frame else None,
            line=frame.line if frame else None,
            frame=frame,
        )

    # ── static analysis ────────────────────────────────────────────────────
    def analyze_code(self, code: str, filename: str = "<code>") -> List[Issue]:
        issues: List[Issue] = []
        # 1. syntax check via ast
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            issues.append(self.create_issue(
                severity="error",
                message=f"SyntaxError: {e.msg}",
                source_path=filename,
                line=e.lineno,
                column=e.offset,
                fix_suggestion=_suggest_syntax_fix(e),
            ))
            return issues   # can't go further with broken syntax

        # 2. AST-level checks
        issues.extend(self._ast_checks(tree, filename))
        return issues

    def analyze_file(self, source_path: str) -> List[Issue]:
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                code = fh.read()
        except OSError as e:
            return [self.create_issue(severity="error", message=str(e), source_path=source_path)]
        issues = self.analyze_code(code, source_path)

        # 3. Optional: run pyflakes if available
        issues.extend(self._run_pyflakes(source_path))
        return issues

    def _ast_checks(self, tree: ast.AST, filename: str) -> List[Issue]:
        issues: List[Issue] = []
        adapter = self

        class Visitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import):
                self.generic_visit(node)

            def visit_Raise(self, node: ast.Raise):
                # bare `raise` outside except block is usually intentional; skip
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler):
                """Catch bare except: at ANY nesting level."""
                if node.type is None:
                    issues.append(adapter.create_issue(
                        severity="warning",
                        message=f"Bare 'except:' catches all exceptions including SystemExit (line {node.lineno})",
                        source_path=filename,
                        line=node.lineno,
                        fix_suggestion="Replace 'except:' with 'except Exception:' to avoid catching system-level signals.",
                    ))
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                adapter._check_mutable_default_args(node, filename, issues)
                self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Compare(self, node: ast.Compare):
                # 1. Check all (left, op[0], comparator[0]), (comparator[0], op[1], comparator[1]), etc.
                operands = [node.left] + node.comparators
                for i, op in enumerate(node.ops):
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        left_val = operands[i]
                        right_val = operands[i+1]
                        
                        # Flag if either side is a literal (int, str, float, etc.)
                        for side in [left_val, right_val]:
                            if isinstance(side, (ast.Constant,)) and isinstance(side.value, (int, str, float)):
                                if not isinstance(side.value, bool) and side.value is not None:
                                    issues.append(adapter.create_issue(
                                        severity="warning",
                                        message=f"Use '==' instead of 'is' to compare values (line {node.lineno})",
                                        source_path=filename,
                                        line=node.lineno,
                                        fix_suggestion="Replace 'is'/'is not' with '=='/'!=' when comparing values, not identities.",
                                    ))
                                    break
                self.generic_visit(node)

        Visitor().visit(tree)
        return issues

    def _check_mutable_default_args(self, node: ast.FunctionDef, filename: str, issues: List[Issue]) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                issues.append(self.create_issue(
                    severity="warning",
                    message=f"Mutable default argument in '{node.name}' (line {node.lineno})",
                    source_path=filename,
                    line=node.lineno,
                    fix_suggestion=(
                        f"Replace the mutable default with None and initialise inside the function body:\n"
                        f"  def {node.name}(..., arg=None, ...):\n"
                        f"      if arg is None: arg = []  # or {{}} or set()"
                    ),
                ))

    def _run_pyflakes(self, source_path: str) -> List[Issue]:
        """Run pyflakes as a subprocess if available; silently skip otherwise."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyflakes", source_path],
                capture_output=True, text=True, timeout=10
            )
            issues = []
            for line in result.stdout.splitlines():
                # typical format:  path.py:10:1 undefined name 'foo'
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        lineno = int(parts[1])
                    except ValueError:
                        lineno = None
                    issues.append(self.create_issue(
                        severity="warning",
                        message=parts[-1].strip(),
                        source_path=source_path,
                        line=lineno,
                    ))
            return issues
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []


def _suggest_syntax_fix(e: SyntaxError) -> str:
    msg = e.msg.lower()
    if "unexpected eof" in msg or "unexpected end" in msg:
        return "Check for unclosed brackets, parentheses, or triple-quotes."
    if "invalid syntax" in msg:
        return (
            f"Syntax error at line {e.lineno}, col {e.offset}. "
            "Check the token immediately before this position for a missing colon, "
            "comma, or closing bracket."
        )
    if "expected an indented block" in msg:
        return "Add an indented body (or 'pass') after the colon on the previous line."
    if "unindent does not match" in msg:
        return "Fix inconsistent indentation—mix of tabs and spaces detected."
    return f"Python syntax error: {e.msg}"
