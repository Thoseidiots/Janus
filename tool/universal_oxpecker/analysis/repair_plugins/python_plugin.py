from __future__ import annotations

import ast
import copy
import re
from typing import Callable, List, Optional

try:
    from ..repair_models import CandidateDraft
    from .base import RepairPlugin
    from ...core.events import Issue
except ImportError:  # pragma: no cover - top-level package fallback
    from analysis.repair_models import CandidateDraft
    from analysis.repair_plugins.base import RepairPlugin
    from core.events import Issue


class PythonRepairPlugin(RepairPlugin):
    name = "python-ast"
    language = "python"

    def build_candidates(
        self,
        source_path: str,
        original_text: str,
        issue: Issue,
        engine,
    ) -> List[CandidateDraft]:
        message = (issue.message or "").lower()
        candidates: List[CandidateDraft] = []

        if issue.line and "bare 'except:'" in message:
            ast_text = _replace_bare_except_ast(original_text, issue.line)
            if ast_text and ast_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="AST patch: replace bare except with except Exception.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=ast_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="ast",
                    )
                )
            text_text = _replace_on_line(
                original_text,
                issue.line,
                lambda line: line.replace("except:", "except Exception:", 1),
            )
            if text_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="Text patch: replace bare except with except Exception.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=text_text,
                        issue_message=issue.message or "",
                        plugin_name="python-text",
                        patch_kind="text",
                    )
                )

        if issue.line and "use '==' instead of 'is'" in message:
            ast_text = _replace_identity_comparison_ast(original_text, issue.line)
            if ast_text and ast_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="AST patch: replace identity comparison with value comparison.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=ast_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="ast",
                    )
                )
            text_text = _replace_on_line(original_text, issue.line, _replace_identity_comparison_text)
            if text_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="Text patch: replace identity comparison with value comparison.",
                        complexity_tier=issue.complexity_tier or "Tier 1",
                        candidate_text=text_text,
                        issue_message=issue.message or "",
                        plugin_name="python-text",
                        patch_kind="text",
                    )
                )

        if issue.line and "mutable default argument" in message:
            ast_text = _replace_mutable_default_arg_ast(original_text, issue.line)
            if ast_text and ast_text != original_text:
                candidates.append(
                    CandidateDraft(
                        source_path=source_path,
                        line=issue.line,
                        description="AST patch: replace mutable default argument with None guard.",
                        complexity_tier=issue.complexity_tier or "Tier 2",
                        candidate_text=ast_text,
                        issue_message=issue.message or "",
                        plugin_name=self.name,
                        patch_kind="ast",
                    )
                )

        return _dedupe(candidates)

    def preflight_validate(
        self,
        candidate_text: str,
        source_path: str,
        issue: Optional[Issue] = None,
    ):
        try:
            ast.parse(candidate_text, filename=source_path)
            return True, "Python AST validation passed."
        except SyntaxError as exc:
            return False, f"Python AST validation failed: {exc.msg} (line {exc.lineno})."


class _BareExceptTransformer(ast.NodeTransformer):
    def __init__(self, target_line: int):
        self.target_line = target_line
        self.changed = False

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self.generic_visit(node)
        if node.lineno == self.target_line and node.type is None:
            new_node = copy.deepcopy(node)
            new_node.type = ast.Name(id="Exception", ctx=ast.Load())
            self.changed = True
            return ast.copy_location(new_node, node)
        return node


class _IdentityComparisonTransformer(ast.NodeTransformer):
    def __init__(self, target_line: int):
        self.target_line = target_line
        self.changed = False

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if node.lineno != self.target_line:
            return node
        if not any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return node

        updated_ops = []
        changed = False
        for index, op in enumerate(node.ops):
            comparator = node.comparators[index] if index < len(node.comparators) else None
            if isinstance(op, ast.Is) and _is_literal_value(comparator):
                updated_ops.append(ast.Eq())
                changed = True
            elif isinstance(op, ast.IsNot) and _is_literal_value(comparator):
                updated_ops.append(ast.NotEq())
                changed = True
            else:
                updated_ops.append(op)
        if changed:
            new_node = copy.deepcopy(node)
            new_node.ops = updated_ops
            self.changed = True
            return ast.copy_location(new_node, node)
        return node


class _MutableDefaultTransformer(ast.NodeTransformer):
    def __init__(self, target_line: int):
        self.target_line = target_line
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._visit_callable(node)

    def _visit_callable(self, node):
        self.generic_visit(node)
        if node.lineno != self.target_line:
            return node

        positional_args = list(node.args.args[-len(node.args.defaults):]) if node.args.defaults else []
        kwonly_args = list(node.args.kwonlyargs)
        kw_defaults = list(node.args.kw_defaults)
        prologue: List[ast.stmt] = []
        changed = False

        new_defaults = []
        for arg, default in zip(positional_args, node.args.defaults):
            if _is_mutable_literal(default):
                new_defaults.append(ast.Constant(value=None))
                prologue.append(_make_none_guard(arg.arg, default))
                changed = True
            else:
                new_defaults.append(default)

        new_kw_defaults = []
        for arg, default in zip(kwonly_args, kw_defaults):
            if default is not None and _is_mutable_literal(default):
                new_kw_defaults.append(ast.Constant(value=None))
                prologue.append(_make_none_guard(arg.arg, default))
                changed = True
            else:
                new_kw_defaults.append(default)

        if changed:
            new_node = copy.deepcopy(node)
            new_node.args.defaults = new_defaults
            new_node.args.kw_defaults = new_kw_defaults
            new_node.body = prologue + new_node.body
            self.changed = True
            return ast.copy_location(new_node, node)
        return node


def _replace_bare_except_ast(source: str, target_line: int) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    transformer = _BareExceptTransformer(target_line)
    updated = transformer.visit(tree)
    if not transformer.changed:
        return None
    ast.fix_missing_locations(updated)
    return ast.unparse(updated) + "\n"


def _replace_identity_comparison_ast(source: str, target_line: int) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    transformer = _IdentityComparisonTransformer(target_line)
    updated = transformer.visit(tree)
    if not transformer.changed:
        return None
    ast.fix_missing_locations(updated)
    return ast.unparse(updated) + "\n"


def _replace_mutable_default_arg_ast(source: str, target_line: int) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    transformer = _MutableDefaultTransformer(target_line)
    updated = transformer.visit(tree)
    if not transformer.changed:
        return None
    ast.fix_missing_locations(updated)
    return ast.unparse(updated) + "\n"


def _make_none_guard(name: str, default_node: ast.AST) -> ast.If:
    return ast.If(
        test=ast.Compare(
            left=ast.Name(id=name, ctx=ast.Load()),
            ops=[ast.Is()],
            comparators=[ast.Constant(value=None)],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_fresh_mutable_value(default_node),
            )
        ],
        orelse=[],
    )


def _fresh_mutable_value(default_node: ast.AST) -> ast.AST:
    if isinstance(default_node, ast.List):
        return ast.List(elts=[], ctx=ast.Load())
    if isinstance(default_node, ast.Dict):
        return ast.Dict(keys=[], values=[])
    if isinstance(default_node, ast.Set):
        return ast.Call(func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[])
    return ast.Constant(value=None)


def _is_mutable_literal(node: ast.AST) -> bool:
    return isinstance(node, (ast.List, ast.Dict, ast.Set))


def _is_literal_value(node: Optional[ast.AST]) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, str, float))
        and not isinstance(node.value, bool)
        and node.value is not None
    )


def _replace_on_line(text: str, line_no: int, transform: Callable[[str], str]) -> str:
    lines = text.splitlines(keepends=True)
    index = line_no - 1
    if index < 0 or index >= len(lines):
        return text
    updated = transform(lines[index])
    if updated == lines[index]:
        return text
    lines[index] = updated
    return "".join(lines)


def _replace_identity_comparison_text(line: str) -> str:
    line = re.sub(r"\bis\s+not\b", "!=", line)
    line = re.sub(r"\bis\b", "==", line)
    return line


def _dedupe(candidates: List[CandidateDraft]) -> List[CandidateDraft]:
    seen = set()
    unique: List[CandidateDraft] = []
    for candidate in candidates:
        key = (candidate.description, candidate.candidate_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
