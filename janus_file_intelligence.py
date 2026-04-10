"""
janus_file_intelligence.py
===========================
File system mastery for Janus.

Janus can now:
  - Understand any codebase dropped in front of it
  - Search across files by content, not just name
  - Summarize what a project does
  - Find bugs, TODOs, and patterns across files
  - Organize and restructure file collections
  - Answer questions about a codebase

No API keys. Pure Python stdlib + JanusBrain for comprehension.

Usage:
    from janus_file_intelligence import JanusFileIntelligence
    fi = JanusFileIntelligence()

    # Understand a project
    summary = fi.understand_project("/path/to/project")

    # Search across files
    results = fi.search("database connection", "/path/to/project")

    # Answer questions about code
    answer = fi.ask_about_file("What does this function do?", "main.py")
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── File record ───────────────────────────────────────────────────────────────

@dataclass
class FileRecord:
    path:       Path
    size_bytes: int
    extension:  str
    language:   str
    lines:      int
    content:    str          # first 4KB
    summary:    str          = ""
    functions:  List[str]    = field(default_factory=list)
    classes:    List[str]    = field(default_factory=list)
    imports:    List[str]    = field(default_factory=list)
    todos:      List[str]    = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path":      str(self.path),
            "language":  self.language,
            "lines":     self.lines,
            "size_kb":   round(self.size_bytes / 1024, 1),
            "functions": self.functions[:10],
            "classes":   self.classes[:10],
            "todos":     self.todos[:5],
            "summary":   self.summary,
        }


# ── Language detector ─────────────────────────────────────────────────────────

_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
    ".go": "go", ".java": "java", ".kt": "kotlin", ".cs": "csharp",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".sql": "sql", ".sh": "shell", ".html": "html",
    ".css": "css", ".scss": "css",
}

def _detect_language(path: Path) -> str:
    return _EXT_LANG.get(path.suffix.lower(), "text")


# ── Code analyzer ─────────────────────────────────────────────────────────────

class CodeAnalyzer:
    """Extracts structure from source files without running them."""

    def analyze(self, path: Path, content: str) -> FileRecord:
        lang = _detect_language(path)
        lines = content.count("\n") + 1
        record = FileRecord(
            path       = path,
            size_bytes = len(content.encode()),
            extension  = path.suffix.lower(),
            language   = lang,
            lines      = lines,
            content    = content[:4096],
        )

        if lang == "python":
            self._analyze_python(content, record)
        else:
            self._analyze_generic(content, record)

        return record

    def _analyze_python(self, content: str, record: FileRecord):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    record.functions.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    record.functions.append(f"async {node.name}")
                elif isinstance(node, ast.ClassDef):
                    record.classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            record.imports.append(alias.name)
                    else:
                        record.imports.append(node.module or "")
        except SyntaxError:
            self._analyze_generic(content, record)

        # TODOs
        record.todos = re.findall(r'#\s*(?:TODO|FIXME|HACK|XXX)[:\s]+(.+)', content)

    def _analyze_generic(self, content: str, record: FileRecord):
        # Functions via regex
        fn_patterns = [
            r'(?:def|function|func|fn)\s+(\w+)\s*\(',
            r'(\w+)\s*=\s*(?:async\s+)?(?:function|\()',
        ]
        for pat in fn_patterns:
            record.functions.extend(re.findall(pat, content))

        # Classes
        record.classes = re.findall(r'(?:class|struct|interface|type)\s+(\w+)', content)

        # TODOs
        record.todos = re.findall(r'(?://|#)\s*(?:TODO|FIXME|HACK)[:\s]+(.+)', content)


# ── Main file intelligence ────────────────────────────────────────────────────

class JanusFileIntelligence:
    """
    File system mastery — understands codebases, searches content,
    answers questions about files.
    """

    SKIP_DIRS  = {".git", "__pycache__", "node_modules", ".venv", "venv",
                  "dist", "build", ".next", "target", ".oxpecker_work"}
    SKIP_EXTS  = {".pyc", ".pyo", ".exe", ".dll", ".so", ".bin",
                  ".jpg", ".png", ".gif", ".mp4", ".mp3", ".zip",
                  ".tar", ".gz", ".lock", ".woff", ".ttf"}
    MAX_FILE_SIZE = 500_000  # 500KB

    def __init__(self):
        self._analyzer = CodeAnalyzer()
        self._index:   Dict[str, FileRecord] = {}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_directory(self, directory: str | Path) -> int:
        """Index all files in a directory. Returns file count."""
        root = Path(directory)
        count = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in self.SKIP_DIRS):
                continue
            if path.suffix.lower() in self.SKIP_EXTS:
                continue
            if path.stat().st_size > self.MAX_FILE_SIZE:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                record  = self._analyzer.analyze(path, content)
                self._index[str(path)] = record
                count += 1
            except Exception:
                pass
        return count

    # ── Understanding ─────────────────────────────────────────────────────────

    def understand_project(self, directory: str | Path) -> str:
        """
        Understand what a project does and how it's structured.
        Returns a plain-English summary.
        """
        root  = Path(directory)
        count = self.index_directory(root)

        if count == 0:
            return f"No readable files found in {directory}"

        # Build project overview
        by_lang: Dict[str, int] = {}
        all_functions: List[str] = []
        all_classes:   List[str] = []
        all_todos:     List[str] = []
        entry_points:  List[str] = []

        for rec in self._index.values():
            by_lang[rec.language] = by_lang.get(rec.language, 0) + 1
            all_functions.extend(rec.functions[:5])
            all_classes.extend(rec.classes[:5])
            all_todos.extend(rec.todos[:2])
            if rec.path.name in ("main.py", "index.py", "app.py", "server.py",
                                  "main.ts", "index.ts", "main.js", "index.js"):
                entry_points.append(str(rec.path.relative_to(root)))

        # Read README if present
        readme = ""
        for name in ("README.md", "README.txt", "README"):
            p = root / name
            if p.exists():
                readme = p.read_text(errors="replace")[:1000]
                break

        overview = (
            f"Project: {root.name}\n"
            f"Files: {count}\n"
            f"Languages: {', '.join(f'{lang}({n})' for lang, n in sorted(by_lang.items(), key=lambda x: -x[1])[:5])}\n"
            f"Entry points: {', '.join(entry_points) or 'not detected'}\n"
            f"Key classes: {', '.join(set(all_classes[:10]))}\n"
            f"Key functions: {', '.join(set(all_functions[:15]))}\n"
            f"Open TODOs: {len(all_todos)}\n"
        )
        if readme:
            overview += f"\nREADME:\n{readme}\n"

        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.summarize(overview)
        except Exception:
            return overview

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, directory: str | Path = ".",
               max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search across all files for content matching a query.
        Returns list of {file, line_number, line, context}.
        """
        root    = Path(directory)
        results = []
        words   = query.lower().split()

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in self.SKIP_DIRS):
                continue
            if path.suffix.lower() in self.SKIP_EXTS:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                for i, line in enumerate(lines):
                    if all(w in line.lower() for w in words):
                        results.append({
                            "file":    str(path.relative_to(root)),
                            "line":    i + 1,
                            "content": line.strip()[:120],
                            "context": "\n".join(lines[max(0,i-1):i+2]),
                        })
                        if len(results) >= max_results:
                            return results
            except Exception:
                pass
        return results

    # ── File comprehension ────────────────────────────────────────────────────

    def ask_about_file(self, question: str, file_path: str | Path) -> str:
        """Ask a question about a specific file."""
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Could not read file: {e}"

        try:
            from avus_brain import get_brain
            brain = get_brain()
            prompt = (
                f"File: {path.name}\n\n"
                f"Content:\n{content[:5000]}\n\n"
                f"Question: {question}"
            )
            return brain.ask(prompt, max_tokens=400)
        except Exception:
            return f"File has {content.count(chr(10))+1} lines, {len(content)} chars."

    def ask_about_project(self, question: str, directory: str | Path = ".") -> str:
        """Ask a question about an entire project."""
        root = Path(directory)
        if not self._index:
            self.index_directory(root)

        # Find most relevant files
        words    = question.lower().split()
        relevant = []
        for path_str, rec in self._index.items():
            score = sum(1 for w in words if w in rec.content.lower())
            if score > 0:
                relevant.append((score, rec))
        relevant.sort(key=lambda x: -x[0])

        context = ""
        for _, rec in relevant[:3]:
            context += f"\n--- {rec.path.name} ---\n{rec.content[:1500]}\n"

        if not context:
            return "No relevant files found for that question."

        try:
            from avus_brain import get_brain
            brain = get_brain()
            return brain.ask(
                f"Project: {root.name}\n\nRelevant files:\n{context}\n\nQuestion: {question}",
                max_tokens=400,
            )
        except Exception:
            return context[:500]

    # ── Organization ─────────────────────────────────────────────────────────

    def find_todos(self, directory: str | Path = ".") -> List[Dict]:
        """Find all TODO/FIXME comments across the project."""
        root = Path(directory)
        if not self._index:
            self.index_directory(root)
        todos = []
        for rec in self._index.values():
            for todo in rec.todos:
                todos.append({
                    "file": str(rec.path.relative_to(root)),
                    "todo": todo,
                })
        return todos

    def find_duplicates(self, directory: str | Path = ".") -> List[Dict]:
        """Find files with similar content (potential duplicates)."""
        root = Path(directory)
        if not self._index:
            self.index_directory(root)

        # Simple hash-based dedup
        from hashlib import md5
        hashes: Dict[str, List[str]] = {}
        for path_str, rec in self._index.items():
            h = md5(rec.content[:500].encode()).hexdigest()
            hashes.setdefault(h, []).append(path_str)

        return [
            {"files": paths, "likely_duplicate": True}
            for paths in hashes.values()
            if len(paths) > 1
        ]

    def get_stats(self, directory: str | Path = ".") -> Dict:
        """Get statistics about a directory."""
        root = Path(directory)
        if not self._index:
            self.index_directory(root)

        total_lines = sum(r.lines for r in self._index.values())
        total_size  = sum(r.size_bytes for r in self._index.values())
        by_lang: Dict[str, int] = {}
        for rec in self._index.values():
            by_lang[rec.language] = by_lang.get(rec.language, 0) + rec.lines

        return {
            "files":       len(self._index),
            "total_lines": total_lines,
            "total_size_kb": round(total_size / 1024, 1),
            "by_language": dict(sorted(by_lang.items(), key=lambda x: -x[1])),
            "todos":       sum(len(r.todos) for r in self._index.values()),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_fi: Optional[JanusFileIntelligence] = None

def get_file_intelligence() -> JanusFileIntelligence:
    global _fi
    if _fi is None:
        _fi = JanusFileIntelligence()
    return _fi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Janus File Intelligence")
    parser.add_argument("--understand", type=str, metavar="DIR")
    parser.add_argument("--search",     type=str, metavar="QUERY")
    parser.add_argument("--ask",        type=str, metavar="QUESTION")
    parser.add_argument("--file",       type=str, metavar="FILE")
    parser.add_argument("--stats",      type=str, metavar="DIR", default=".")
    parser.add_argument("--todos",      type=str, metavar="DIR")
    args = parser.parse_args()

    fi = JanusFileIntelligence()

    if args.understand:
        print(fi.understand_project(args.understand))
    elif args.search:
        results = fi.search(args.search)
        for r in results:
            print(f"{r['file']}:{r['line']}  {r['content']}")
    elif args.ask and args.file:
        print(fi.ask_about_file(args.ask, args.file))
    elif args.ask:
        print(fi.ask_about_project(args.ask))
    elif args.todos:
        todos = fi.find_todos(args.todos)
        for t in todos:
            print(f"{t['file']}: {t['todo']}")
    else:
        stats = fi.get_stats(args.stats)
        print(json.dumps(stats, indent=2))
