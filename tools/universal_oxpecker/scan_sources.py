"""
Targeted source-file scan over repository to avoid binary decode errors.
Scans files with common source extensions and prints a summary.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List

try:
    from .core.orchestrator import OxpeckerOrchestrator
except Exception:
    from core.orchestrator import OxpeckerOrchestrator

ROOT = Path(__file__).resolve().parents[2]  # repo root
EXTS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.kt', '.c', '.cpp', '.cc', '.h', '.hpp',
    '.rs', '.go', '.cs', '.rb', '.php', '.lua', '.swift', '.zig'
}

orchestrator = OxpeckerOrchestrator()

found_files: List[Path] = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip virtualenvs, dist folders, .git
    skip = any(part in ('venv', 'env', '.git', 'dist', 'build', '__pycache__') for part in Path(dirpath).parts)
    if skip:
        continue
    for fn in filenames:
        if Path(fn).suffix.lower() in EXTS:
            found_files.append(Path(dirpath) / fn)

print(f"Scanning {len(found_files)} source files...")
summary = {'files': 0, 'errors': 0, 'warnings': 0, 'info': 0}
for p in found_files:
    try:
        issues = orchestrator.debug_file(str(p))
    except Exception as e:
        print(f"ERROR scanning {p}: {e}")
        continue
    summary['files'] += 1
    for i in issues:
        if i.severity == 'error':
            summary['errors'] += 1
        elif i.severity == 'warning':
            summary['warnings'] += 1
        else:
            summary['info'] += 1

print('\nScan summary:')
print(f"  Files scanned: {summary['files']}")
print(f"  Errors: {summary['errors']}")
print(f"  Warnings: {summary['warnings']}")
print(f"  Info: {summary['info']}")
