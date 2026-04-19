"""
janus_install_deps.py
======================
Auto-installer and code-health tool for all Janus / Avus files.

Uses the Universal Oxpecker (tools/universal_oxpecker) to:
  - Scan every janus_*.py / avus_*.py for third-party imports
  - Report and install any missing packages
  - Optionally run a full static-analysis scan or auto-repair pass

Usage
-----
  python janus_install_deps.py                  # install missing deps
  python janus_install_deps.py --check          # report only, no install
  python janus_install_deps.py --force          # reinstall even if present
  python janus_install_deps.py --file janus_metadata_reader.py
  python janus_install_deps.py --scan           # Oxpecker static analysis
  python janus_install_deps.py --repair         # Oxpecker auto-repair (safe copy)
  python janus_install_deps.py --scan --file janus_corrupt_file_reader.py
"""

import ast
import importlib
import importlib.util
import subprocess
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("janus.deps")


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT NAME  →  pip package name
# Some packages are imported under a different name than their PyPI slug.
# ═══════════════════════════════════════════════════════════════════════════════

IMPORT_TO_PIP: Dict[str, str] = {
    # imaging / vision
    "PIL":              "Pillow",
    "cv2":              "opencv-python",
    "imagehash":        "ImageHash",
    "mss":              "mss",
    "easyocr":          "easyocr",
    "pytesseract":      "pytesseract",

    # audio
    "mutagen":          "mutagen",
    "pyttsx3":          "pyttsx3",

    # PDF
    "pypdf":            "pypdf",
    "PyPDF2":           "PyPDF2",

    # data / ML
    "numpy":            "numpy",
    "torch":            "torch",
    "tiktoken":         "tiktoken",
    "kagglehub":        "kagglehub",

    # web / API
    "requests":         "requests",
    "fastapi":          "fastapi",
    "uvicorn":          "uvicorn",
    "flask":            "Flask",
    "werkzeug":         "Werkzeug",
    "pydantic":         "pydantic",
    "stripe":           "stripe",
    "grpc":             "grpcio",

    # system / automation
    "pyautogui":        "pyautogui",
    "pygetwindow":      "pygetwindow",
    "pyperclip":        "pyperclip",
    "psutil":           "psutil",
    "schedule":         "schedule",
    "win32process":     "pywin32",

    # QR / barcode
    "qrcode":           "qrcode",
    "pyzbar":           "pyzbar",

    # encoding detection
    "chardet":          "chardet",
}

# Packages that are optional / heavy and should only install when explicitly
# requested (skipped in default mode).
OPTIONAL_HEAVY: Set[str] = {
    "torch",        # multi-GB, user should install manually with CUDA variant
    "easyocr",      # pulls torch transitively
}

# Packages that are Windows-only
WINDOWS_ONLY: Set[str] = {
    "pygetwindow",
    "win32process",
    "pyperclip",    # works on all platforms but needs xclip on Linux
}

# Internal / local module names that look third-party but aren't pip packages
LOCAL_OR_INTERNAL: Set[str] = {
    "holographic_brain_memory",
    "janus_comms",
    "janus_core",
    "janus_finance",
    "janus_identity",
    "generators",
    "tools",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNING
# ═══════════════════════════════════════════════════════════════════════════════

def _stdlib_names() -> Set[str]:
    """Return the set of stdlib top-level module names."""
    if hasattr(sys, "stdlib_module_names"):          # Python 3.10+
        return set(sys.stdlib_module_names)
    # Fallback for older Python
    import sysconfig
    stdlib_path = sysconfig.get_paths()["stdlib"]
    names: Set[str] = set()
    for p in Path(stdlib_path).iterdir():
        names.add(p.stem)
    return names


_STDLIB = _stdlib_names()


def _local_module_names(root: Path) -> Set[str]:
    """All .py stems in the workspace root (local modules)."""
    return {f.stem for f in root.glob("*.py")}


def scan_file(path: Path, local_modules: Set[str]) -> Set[str]:
    """Return third-party top-level import names found in a single file."""
    found: Set[str] = set()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError) as e:
        logger.warning(f"Could not parse {path.name}: {e}")
        return found

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                _maybe_add(top, local_modules, found)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                _maybe_add(top, local_modules, found)
    return found


def _maybe_add(name: str, local_modules: Set[str], found: Set[str]) -> None:
    if (
        name
        and name not in _STDLIB
        and name not in local_modules
        and name not in LOCAL_OR_INTERNAL
        and not name.startswith("_")
    ):
        found.add(name)


def scan_files(paths: List[Path]) -> Dict[str, Set[str]]:
    """
    Scan a list of files.
    Returns {import_name: {file1, file2, ...}} mapping.
    """
    root = Path(".")
    local_modules = _local_module_names(root)
    result: Dict[str, Set[str]] = {}

    for path in paths:
        imports = scan_file(path, local_modules)
        for imp in imports:
            result.setdefault(imp, set()).add(path.name)

    return result


def collect_target_files(specific_file: Optional[str] = None) -> List[Path]:
    """Return the list of .py files to scan."""
    root = Path(".")
    if specific_file:
        p = Path(specific_file)
        if not p.exists():
            logger.error(f"File not found: {specific_file}")
            sys.exit(1)
        return [p]

    prefixes = ("janus_", "avus_", "autonomous_", "advanced_3d", "archive/")
    files: List[Path] = []
    for f in sorted(root.glob("*.py")):
        if any(f.name.startswith(p) for p in prefixes):
            files.append(f)
    for f in sorted((root / "archive").glob("*.py")):
        files.append(f)
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

def is_installed(import_name: str) -> bool:
    """Check if a package is importable in the current environment."""
    spec = importlib.util.find_spec(import_name)
    return spec is not None


def pip_install(pip_name: str) -> Tuple[bool, str]:
    """
    Run pip install for a single package.
    Returns (success, output).
    """
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "pip install timed out after 300s"
    except Exception as e:
        return False, str(e)


def resolve_pip_name(import_name: str) -> Optional[str]:
    """
    Map an import name to its pip package name.
    Returns None if it's not a known installable package.
    """
    if import_name in LOCAL_OR_INTERNAL:
        return None
    return IMPORT_TO_PIP.get(import_name, import_name)  # default: same name


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    check_only: bool = False,
    force: bool = False,
    specific_file: Optional[str] = None,
    skip_heavy: bool = True,
) -> None:
    is_windows = sys.platform == "win32"

    # ── Scan ──────────────────────────────────────────────────────────────
    files = collect_target_files(specific_file)
    logger.info(f"Scanning {len(files)} file(s)...")
    import_map = scan_files(files)

    if not import_map:
        logger.info("No third-party imports found.")
        return

    # ── Categorise ────────────────────────────────────────────────────────
    to_install:  List[Tuple[str, str]] = []   # (import_name, pip_name)
    already_ok:  List[str] = []
    skipped:     List[str] = []
    unknown:     List[str] = []

    for imp in sorted(import_map):
        pip_name = resolve_pip_name(imp)
        if pip_name is None:
            skipped.append(imp)
            continue

        if imp in OPTIONAL_HEAVY and skip_heavy:
            logger.info(f"  SKIP (heavy)  {imp} → {pip_name}  "
                        f"(pass --include-heavy to install)")
            skipped.append(imp)
            continue

        if imp in WINDOWS_ONLY and not is_windows:
            logger.info(f"  SKIP (win32)  {imp} → {pip_name}")
            skipped.append(imp)
            continue

        if not force and is_installed(imp):
            already_ok.append(imp)
        else:
            to_install.append((imp, pip_name))

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  JANUS DEPENDENCY SCAN RESULTS")
    print("=" * 60)

    if already_ok:
        print(f"\n  Already installed ({len(already_ok)}):")
        for imp in already_ok:
            print(f"    ✓  {imp}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for imp in skipped:
            print(f"    -  {imp}")

    if to_install:
        print(f"\n  {'Would install' if check_only else 'Installing'} ({len(to_install)}):")
        for imp, pip_name in to_install:
            used_in = ", ".join(sorted(import_map[imp])[:3])
            suffix = " ..." if len(import_map[imp]) > 3 else ""
            print(f"    ↓  {pip_name:<30} (imported as '{imp}' in {used_in}{suffix})")
    else:
        print("\n  Nothing to install — all dependencies satisfied.")

    print()

    if check_only or not to_install:
        return

    # ── Install ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  INSTALLING")
    print("=" * 60)

    succeeded: List[str] = []
    failed:    List[Tuple[str, str]] = []

    for imp, pip_name in to_install:
        logger.info(f"Installing {pip_name}...")
        ok, msg = pip_install(pip_name)
        if ok:
            logger.info(f"  ✓  {pip_name} installed")
            succeeded.append(pip_name)
        else:
            logger.error(f"  ✗  {pip_name} FAILED: {msg}")
            failed.append((pip_name, msg))

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Installed : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Already OK: {len(already_ok)}")

    if failed:
        print("\n  Failed packages (install manually):")
        for pkg, reason in failed:
            print(f"    ✗  {pkg}: {reason}")
        print()
        print("  Tip: for torch, install the CUDA variant from https://pytorch.org")
        sys.exit(1)

    print("\n  All dependencies ready.\n")


def _check_only_exit(to_install: list) -> None:
    """Exit 0 even when packages are missing in --check mode."""
    pass  # intentional — check mode is informational only


# ═══════════════════════════════════════════════════════════════════════════════
# OXPECKER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

_OXPECKER_PATH = Path(__file__).parent / "tools" / "universal_oxpecker"

_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _load_oxpecker():
    """Import OxpeckerOrchestrator, adding the tools path to sys.path if needed."""
    tools_path = str(_OXPECKER_PATH.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    try:
        from universal_oxpecker.core.orchestrator import OxpeckerOrchestrator
        return OxpeckerOrchestrator
    except ImportError as e:
        logger.error(f"Could not load Universal Oxpecker: {e}")
        logger.error(f"Expected at: {_OXPECKER_PATH}")
        sys.exit(1)


def _sev_colour(severity: str) -> str:
    return {"error": _RED, "warning": _YELLOW, "info": _CYAN}.get(
        severity.lower(), _RESET
    )


def _print_issue(issue, index: int) -> None:
    col = _sev_colour(issue.severity)
    tier = getattr(issue, "complexity_tier", None) or "Tier ?"
    loc = ""
    if getattr(issue, "source_path", None):
        loc = f"  {issue.source_path}"
        if getattr(issue, "line", None):
            loc += f":{issue.line}"
    print(
        f"\n  {_BOLD}#{index}{_RESET} "
        f"{col}[{issue.severity.upper()}]{_RESET} "
        f"({issue.language}) [{tier}]{loc}"
    )
    print(f"     {issue.message}")
    if getattr(issue, "fix_suggestion", None):
        print(f"     {_GREEN}>> {issue.fix_suggestion}{_RESET}")


def run_oxpecker_scan(specific_file: Optional[str] = None, workers: int = 4) -> None:
    """Run Oxpecker static analysis on Janus files and print a report."""
    OxpeckerOrchestrator = _load_oxpecker()
    orchestrator = OxpeckerOrchestrator()

    if specific_file:
        target = specific_file
        print(f"\n{_BOLD}Oxpecker scan:{_RESET} {target}\n")
        try:
            issues = orchestrator.debug_file(target)
        except ValueError as e:
            print(f"  {_RED}ERROR:{_RESET} {e}")
            sys.exit(1)

        if not issues:
            print(f"  {_GREEN}No issues found.{_RESET}\n")
            return

        errors   = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        print(
            f"  {_RED}{len(errors)} error(s){_RESET}, "
            f"{_YELLOW}{len(warnings)} warning(s){_RESET}\n"
        )
        for idx, issue in enumerate(issues, 1):
            _print_issue(issue, idx)
        print()
        return

    # Scan the whole Janus file set
    files = collect_target_files()
    # Filter to only files Oxpecker supports (it will skip unsupported ones)
    target_dir = str(Path("."))
    print(f"\n{_BOLD}Oxpecker scan:{_RESET} {len(files)} Janus/Avus files\n")

    total_errors = 0
    total_warnings = 0
    files_with_issues = 0

    for fpath in files:
        try:
            issues = orchestrator.debug_file(str(fpath))
        except ValueError:
            continue  # no adapter for this file
        if not issues:
            continue

        files_with_issues += 1
        errors   = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        total_errors   += len(errors)
        total_warnings += len(warnings)

        print(
            f"  {_BOLD}{fpath.name}{_RESET}  "
            f"{_RED}{len(errors)}E{_RESET} "
            f"{_YELLOW}{len(warnings)}W{_RESET}"
        )
        for idx, issue in enumerate(issues, 1):
            _print_issue(issue, idx)
        print()

    print("=" * 60)
    print(
        f"  Files scanned : {len(files)}\n"
        f"  Files with issues: {files_with_issues}\n"
        f"  Total errors  : {_RED}{total_errors}{_RESET}\n"
        f"  Total warnings: {_YELLOW}{total_warnings}{_RESET}\n"
    )


def run_oxpecker_repair(
    specific_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_rounds: int = 10,
    workers: int = 4,
) -> None:
    """Run Oxpecker auto-repair on Janus files (safe working copies only)."""
    OxpeckerOrchestrator = _load_oxpecker()
    orchestrator = OxpeckerOrchestrator()

    output_dir = output_dir or ".oxpecker_work"

    if specific_file:
        print(f"\n{_BOLD}Oxpecker repair:{_RESET} {specific_file}\n")
        repair = orchestrator.repair_file(
            specific_file, output_dir=output_dir, max_rounds=max_rounds
        )
        repairs = [repair]
    else:
        files = collect_target_files()
        print(
            f"\n{_BOLD}Oxpecker repair:{_RESET} {len(files)} files "
            f"(safe copies in {output_dir}/)\n"
        )
        repairs = []
        for fpath in files:
            try:
                repairs.append(
                    orchestrator.repair_file(
                        str(fpath), output_dir=output_dir, max_rounds=max_rounds
                    )
                )
            except ValueError:
                continue

    for repair in repairs:
        fixed = repair.fixed_issue_count
        remaining = repair.final_issue_count
        colour = _GREEN if fixed > 0 else _YELLOW
        print(
            f"  {_BOLD}{Path(repair.original_path).name}{_RESET}  "
            f"{colour}fixed {fixed}{_RESET}  "
            f"remaining {remaining}  "
            f"→ {repair.working_path}"
        )
        for patch in repair.applied_patches:
            print(f"    ✓ {patch.description} [{patch.complexity_tier}]")
        if repair.pending_manual_fixes:
            print(
                f"    {_YELLOW}⚠ {len(repair.pending_manual_fixes)} "
                f"issue(s) need manual review{_RESET}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-install Janus/Avus deps and run Oxpecker code health checks"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report missing deps without installing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall packages even if already present",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Target a single file instead of all Janus files",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        dest="include_heavy",
        help="Also install heavy optional packages (torch, easyocr)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run Oxpecker static analysis on Janus files",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Run Oxpecker auto-repair on safe working copies of Janus files",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=".oxpecker_work",
        help="Output directory for Oxpecker repair working copies (default: .oxpecker_work)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        dest="max_rounds",
        help="Max repair rounds per file (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for Oxpecker scan/repair (default: 4)",
    )
    args = parser.parse_args()

    # Oxpecker modes — can combine with dep install
    if args.scan:
        run_oxpecker_scan(specific_file=args.file, workers=args.workers)

    if args.repair:
        run_oxpecker_repair(
            specific_file=args.file,
            output_dir=args.output_dir,
            max_rounds=args.max_rounds,
            workers=args.workers,
        )

    # Dep install (default behaviour unless only --scan/--repair was requested)
    if not args.scan and not args.repair:
        run(
            check_only=args.check,
            force=args.force,
            specific_file=args.file,
            skip_heavy=not args.include_heavy,
        )
    elif not args.scan and not args.repair:
        # explicit dep install alongside scan/repair
        run(
            check_only=args.check,
            force=args.force,
            specific_file=args.file,
            skip_heavy=not args.include_heavy,
        )


if __name__ == "__main__":
    main()
