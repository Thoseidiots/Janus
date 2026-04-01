#!/usr/bin/env python3
"""
Universal Oxpecker CLI
======================
A universal multi-language static analyser, scanner, and stronger safe APR entry-point.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from typing import List, Any

try:
    from .core.orchestrator import OxpeckerOrchestrator
except ImportError:  # pragma: no cover - script-mode fallback
    from core.orchestrator import OxpeckerOrchestrator

_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

VERSION = "1.2.0"
BANNER = (
    f"{_BOLD}{_GREEN}\n"
    f"  Universal Oxpecker  v{VERSION}\n"
    f"  Multi-language debugger, scanner & stronger safe auto-repair\n"
    f"{_RESET}"
)


def _sev_colour(severity: str) -> str:
    return {"error": _RED, "warning": _YELLOW, "info": _CYAN}.get(severity.lower(), _RESET)


def _tier_colour(tier: str) -> str:
    return {"Tier 1": _GREEN, "Tier 2": _YELLOW, "Tier 3": _MAGENTA}.get(tier, _RESET)


def _print_issue(issue: Any, index: int) -> None:
    col = _sev_colour(issue.severity)
    tier = issue.complexity_tier or "Tier ?"
    tier_col = _tier_colour(tier)
    loc = ""
    if issue.source_path:
        loc = f"  {issue.source_path}"
        if issue.line:
            loc += f":{issue.line}"
            if issue.column:
                loc += f":{issue.column}"
    print(f"\n  {_BOLD}#{index}{_RESET} {col}[{issue.severity.upper()}]{_RESET} ({issue.language}) {tier_col}[{tier}]{_RESET}{loc}")
    print(f"     {issue.message}")
    if issue.fix_suggestion:
        for line in textwrap.wrap(issue.fix_suggestion, width=76):
            print(f"     {_GREEN}>> {line}{_RESET}")


def cmd_langs(_args) -> None:
    orchestrator = OxpeckerOrchestrator()
    print(BANNER)
    print(f"  {_BOLD}Registered language adapters:{_RESET}")
    for lang in orchestrator.get_registered_languages():
        print(f"    - {lang}")
    print()


def cmd_version(_args) -> None:
    print(f"Universal Oxpecker v{VERSION}")


def cmd_debug(args) -> None:
    target: str = args.target
    print(BANNER)
    print(f"  {_BOLD}Analysing:{_RESET} {target}\n")

    orchestrator = OxpeckerOrchestrator()

    try:
        issues = orchestrator.debug_file(target)
    except ValueError as e:
        print(f"  {_RED}ERROR:{_RESET} {e}")
        sys.exit(1)

    if not issues:
        print(f"  {_GREEN}No issues found.{_RESET}\n")
        return

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    print(
        f"  Found {_RED}{len(errors)} error(s){_RESET}, "
        f"{_YELLOW}{len(warnings)} warning(s){_RESET}, "
        f"{_CYAN}{len(infos)} info(s){_RESET}\n"
    )

    for idx, issue in enumerate(issues, start=1):
        _print_issue(issue, idx)

    print()
    if errors:
        sys.exit(2)


cmd_check = cmd_debug


def _print_scan_result(scan: Any) -> None:
    print(
        f"  {_BOLD}{scan.source_path}{_RESET} "
        f"({_CYAN}{scan.language}{_RESET}) -> "
        f"{_RED}{scan.error_count} error(s){_RESET}, "
        f"{_YELLOW}{scan.warning_count} warning(s){_RESET}, "
        f"{_CYAN}{scan.info_count} info(s){_RESET}"
    )


def cmd_scan(args) -> None:
    print(BANNER)
    orchestrator = OxpeckerOrchestrator()
    scan_results = orchestrator.scan_path(args.target, recursive=not args.no_recursive, workers=args.workers)
    if not scan_results:
        print(f"  {_YELLOW}No supported source files found.{_RESET}")
        return

    total_issues = 0
    for scan in scan_results:
        _print_scan_result(scan)
        total_issues += len(scan.issues)
        if args.verbose:
            for idx, issue in enumerate(scan.issues, start=1):
                _print_issue(issue, idx)
                print()

    print(f"\n  {_BOLD}Scanned files:{_RESET} {len(scan_results)}")
    print(f"  {_BOLD}Total issues:{_RESET} {total_issues}\n")


def cmd_repair(args) -> None:
    print(BANNER)
    orchestrator = OxpeckerOrchestrator()

    if args.project:
        repairs = orchestrator.repair_project(
            args.target,
            recursive=not args.no_recursive,
            workers=args.workers,
            output_dir=args.output_dir,
            max_rounds=args.max_rounds,
        )
    else:
        repairs = [
            orchestrator.repair_file(
                args.target,
                output_dir=args.output_dir,
                max_rounds=args.max_rounds,
            )
        ]

    if not repairs:
        print(f"  {_YELLOW}No repairable source files found.{_RESET}")
        return

    for repair in repairs:
        print(f"\n  {_BOLD}Original:{_RESET} {repair.original_path}")
        print(f"  {_BOLD}Working copy:{_RESET} {repair.working_path}")
        print(f"  {_BOLD}Language:{_RESET} {repair.language}")
        print(
            f"  {_BOLD}Issues:{_RESET} {repair.initial_issue_count} -> {repair.final_issue_count} "
            f"({_GREEN}fixed {repair.fixed_issue_count}{_RESET})"
        )

        if repair.applied_patches:
            print(f"  {_BOLD}Applied patches:{_RESET}")
            for idx, patch in enumerate(repair.applied_patches, start=1):
                print(
                    f"    {idx}. {patch.description} "
                    f"[{patch.complexity_tier}] via {patch.plugin_name} / {patch.patch_kind}"
                )
                print(
                    f"       score={patch.score} issue_reduction={patch.issue_reduction} "
                    f"tests={patch.test_status}"
                )
                print(f"       {patch.validation_notes}")
        else:
            print(f"  {_YELLOW}No auto-approved patches were applied.{_RESET}")

        if args.show_candidates and repair.candidate_history:
            print(f"  {_BOLD}Candidate ranking:{_RESET}")
            for idx, candidate in enumerate(repair.candidate_history[: args.show_candidates], start=1):
                print(
                    f"    {idx}. {candidate.description} -> {candidate.approval} "
                    f"(score={candidate.score}, plugin={candidate.plugin_name}, tests={candidate.test_status})"
                )

        if repair.test_runs:
            print(f"  {_BOLD}Test validation:{_RESET}")
            for run in repair.test_runs:
                status = f"{_GREEN}passed{_RESET}" if run.success else f"{_RED}failed{_RESET}"
                print(f"    - {run.kind}: {run.command} -> {status}")

        if repair.rollback_history:
            print(f"  {_BOLD}Rollback snapshots:{_RESET} {len(repair.rollback_history)}")
            latest = repair.rollback_history[-1]
            print(f"    latest revision={latest.revision} snapshot={latest.snapshot_path}")

        if repair.pending_manual_fixes:
            print(f"  {_BOLD}Manual review queue:{_RESET}")
            for issue in repair.pending_manual_fixes:
                print(f"    - [{issue.complexity_tier or 'Tier ?'}] {issue.message}")

        if repair.remaining_issues:
            print(f"  {_BOLD}Remaining issues:{_RESET}")
            for idx, issue in enumerate(repair.remaining_issues, start=1):
                _print_issue(issue, idx)
                print()


def cmd_rollback(args) -> None:
    print(BANNER)
    orchestrator = OxpeckerOrchestrator()
    restored = orchestrator.rollback_file(args.target, steps=args.steps)
    print(f"  {_BOLD}Restored working copy:{_RESET} {args.target}")
    print(f"  {_BOLD}Revision:{_RESET} {restored.revision}")
    print(f"  {_BOLD}Snapshot:{_RESET} {restored.snapshot_path}")
    print(f"  {_BOLD}Description:{_RESET} {restored.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oxpecker",
        description="Universal Oxpecker - multi-language debugger, scanner & stronger safe auto-repair",
    )
    sub = parser.add_subparsers(dest="command")

    for name, hlp in [
        ("debug", "Analyse and debug a source file"),
        ("check", "Alias for debug (static analysis only)"),
    ]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("target", help="Path to source file or language tag")

    p_scan = sub.add_parser("scan", help="Scan one file or a whole project tree")
    p_scan.add_argument("target", help="Path to file or directory")
    p_scan.add_argument("--workers", type=int, default=4, help="Number of Oxpecker workers")
    p_scan.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    p_scan.add_argument("--verbose", action="store_true", help="Print per-issue details")

    for name, hlp in [
        ("repair", "Run stronger safe APR on a file copy"),
        ("autofix", "Alias for repair"),
    ]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("target", help="Path to file or directory")
        p.add_argument("--project", action="store_true", help="Treat target as a project directory")
        p.add_argument("--workers", type=int, default=4, help="Number of Oxpecker workers for project mode")
        p_scan.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
        p.add_argument("--output-dir", default=None, help="Directory for safe working copies")
        p.add_argument("--max-rounds", type=int, default=10, help="Maximum repair rounds per file")
        p.add_argument(
            "--show-candidates",
            type=int,
            default=0,
            help="Show the top N ranked candidates after validation",
        )

    p_rollback = sub.add_parser("rollback", help="Rollback a repaired working copy using snapshot history")
    p_rollback.add_argument("target", help="Path to a repaired working copy")
    p_rollback.add_argument("--steps", type=int, default=1, help="How many snapshots to roll back")

    sub.add_parser("langs", help="List all registered language adapters")
    sub.add_parser("version", help="Print version and exit")

    args = parser.parse_args()
    dispatch = {
        "debug": cmd_debug,
        "check": cmd_check,
        "scan": cmd_scan,
        "repair": cmd_repair,
        "autofix": cmd_repair,
        "rollback": cmd_rollback,
        "langs": cmd_langs,
        "version": cmd_version,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
