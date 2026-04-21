"""
janus_cli.py
============
Single entry point for Janus. Run goals, check status, view history.

Usage:
    python janus_cli.py "Read TODO.md and tell me what's most important"
    python janus_cli.py --status
    python janus_cli.py --history
    python janus_cli.py --tools
    python janus_cli.py --interactive
"""

import sys
import json
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from janus_agent_core import JanusAgent


def print_banner():
    print("""
     ╔═══════════════════════════════════════╗
     ║           J A N U S   A I             ║
     ║       Autonomous Agent System         ║
     ╚═══════════════════════════════════════╝
    """)


def interactive_mode(agent: JanusAgent):
    """REPL-style interactive mode."""
    print_banner()
    print("Type a goal and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            goal = input("janus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not goal:
            continue
        if goal.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if goal.lower() == "status":
            status = agent.status()
            print(json.dumps(status, indent=2, default=str))
            continue
        if goal.lower() == "history":
            plans = agent.memory.recent_plans(10)
            for p in plans:
                icon = "✓" if p["completed"] else "✗"
                print(f"  {icon} {p['goal'][:50]}")
            continue
        if goal.lower() == "tools":
            for t in agent.executor.registry.list_all():
                print(f"  {t['name']:<15} [{t['risk']}] {t['description']}")
            continue

        plan = agent.run(goal)
        print(f"\nResult: {plan.outcome}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Janus AI — Autonomous Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--history", action="store_true", help="Show plan history")
    parser.add_argument("--tools", action="store_true", help="List available tools")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()
    agent = JanusAgent(verbose=not args.quiet)

    if args.interactive:
        interactive_mode(agent)
        return

    if args.status:
        status = agent.status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.history:
        plans = agent.memory.recent_plans(20)
        if not plans:
            print("No plan history found.")
            return
        print(f"\nRecent plans ({len(plans)}):")
        for p in plans:
            icon = "✓" if p["completed"] else "✗"
            ts = p["timestamp"][:19]
            print(f"  {icon} [{ts}] {p['goal'][:50]}")
        rate = agent.memory.success_rate()
        print(f"\nSuccess rate: {rate:.0%}")
        return

    if args.tools:
        tools = agent.executor.registry.list_all()
        print(f"\nAvailable tools ({len(tools)}):\n")
        for t in tools:
            print(f"  {t['name']:<15} [{t['risk']:<6}] {t['description']}")
            for param, ptype in t["parameters"].items():
                print(f"    ├─ {param}: {ptype}")
        return

    if args.goal:
        plan = agent.run(args.goal)
        print(f"\n{'─'*40}")
        print(f"Result: {plan.outcome}")
        for step in plan.steps:
            icon = "✓" if step.success else "✗"
            print(f"  {icon} {step.description}")
            if step.result and step.success:
                preview = str(step.result)[:300]
                print(f"    → {preview}")
        return

    # No args — show banner and help
    print_banner()
    parser.print_help()


if __name__ == "__main__":
    main()
