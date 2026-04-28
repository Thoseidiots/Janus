"""
Command-line interface for the Janus Autonomous Reasoning Engine.

Commands:
  start          Initialize the engine and run autonomous cycles
  status         Print current engine status
  set-goal GOAL  Set the current high-level goal
  stop           Stop a running engine (sends SIGTERM to PID file)

Requirements: REQ-9.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time

logger = logging.getLogger("janus.cli")

_PID_FILE = "janus_engine.pid"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="janus-engine",
        description="Janus Autonomous Reasoning Engine CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    start_p = sub.add_parser("start", help="Initialize engine and run cycles")
    start_p.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run (0 = run until stopped)",
    )
    start_p.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Initial goal to set before running",
    )
    start_p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )

    # status
    sub.add_parser("status", help="Print current engine status")

    # set-goal
    goal_p = sub.add_parser("set-goal", help="Set the current goal")
    goal_p.add_argument("goal", type=str, help="Goal description")

    # stop
    sub.add_parser("stop", help="Stop a running engine")

    return parser


def cmd_start(args: argparse.Namespace) -> int:
    """Start the engine and run cycles."""
    from janus_reasoning_engine.engine import JanusReasoningEngine
    from janus_reasoning_engine.core.config import EngineConfig

    # Load config
    config = EngineConfig()
    if args.config:
        try:
            config = EngineConfig.load(args.config)
            print(f"Loaded config from {args.config}")
        except Exception as exc:
            print(f"Warning: could not load config '{args.config}': {exc}", file=sys.stderr)

    engine = JanusReasoningEngine(config=config)
    engine.initialize()

    # Write PID file
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Set initial goal
    if args.goal:
        goal_id = engine.set_goal(args.goal)
        if goal_id:
            print(f"Goal set: {args.goal!r} (id={goal_id})")
        else:
            print(f"Goal set: {args.goal!r} (goal manager unavailable)")

    # Handle SIGTERM / SIGINT gracefully
    _running = [True]

    def _stop_handler(signum, frame):
        print("\nReceived stop signal — shutting down …")
        _running[0] = False

    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)

    print("Engine started. Press Ctrl+C to stop.")
    cycle = 0
    try:
        while _running[0]:
            outcome = engine.run_cycle()
            cycle += 1
            print(
                f"Cycle {outcome.cycle_num}: action={outcome.action_taken} "
                f"success={outcome.success} earnings=${outcome.earnings:.2f}"
            )
            if args.cycles > 0 and cycle >= args.cycles:
                break
            time.sleep(1)
    finally:
        engine.shutdown()
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
        print("Engine stopped.")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print engine status (reads from a running engine via status file if available)."""
    status_file = "janus_engine_status.json"
    if os.path.exists(status_file):
        try:
            with open(status_file) as f:
                status = json.load(f)
            print(json.dumps(status, indent=2))
            return 0
        except Exception:
            pass

    # Fallback: create a fresh engine and print its initial status
    from janus_reasoning_engine.engine import JanusReasoningEngine
    engine = JanusReasoningEngine()
    engine.initialize()
    status = engine.get_status()
    engine.shutdown()
    print(json.dumps(status, indent=2, default=str))
    return 0


def cmd_set_goal(args: argparse.Namespace) -> int:
    """Set a goal on a running engine (or start a temporary one)."""
    from janus_reasoning_engine.engine import JanusReasoningEngine
    engine = JanusReasoningEngine()
    engine.initialize()
    goal_id = engine.set_goal(args.goal)
    if goal_id:
        print(f"Goal set: {args.goal!r} (id={goal_id})")
    else:
        print(f"Goal queued: {args.goal!r} (goal manager unavailable)")
    engine.shutdown()
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Send SIGTERM to a running engine process."""
    if not os.path.exists(_PID_FILE):
        print("No running engine found (no PID file).", file=sys.stderr)
        return 1
    try:
        with open(_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stop signal sent to engine (PID {pid})")
        return 0
    except (ValueError, ProcessLookupError) as exc:
        print(f"Could not stop engine: {exc}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "start": cmd_start,
        "status": cmd_status,
        "set-goal": cmd_set_goal,
        "stop": cmd_stop,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
