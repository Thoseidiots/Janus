"""
launch_arania.py
================
Master launcher for the Arania full-body roaming avatar.

Steps:
  1. Generate Arania's full-body mesh + scene (arania_body_generator.py)
  2. Build the OSS engine (cargo build --release)
  3. Start the engine with the generated scene
  4. Open a control terminal for sending expression commands

Usage:
    python launch_arania.py
    python launch_arania.py --skip-build    # if engine already built
    python launch_arania.py --skip-generate # if assets already exist
    python launch_arania.py --headless      # run engine tests only
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

WORKSPACE  = Path(__file__).parent
ENGINE_DIR = WORKSPACE / "oss-game-engine"
ASSETS_DIR = WORKSPACE / "assets"
SCENE_FILE = WORKSPACE / "arania_world.ks"


# ── Step 1 — Generate body asset ─────────────────────────────────────────────

def generate_body(force: bool = False) -> bool:
    mesh = ASSETS_DIR / "arania_body.obj"
    if mesh.exists() and not force:
        print(f"[launch] Mesh already exists: {mesh}  (pass --regen to rebuild)")
        return True

    print("[launch] Generating Arania body mesh...")
    result = subprocess.run(
        [sys.executable, str(WORKSPACE / "arania_body_generator.py")],
        cwd=str(WORKSPACE),
    )
    if result.returncode != 0:
        print("[launch] ❌ Body generation failed.")
        return False
    print("[launch] ✅ Body mesh generated.")
    return True


# ── Step 2 — Build the Rust engine ───────────────────────────────────────────

def build_engine(release: bool = True) -> bool:
    profile = "--release" if release else "--debug"
    mode    = "release" if release else "debug"
    binary  = ENGINE_DIR / "target" / mode / "engine-runtime"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")

    print(f"[launch] Building OSS engine ({mode})...")
    print(f"[launch]   cargo build {profile} -p engine-runtime")

    # Run cargo build
    result = subprocess.run(
        ["cargo", "build", profile, "-p", "engine-runtime"],
        cwd=str(ENGINE_DIR),
    )
    if result.returncode != 0:
        print("[launch] ❌ Engine build failed.")
        return False

    if not binary.exists():
        print(f"[launch] ⚠️  Binary not found at {binary} — may be a lib-only crate.")
        print("[launch] Running engine tests instead...")
        return run_engine_tests()

    print(f"[launch] ✅ Engine built: {binary}")
    return True


def run_engine_tests() -> bool:
    """Run cargo test to validate all engine modules."""
    print("[launch] Running engine test suite...")
    result = subprocess.run(
        ["cargo", "test", "--workspace", "--", "--test-threads=4"],
        cwd=str(ENGINE_DIR),
    )
    passed = result.returncode == 0
    print(f"[launch] {'✅' if passed else '❌'} Engine tests {'passed' if passed else 'failed'}.")
    return passed


# ── Step 3 — Start engine process ────────────────────────────────────────────

def start_engine(scene: Path, headless: bool = False) -> subprocess.Popen | None:
    binary = ENGINE_DIR / "target" / "release" / "engine-runtime"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")

    if not binary.exists():
        print(f"[launch] Engine binary not found at {binary}")
        print("[launch] Running in test mode (no window)...")
        run_engine_tests()
        return None

    args = [str(binary), "--scene", str(scene)]
    if headless:
        args.append("--headless")

    print(f"[launch] Starting engine: {' '.join(args)}")
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


# ── Step 4 — Command bridge (Python → engine) ─────────────────────────────────

def send_command(proc: subprocess.Popen, cmd: dict) -> None:
    """Send a JSON command to the engine's stdin."""
    if proc and proc.stdin and not proc.stdin.closed:
        line = json.dumps(cmd) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()


def engine_stdout_reader(proc: subprocess.Popen) -> None:
    """Thread: print engine stdout in real time."""
    for line in proc.stdout:
        print(f"[engine] {line}", end="")


def interactive_control(proc: subprocess.Popen | None) -> None:
    """Simple REPL for controlling Arania from the terminal."""
    print("\n[control] Arania command shell — type 'help' for options.\n")
    commands = {
        "smile":    {"cmd": "expr",   "value": "smile"},
        "neutral":  {"cmd": "expr",   "value": "neutral"},
        "thinking": {"cmd": "expr",   "value": "thinking"},
        "walk":     {"cmd": "walk"},
        "stop":     {"cmd": "stop"},
        "help": None,
        "quit": None,
    }

    while True:
        try:
            raw = input("[arania]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[control] Shutting down.")
            break

        if raw == "quit":
            break
        elif raw == "help":
            print("  Commands: " + ", ".join(k for k in commands if k not in ("help", "quit")))
            print("  status <text>   — set status bar text")
            print("  say <text>      — make Arania appear to speak")
        elif raw.startswith("status "):
            msg = raw[7:]
            if proc:
                send_command(proc, {"cmd": "status", "value": msg})
            else:
                print(f"[control] (no engine) Status: {msg}")
        elif raw.startswith("say "):
            phonemes = [c.upper() for c in raw[4:] if c.isalpha()][:20]
            if proc:
                send_command(proc, {"cmd": "talk", "phonemes": phonemes})
            else:
                print(f"[control] (no engine) Talk: {phonemes}")
        elif raw in commands and commands[raw] is not None:
            if proc:
                send_command(proc, commands[raw])
            else:
                print(f"[control] (no engine) Command: {raw}")
        elif raw:
            print(f"[control] Unknown command: '{raw}'  (type 'help')")

    if proc:
        proc.terminate()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch Arania roaming avatar")
    parser.add_argument("--skip-build",    action="store_true",
                        help="Skip cargo build (use existing binary)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip body mesh generation")
    parser.add_argument("--regen",         action="store_true",
                        help="Force regenerate body mesh even if it exists")
    parser.add_argument("--headless",      action="store_true",
                        help="Run engine tests only (no window)")
    parser.add_argument("--debug",         action="store_true",
                        help="Build engine in debug mode (faster compile)")
    args = parser.parse_args()

    print("=" * 60)
    print("  JANUS — ARANIA ROAMING AVATAR LAUNCHER")
    print("=" * 60)

    # Step 1 — Generate body
    if not args.skip_generate:
        if not generate_body(force=args.regen):
            sys.exit(1)

    # Step 2 — Build engine
    if not args.skip_build:
        if not build_engine(release=not args.debug):
            print("[launch] Build failed — running engine tests as fallback...")
            run_engine_tests()

    # Step 3 — Validate scene file
    if not SCENE_FILE.exists():
        print(f"[launch] ⚠️  Scene file not found: {SCENE_FILE}")
        print("[launch] Run arania_body_generator.py first.")
        if not args.skip_generate:
            sys.exit(1)

    # Step 4 — Start engine
    proc = start_engine(SCENE_FILE, headless=args.headless)

    # Pipe engine stdout to terminal
    if proc:
        t = threading.Thread(target=engine_stdout_reader, args=(proc,), daemon=True)
        t.start()
        time.sleep(0.5)

    # Step 5 — Interactive control shell
    if not args.headless:
        interactive_control(proc)
    else:
        print("[launch] Headless mode — running 3s simulation then exiting.")
        time.sleep(3)
        if proc:
            proc.terminate()

    print("[launch] Done.")


if __name__ == "__main__":
    main()
