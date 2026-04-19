"""
janus_daemon.py — Windows process supervisor for the Janus autonomous worker.

Manages the lifecycle of JanusAutonomousWorker, including:
- Startup registration via Windows registry
- Crash detection and restart with rate limiting
- Health file writing every 60 seconds
- FastAPI status endpoint on port 8006
- Rotating log file
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import signal
import time
import pathlib
import argparse
import threading
from datetime import datetime, timezone

try:
    import winreg
    WINREG_AVAILABLE = True
except Exception:
    WINREG_AVAILABLE = False

try:
    import subprocess
except Exception:
    subprocess = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FILE = pathlib.Path("janus_daemon.log")
HEALTH_FILE = pathlib.Path("janus_health.json")
REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
REGISTRY_VALUE = "JanusDaemon"
API_PORT = 8006
HEALTH_INTERVAL = 60          # seconds between health file writes
RESTART_WINDOW = 3600         # 1 hour in seconds
MAX_RESTARTS_PER_HOUR = 10
RESTART_DELAY = 30            # seconds to wait before restarting


def _setup_logging() -> logging.Logger:
    """Configure rotating file logger plus console output."""
    logger = logging.getLogger("janus_daemon")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Rotating file handler: max 10 MB, keep 3 backups
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# Daemon state (shared between threads)
# ---------------------------------------------------------------------------

_state = {
    "status": "starting",
    "start_time": time.time(),
    "restart_count": 0,
    "last_error": "",
    "cycle_count": 0,
    "current_mood": "neutral",
    "current_energy": 1.0,
    "stop_requested": False,
}
_state_lock = threading.Lock()


def _get_state_snapshot() -> dict:
    """Return a thread-safe copy of the current daemon state."""
    with _state_lock:
        snap = dict(_state)
    snap["uptime_seconds"] = round(time.time() - snap["start_time"], 1)
    snap["timestamp"] = datetime.now(timezone.utc).isoformat()
    return snap


def _update_state(**kwargs) -> None:
    """Thread-safe update of daemon state fields."""
    with _state_lock:
        _state.update(kwargs)


# ---------------------------------------------------------------------------
# Health file
# ---------------------------------------------------------------------------

def _write_health_file() -> None:
    """Write current daemon state to janus_health.json."""
    try:
        snap = _get_state_snapshot()
        HEALTH_FILE.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Failed to write health file: %s", exc)


def _health_writer_loop() -> None:
    """Background thread: writes health file every HEALTH_INTERVAL seconds."""
    while not _state.get("stop_requested"):
        _write_health_file()
        time.sleep(HEALTH_INTERVAL)
    _write_health_file()  # final write on shutdown


# ---------------------------------------------------------------------------
# FastAPI status endpoint
# ---------------------------------------------------------------------------

def _build_app():
    """Build and return the FastAPI application."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn  # noqa: F401 — imported to verify availability
    except Exception as exc:
        log.warning("FastAPI/uvicorn not available, status endpoint disabled: %s", exc)
        return None, None

    app = FastAPI(title="Janus Daemon", version="1.0.0")

    @app.get("/health")
    def health():
        """Return a simple health check."""
        snap = _get_state_snapshot()
        return JSONResponse({"ok": snap["status"] == "running", "status": snap["status"]})

    @app.get("/status")
    def status():
        """Return full daemon status."""
        return JSONResponse(_get_state_snapshot())

    @app.post("/stop")
    def stop():
        """Request a graceful daemon shutdown."""
        _update_state(stop_requested=True, status="stopping")
        log.info("Stop requested via API.")
        return JSONResponse({"ok": True, "message": "Stop requested."})

    @app.post("/restart")
    def restart():
        """Request a daemon restart (stops current worker; supervisor will restart it)."""
        _update_state(stop_requested=True, status="restarting")
        log.info("Restart requested via API.")
        return JSONResponse({"ok": True, "message": "Restart requested."})

    return app, uvicorn


def _run_api_server() -> None:
    """Run the FastAPI server in a background thread."""
    try:
        from fastapi import FastAPI  # noqa: F401
        import uvicorn
    except Exception as exc:
        log.warning("Skipping API server startup: %s", exc)
        return

    app, _ = _build_app()
    if app is None:
        return

    config = uvicorn.Config(app, host="0.0.0.0", port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    try:
        server.run()
    except Exception as exc:
        log.error("API server error: %s", exc)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _get_script_path() -> str:
    """Return the absolute path to this script (or its compiled exe)."""
    if getattr(sys, "frozen", False):
        return sys.executable
    return str(pathlib.Path(sys.executable).resolve()) + " " + str(pathlib.Path(__file__).resolve())


def install_startup() -> None:
    """Register the daemon in HKCU Run so it starts on Windows login."""
    if not WINREG_AVAILABLE:
        log.error("winreg not available — cannot install startup entry (non-Windows?).")
        return
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, REGISTRY_KEY, 0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, REGISTRY_VALUE, 0, winreg.REG_SZ, _get_script_path() + " --run")
        winreg.CloseKey(key)
        log.info("Startup entry installed: %s", REGISTRY_VALUE)
        print(f"Installed '{REGISTRY_VALUE}' in HKCU\\{REGISTRY_KEY}")
    except Exception as exc:
        log.error("Failed to install startup entry: %s", exc)
        print(f"Error: {exc}")


def uninstall_startup() -> None:
    """Remove the daemon from HKCU Run."""
    if not WINREG_AVAILABLE:
        log.error("winreg not available — cannot uninstall startup entry.")
        return
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, REGISTRY_KEY, 0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, REGISTRY_VALUE)
        winreg.CloseKey(key)
        log.info("Startup entry removed: %s", REGISTRY_VALUE)
        print(f"Removed '{REGISTRY_VALUE}' from HKCU\\{REGISTRY_KEY}")
    except Exception as exc:
        log.error("Failed to remove startup entry: %s", exc)
        print(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Worker import helper
# ---------------------------------------------------------------------------

def _load_worker():
    """
    Attempt to import JanusAutonomousWorker.
    Returns the class or None if unavailable.
    """
    try:
        from janus_autonomous_worker import JanusAutonomousWorker  # type: ignore
        return JanusAutonomousWorker
    except Exception as exc:
        log.error("Could not import JanusAutonomousWorker: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------

def _run_work_cycle(worker_instance) -> None:
    """
    Run the worker's work_cycle() method.
    Supports both sync and async work_cycle implementations.
    """
    work_cycle = getattr(worker_instance, "work_cycle", None)
    if work_cycle is None:
        raise AttributeError("Worker has no work_cycle() method.")

    if asyncio.iscoroutinefunction(work_cycle):
        asyncio.run(work_cycle())
    else:
        work_cycle()


def _supervisor_loop() -> None:
    """
    Core supervisor: runs self-heal check, then instantiates the worker,
    runs work_cycle(), and handles crashes with rate-limited restarts.
    """
    WorkerClass = _load_worker()
    if WorkerClass is None:
        log.critical("Worker class unavailable. Daemon cannot start work cycle.")
        _update_state(status="error", last_error="Worker class unavailable.")
        return

    # ── Self-heal check before first start ────────────────────────────────
    log.info("Running self-heal check before starting work cycle...")
    _update_state(status="self_healing")

    # Re-apply Defender exclusion on every start (Defender can remove it)
    try:
        import subprocess as _sp
        import pathlib as _pl
        janus_dir = str(_pl.Path(__file__).parent.resolve())
        _sp.run(
            ["powershell", "-Command",
             f"Add-MpPreference -ExclusionPath '{janus_dir}' -ErrorAction SilentlyContinue"],
            capture_output=True, timeout=10,
        )
        log.debug("Defender exclusion refreshed for %s", janus_dir)
    except Exception as exc:
        log.debug("Could not refresh Defender exclusion: %s", exc)
    try:
        import asyncio as _asyncio
        from janus_selfheal import SelfHeal
        healer = SelfHeal()
        healthy = _asyncio.run(healer.run())
        if healthy:
            log.info("Self-heal check passed — system healthy")
        else:
            log.warning(
                "Self-heal check: some tests failed. "
                "Starting in degraded mode — check janus_selfheal_report.json"
            )
    except Exception as exc:
        log.warning("Self-heal check failed to run: %s — continuing anyway", exc)

    restart_timestamps: list[float] = []

    while not _state.get("stop_requested"):
        # Prune restart timestamps older than 1 hour
        now = time.time()
        restart_timestamps = [t for t in restart_timestamps if now - t < RESTART_WINDOW]

        if len(restart_timestamps) >= MAX_RESTARTS_PER_HOUR:
            log.warning(
                "Reached %d restarts in the last hour. Pausing for 1 hour.",
                MAX_RESTARTS_PER_HOUR,
            )
            _update_state(status="paused_rate_limit")
            _write_health_file()
            # Sleep in small increments so we can respond to stop requests
            for _ in range(360):  # 360 × 10s = 1 hour
                if _state.get("stop_requested"):
                    break
                time.sleep(10)
            restart_timestamps.clear()
            continue

        try:
            log.info("Starting JanusAutonomousWorker.work_cycle() ...")
            _update_state(status="running", last_error="")
            worker = WorkerClass()

            # Sync worker state from worker attributes if available
            def _sync_worker_state():
                """Pull mood/energy/cycle_count from worker into daemon state."""
                while not _state.get("stop_requested"):
                    try:
                        mood = getattr(worker, "current_mood", None) or getattr(
                            worker, "mood", _state["current_mood"]
                        )
                        energy = getattr(worker, "energy", _state["current_energy"])
                        cycles = getattr(worker, "cycle_count", _state["cycle_count"])
                        _update_state(
                            current_mood=str(mood),
                            current_energy=float(energy),
                            cycle_count=int(cycles),
                        )
                    except Exception:
                        pass
                    time.sleep(5)

            sync_thread = threading.Thread(target=_sync_worker_state, daemon=True)
            sync_thread.start()

            _run_work_cycle(worker)
            log.info("work_cycle() returned normally.")

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            log.error("work_cycle() crashed: %s", error_msg)
            _update_state(
                status="crashed",
                last_error=error_msg,
                restart_count=_state["restart_count"] + 1,
            )
            _write_health_file()

            if _state.get("stop_requested"):
                break

            restart_timestamps.append(time.time())
            log.info("Waiting %d seconds before restart ...", RESTART_DELAY)
            for _ in range(RESTART_DELAY):
                if _state.get("stop_requested"):
                    break
                time.sleep(1)

    _update_state(status="stopped")
    log.info("Supervisor loop exited.")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame) -> None:
    """Handle SIGINT / SIGTERM for graceful shutdown."""
    log.info("Signal %s received — requesting stop.", signum)
    _update_state(stop_requested=True, status="stopping")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_daemon() -> None:
    """Start the Janus daemon: health writer, API server, and supervisor loop."""
    log.info("Janus daemon starting (PID %d).", os.getpid())
    _update_state(status="starting", start_time=time.time())

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass  # SIGTERM not available on all platforms

    # Health writer thread
    health_thread = threading.Thread(target=_health_writer_loop, daemon=True, name="health-writer")
    health_thread.start()

    # API server thread
    api_thread = threading.Thread(target=_run_api_server, daemon=True, name="api-server")
    api_thread.start()

    # Supervisor (blocking)
    _supervisor_loop()

    log.info("Janus daemon stopped.")
    _write_health_file()


def print_status() -> None:
    """Print the contents of janus_health.json to stdout."""
    if HEALTH_FILE.exists():
        try:
            data = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
            print(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"Error reading health file: {exc}")
    else:
        print("Health file not found. Is the daemon running?")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate action."""
    parser = argparse.ArgumentParser(
        description="Janus Daemon — Windows process supervisor for JanusAutonomousWorker"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--install",
        action="store_true",
        help="Register daemon in Windows startup (HKCU Run registry key).",
    )
    group.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove daemon from Windows startup registry.",
    )
    group.add_argument(
        "--run",
        action="store_true",
        help="Start the daemon (supervisor + health writer + API server).",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print the current health file to stdout.",
    )
    group.add_argument(
        "--selfheal",
        action="store_true",
        help="Run the self-heal check and exit (tests + auto-repair).",
    )

    args = parser.parse_args()

    if args.install:
        install_startup()
    elif args.uninstall:
        uninstall_startup()
    elif args.run:
        run_daemon()
    elif args.status:
        print_status()
    elif args.selfheal:
        import asyncio as _asyncio
        from janus_selfheal import SelfHeal
        healer = SelfHeal()
        healthy = _asyncio.run(healer.run())
        print_status()
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
