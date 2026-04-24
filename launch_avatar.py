"""
launch_avatar.py
================
Starts a local HTTP server and opens the Janus avatar in the browser.
Must be run from the Janus workspace directory.

Usage:
    python launch_avatar.py
    python launch_avatar.py --port 8765
"""
import argparse
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

WORKSPACE = Path(__file__).parent
DEFAULT_PORT = 8765


class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses the per-request log spam."""

    def log_message(self, fmt, *args):
        # Only print errors (4xx/5xx)
        code = args[1] if len(args) > 1 else "000"
        if str(code).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def end_headers(self):
        # Allow any origin (needed for Three.js importmap in some browsers)
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def serve(port: int):
    os.chdir(WORKSPACE)
    server = HTTPServer(("127.0.0.1", port), QuietHandler)
    print(f"[avatar] Serving workspace at http://127.0.0.1:{port}/")
    print(f"[avatar] Press Ctrl+C to stop.")
    server.serve_forever()


def open_browser(port: int, delay: float = 0.8):
    time.sleep(delay)
    url = f"http://127.0.0.1:{port}/janus_avatar.html"
    print(f"[avatar] Opening {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Launch Janus avatar viewer")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Start server without opening the browser")
    args = parser.parse_args()

    avatar_html = WORKSPACE / "janus_avatar.html"
    if not avatar_html.exists():
        print(f"[avatar] ERROR: {avatar_html} not found.")
        sys.exit(1)

    if not args.no_browser:
        t = threading.Thread(target=open_browser, args=(args.port,), daemon=True)
        t.start()

    try:
        serve(args.port)
    except KeyboardInterrupt:
        print("\n[avatar] Stopped.")


if __name__ == "__main__":
    main()
