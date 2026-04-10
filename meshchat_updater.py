"""
meshchat_updater.py
====================
Lets Janus update MeshChat autonomously.

When Janus modifies MeshChat.tsx or any dashboard file, it calls
bump_version() to increment the version number. The next time
your phone checks (every 30 min), it sees the new version and
shows the "Update available" banner.

Usage:
    from meshchat_updater import bump_version, set_changelog

    # After making changes to MeshChat
    bump_version(changelog="Added dark mode and voice message waveforms")
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_VERSION_FILE = Path(__file__).parent / "mesh_isp_dashboard" / "meshchat-version.json"
_SW_FILE      = Path(__file__).parent / "mesh_isp_dashboard" / "meshchat-sw.js"
_DASHBOARD_DIR = Path(__file__).parent / "mesh_isp_dashboard"


def get_version() -> dict:
    if _VERSION_FILE.exists():
        return json.loads(_VERSION_FILE.read_text())
    return {"version": "1.0.0", "buildTime": datetime.now(timezone.utc).isoformat(), "changelog": ""}


def bump_version(changelog: str = "", bump: str = "patch") -> str:
    """
    Increment the version number and update the version file.
    bump: "patch" (1.0.0→1.0.1), "minor" (1.0.0→1.1.0), "major" (1.0.0→2.0.0)
    Returns the new version string.
    """
    data    = get_version()
    current = data.get("version", "1.0.0")
    parts   = [int(x) for x in current.split(".")]

    if bump == "major":
        parts = [parts[0] + 1, 0, 0]
    elif bump == "minor":
        parts = [parts[0], parts[1] + 1, 0]
    else:
        parts = [parts[0], parts[1], parts[2] + 1]

    new_version = ".".join(str(p) for p in parts)

    data["version"]   = new_version
    data["buildTime"] = datetime.now(timezone.utc).isoformat()
    data["changelog"] = changelog or f"Updated at {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    _VERSION_FILE.write_text(json.dumps(data, indent=2))

    # Also bump the service worker cache name so iOS picks up the new version
    _bump_sw_cache(new_version)

    print(f"[MeshChat] Version bumped: {current} → {new_version}")
    if changelog:
        print(f"[MeshChat] Changelog: {changelog}")

    return new_version


def _bump_sw_cache(version: str):
    """Update the cache name in the service worker to force cache invalidation."""
    if not _SW_FILE.exists():
        return
    content = _SW_FILE.read_text()
    new_content = re.sub(
        r'const CACHE_NAME = "meshchat-[^"]+";',
        f'const CACHE_NAME = "meshchat-{version}";',
        content,
    )
    if new_content != content:
        _SW_FILE.write_text(new_content)


def rebuild_dashboard() -> bool:
    """
    Rebuild the dashboard after changes.
    Returns True if build succeeded.
    """
    try:
        result = subprocess.run(
            ["pnpm", "build"],
            cwd=str(_DASHBOARD_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("[MeshChat] Dashboard rebuilt successfully")
            return True
        else:
            print(f"[MeshChat] Build failed:\n{result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[MeshChat] Build error: {e}")
        return False


def update_and_deploy(changelog: str, bump: str = "patch") -> bool:
    """
    Bump version and rebuild in one call.
    Janus calls this after making code changes.
    """
    new_version = bump_version(changelog=changelog, bump=bump)
    ok = rebuild_dashboard()
    if ok:
        print(f"[MeshChat] v{new_version} deployed. Phone will see update banner within 30 min.")
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MeshChat Updater")
    parser.add_argument("--bump",      type=str, default="patch",
                        choices=["patch", "minor", "major"])
    parser.add_argument("--changelog", type=str, default="")
    parser.add_argument("--build",     action="store_true", help="Also rebuild dashboard")
    parser.add_argument("--version",   action="store_true", help="Show current version")
    args = parser.parse_args()

    if args.version:
        v = get_version()
        print(f"Current version: {v['version']}")
        print(f"Built: {v['buildTime']}")
        print(f"Changelog: {v['changelog']}")
    elif args.build:
        update_and_deploy(args.changelog, args.bump)
    else:
        bump_version(args.changelog, args.bump)
