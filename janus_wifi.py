"""
janus_wifi.py
=============
Janus WiFi hotspot manager for Windows.

Uses the TP-Link USB adapter (cc:ba:bd:8b:71:74) to broadcast
a WiFi hotspot called "Janus-Net" while staying connected to FiOS.

Windows Hosted Network allows a single WiFi adapter to simultaneously
connect to an upstream network AND broadcast a hotspot — the adapter
handles both roles via virtual interfaces.

Usage:
    from janus_wifi import JanusWifi
    wifi = JanusWifi()
    wifi.start()          # Start Janus-Net hotspot
    wifi.stop()           # Stop hotspot
    wifi.status()         # Get current status
    wifi.auto_manage()    # Start + monitor in background
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("janus_wifi")

# ── Config ────────────────────────────────────────────────────────────────────

HOTSPOT_SSID     = "Janus-Net"
HOTSPOT_PASSWORD = "JanusAI2025"   # Change this to something personal
ADAPTER_MAC      = "cc:ba:bd:8b:71:74"  # TP-Link adapter
MONITOR_INTERVAL = 30   # seconds between health checks


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ConnectedDevice:
    mac_address: str
    ip_address: str = ""
    hostname: str = ""
    first_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class HotspotStatus:
    running: bool
    ssid: str
    connected_devices: int
    uptime_seconds: float
    last_check: str
    error: Optional[str] = None


# ── Core manager ──────────────────────────────────────────────────────────────

class JanusWifi:
    """
    Manages the Janus-Net WiFi hotspot using Windows netsh commands.
    Requires admin privileges to start/stop the hosted network.
    """

    def __init__(self, ssid: str = HOTSPOT_SSID, password: str = HOTSPOT_PASSWORD) -> None:
        self.ssid = ssid
        self.password = password
        self._running = False
        self._start_time: Optional[float] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._devices: List[ConnectedDevice] = []

    # ── netsh helpers ─────────────────────────────────────────────────────────

    def _run(self, cmd: str, check: bool = False) -> tuple[int, str, str]:
        """Run a netsh command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as exc:
            return 1, "", str(exc)

    def _is_admin(self) -> bool:
        """Check if running with admin privileges."""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def is_supported(self) -> bool:
        """Check if Windows Hosted Network is supported on this machine."""
        code, out, _ = self._run("netsh wlan show drivers")
        return "Hosted network supported  : Yes" in out or "Yes" in out

    def start(self) -> dict:
        """
        Configure and start the Janus-Net hotspot.
        Returns {"success": bool, "message": str}
        """
        if not self._is_admin():
            return {
                "success": False,
                "message": "Admin privileges required. Run as administrator.",
            }

        if not self.is_supported():
            return {
                "success": False,
                "message": "WiFi adapter does not support hosted network mode.",
            }

        logger.info("Starting Janus-Net hotspot (SSID: %s)...", self.ssid)

        # Configure the hosted network
        code, out, err = self._run(
            f'netsh wlan set hostednetwork mode=allow ssid="{self.ssid}" key="{self.password}"'
        )
        if code != 0:
            return {"success": False, "message": f"Failed to configure hotspot: {err}"}

        # Start it
        code, out, err = self._run("netsh wlan start hostednetwork")
        if code != 0:
            return {"success": False, "message": f"Failed to start hotspot: {err or out}"}

        self._running = True
        self._start_time = time.time()
        logger.info("Janus-Net hotspot started successfully")

        return {
            "success": True,
            "message": f'Janus-Net hotspot "{self.ssid}" is now broadcasting. Password: {self.password}',
        }

    def stop(self) -> dict:
        """Stop the hotspot."""
        logger.info("Stopping Janus-Net hotspot...")
        self._stop_monitor.set()

        code, out, err = self._run("netsh wlan stop hostednetwork")
        self._running = False
        self._start_time = None

        if code != 0:
            return {"success": False, "message": f"Failed to stop hotspot: {err or out}"}

        logger.info("Janus-Net hotspot stopped")
        return {"success": True, "message": "Janus-Net hotspot stopped"}

    def status(self) -> HotspotStatus:
        """Get current hotspot status."""
        code, out, err = self._run("netsh wlan show hostednetwork")

        running = False
        connected = 0

        if code == 0:
            running = "Status" in out and "Started" in out
            # Count connected clients
            import re
            clients_match = re.search(r"Number of clients\s*:\s*(\d+)", out)
            if clients_match:
                connected = int(clients_match.group(1))

        uptime = time.time() - self._start_time if self._start_time else 0.0

        return HotspotStatus(
            running=running,
            ssid=self.ssid,
            connected_devices=connected,
            uptime_seconds=round(uptime, 1),
            last_check=datetime.utcnow().isoformat(),
            error=err if code != 0 else None,
        )

    def get_connected_devices(self) -> List[ConnectedDevice]:
        """Get list of connected devices via ARP table."""
        devices = []
        try:
            code, out, _ = self._run("arp -a")
            if code == 0:
                import re
                # Parse ARP table — look for entries in the 192.168.137.x range
                # (Windows default hosted network subnet)
                for line in out.splitlines():
                    match = re.search(
                        r"(192\.168\.137\.\d+)\s+([\da-f-]+)\s+dynamic", line, re.IGNORECASE
                    )
                    if match:
                        ip = match.group(1)
                        mac = match.group(2).replace("-", ":").lower()
                        devices.append(ConnectedDevice(mac_address=mac, ip_address=ip))
        except Exception as exc:
            logger.warning("Failed to get connected devices: %s", exc)
        return devices

    def restart_if_down(self) -> bool:
        """Check if hotspot is running; restart if not. Returns True if action taken."""
        s = self.status()
        if not s.running and self._running:
            logger.warning("Janus-Net hotspot went down — restarting...")
            result = self.start()
            if result["success"]:
                logger.info("Janus-Net hotspot restarted successfully")
                return True
            else:
                logger.error("Failed to restart hotspot: %s", result["message"])
        return False

    def auto_manage(self) -> None:
        """
        Start the hotspot and launch a background monitor thread
        that restarts it if it goes down.
        """
        result = self.start()
        logger.info(result["message"])

        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="janus-wifi-monitor"
        )
        self._monitor_thread.start()
        logger.info("Janus-Net auto-management started (checking every %ds)", MONITOR_INTERVAL)

    def _monitor_loop(self) -> None:
        """Background thread: periodically checks hotspot health."""
        while not self._stop_monitor.wait(MONITOR_INTERVAL):
            try:
                self.restart_if_down()
                devices = self.get_connected_devices()
                if devices:
                    logger.info(
                        "Janus-Net: %d device(s) connected: %s",
                        len(devices),
                        [d.ip_address for d in devices],
                    )
            except Exception as exc:
                logger.error("Janus-Net monitor error: %s", exc)

    def to_dict(self) -> dict:
        """Return status as a JSON-serialisable dict (for the daemon API)."""
        s = self.status()
        return {
            "ssid": s.ssid,
            "running": s.running,
            "connected_devices": s.connected_devices,
            "uptime_seconds": s.uptime_seconds,
            "last_check": s.last_check,
            "password": self.password,
            "adapter_mac": ADAPTER_MAC,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_wifi_instance: Optional[JanusWifi] = None


def get_wifi() -> JanusWifi:
    global _wifi_instance
    if _wifi_instance is None:
        _wifi_instance = JanusWifi()
    return _wifi_instance


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Janus WiFi Hotspot Manager")
    parser.add_argument("--start",   action="store_true", help="Start Janus-Net hotspot")
    parser.add_argument("--stop",    action="store_true", help="Stop hotspot")
    parser.add_argument("--status",  action="store_true", help="Show hotspot status")
    parser.add_argument("--devices", action="store_true", help="List connected devices")
    parser.add_argument("--manage",  action="store_true", help="Start + auto-manage (blocking)")
    parser.add_argument("--ssid",    type=str, default=HOTSPOT_SSID, help="Custom SSID")
    parser.add_argument("--password",type=str, default=HOTSPOT_PASSWORD, help="Custom password")
    args = parser.parse_args()

    wifi = JanusWifi(ssid=args.ssid, password=args.password)

    if args.start:
        result = wifi.start()
        print(result["message"])

    elif args.stop:
        result = wifi.stop()
        print(result["message"])

    elif args.status:
        s = wifi.status()
        print(json.dumps(vars(s), indent=2))

    elif args.devices:
        devices = wifi.get_connected_devices()
        if devices:
            for d in devices:
                print(f"  {d.ip_address}  {d.mac_address}")
        else:
            print("No devices connected")

    elif args.manage:
        print(f"Starting Janus-Net auto-management...")
        wifi.auto_manage()
        print("Monitoring... Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            wifi.stop()
            print("Stopped.")

    else:
        parser.print_help()
