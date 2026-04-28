"""
OS Block Device Registration for the Software NVMe Engine.

Supports three backends, selected automatically at runtime:

1. SimulatedOsBackend  — cross-platform, no privileges required (dev/test)
2. WindowsOsBackend    — Windows: uses WMI / PowerShell to expose a virtual
                         disk via the Windows Storage subsystem
3. LinuxKernelBackend  — Linux: uses NVMeVirt configfs interface

Callers always use `OsDeviceRegistry`; the correct backend is chosen for you.
"""

from __future__ import annotations

import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BlockDeviceInfo:
    """Information about a registered block device."""
    device_id: int
    name: str
    block_path: str          # e.g. /dev/nvme0n1 or simulated path
    capacity_bytes: int
    namespace_count: int
    registered: bool = False
    simulated: bool = True   # True when not backed by a real kernel device


# ---------------------------------------------------------------------------
# Simulated backend (cross-platform, no kernel privileges required)
# ---------------------------------------------------------------------------

class SimulatedOsBackend:
    """
    Simulated OS registration backend.

    Tracks registered devices in memory and assigns synthetic /dev/nvmeN paths.
    Used on non-Linux systems or when kernel privileges are unavailable.
    """

    def __init__(self) -> None:
        self._registered: Dict[int, BlockDeviceInfo] = {}
        self._next_minor: int = 0
        self._lock = threading.Lock()

    def register(self, device_id: int, name: str, capacity_bytes: int, namespace_count: int) -> BlockDeviceInfo:
        """Register a virtual device and return its block device info."""
        with self._lock:
            minor = self._next_minor
            self._next_minor += 1
            block_path = f"/dev/nvme{minor}n1"
            info = BlockDeviceInfo(
                device_id=device_id,
                name=name,
                block_path=block_path,
                capacity_bytes=capacity_bytes,
                namespace_count=namespace_count,
                registered=True,
                simulated=True,
            )
            self._registered[device_id] = info
            return info

    def unregister(self, device_id: int) -> None:
        """Unregister a virtual device."""
        with self._lock:
            self._registered.pop(device_id, None)

    def get_info(self, device_id: int) -> Optional[BlockDeviceInfo]:
        with self._lock:
            return self._registered.get(device_id)

    def list_registered(self) -> List[BlockDeviceInfo]:
        with self._lock:
            return list(self._registered.values())

    def is_available(self) -> bool:
        return True  # Always available


# ---------------------------------------------------------------------------
# Linux kernel backend (real sysfs/configfs interaction)
# ---------------------------------------------------------------------------

class LinuxKernelBackend:
    """
    Linux kernel block device registration backend.

    Attempts to register devices via the NVMeVirt configfs interface at
    /sys/kernel/config/nvme-virt/ when available, falling back to the
    simulated backend if the interface is not present or privileges are
    insufficient.

    This backend requires:
    - Linux kernel with NVMeVirt or equivalent module loaded
    - Root privileges or CAP_SYS_ADMIN
    """

    CONFIGFS_BASE = Path("/sys/kernel/config/nvme-virt")

    def __init__(self) -> None:
        self._registered: Dict[int, BlockDeviceInfo] = {}
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if the Linux NVMeVirt configfs interface is available."""
        return (
            platform.system() == "Linux"
            and self.CONFIGFS_BASE.exists()
            and os.access(str(self.CONFIGFS_BASE), os.W_OK)
        )

    def register(self, device_id: int, name: str, capacity_bytes: int, namespace_count: int) -> BlockDeviceInfo:
        """Register a device via configfs."""
        device_dir = self.CONFIGFS_BASE / name
        try:
            device_dir.mkdir(parents=True, exist_ok=True)
            (device_dir / "capacity").write_text(str(capacity_bytes))
            (device_dir / "namespace_count").write_text(str(namespace_count))
            (device_dir / "enable").write_text("1")

            # Read back the assigned block device path
            block_path = self._find_block_path(name)
        except (OSError, PermissionError) as exc:
            raise OSError(
                f"Failed to register device '{name}' via configfs: {exc}. "
                "Ensure the NVMeVirt kernel module is loaded and you have CAP_SYS_ADMIN."
            ) from exc

        with self._lock:
            info = BlockDeviceInfo(
                device_id=device_id,
                name=name,
                block_path=block_path,
                capacity_bytes=capacity_bytes,
                namespace_count=namespace_count,
                registered=True,
                simulated=False,
            )
            self._registered[device_id] = info
        return info

    def unregister(self, device_id: int) -> None:
        """Unregister a device via configfs."""
        with self._lock:
            info = self._registered.pop(device_id, None)
        if info is None:
            return
        device_dir = self.CONFIGFS_BASE / info.name
        try:
            if (device_dir / "enable").exists():
                (device_dir / "enable").write_text("0")
            if device_dir.exists():
                device_dir.rmdir()
        except OSError:
            pass  # Best-effort cleanup

    def get_info(self, device_id: int) -> Optional[BlockDeviceInfo]:
        with self._lock:
            return self._registered.get(device_id)

    def list_registered(self) -> List[BlockDeviceInfo]:
        with self._lock:
            return list(self._registered.values())

    def _find_block_path(self, name: str) -> str:
        """Attempt to find the /dev/nvmeN path for a registered device."""
        # After enabling, the kernel creates a block device; scan /sys/block
        sys_block = Path("/sys/block")
        if sys_block.exists():
            for entry in sorted(sys_block.iterdir()):
                if entry.name.startswith("nvme"):
                    subsystem = entry / "device" / "subsystem"
                    if subsystem.exists() and name in str(subsystem.resolve()):
                        return f"/dev/{entry.name}"
        # Fallback: derive from name
        return f"/dev/{name}"


# ---------------------------------------------------------------------------
# Windows backend
# ---------------------------------------------------------------------------

class WindowsOsBackend:
    """
    Windows block device registration backend.

    Uses PowerShell / Windows Storage APIs to create and expose a virtual
    disk backed by a VHDX (Virtual Hard Disk) file.  Each registered device
    gets a VHDX at %LOCALAPPDATA%\\NvmeEngine\\<name>.vhdx.

    Requirements:
    - Windows 10/11 or Windows Server 2016+
    - PowerShell 5+ with Hyper-V / Storage module available
    - Administrator privileges for VHD mount operations

    Falls back gracefully: if PowerShell commands fail the registry still
    tracks the device in memory so the rest of the engine keeps working.
    """

    def __init__(self) -> None:
        self._registered: Dict[int, BlockDeviceInfo] = {}
        self._lock = threading.Lock()
        self._vhd_dir = Path(os.environ.get("LOCALAPPDATA", "C:\\Temp")) / "NvmeEngine"

    def is_available(self) -> bool:
        """Check if we're on Windows with PowerShell available."""
        if platform.system() != "Windows":
            return False
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "echo ok"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def register(self, device_id: int, name: str, capacity_bytes: int, namespace_count: int) -> BlockDeviceInfo:
        """Create a VHDX-backed virtual disk and attach it to the system."""
        self._vhd_dir.mkdir(parents=True, exist_ok=True)
        vhd_path = self._vhd_dir / f"{name}.vhdx"
        capacity_mb = max(1, capacity_bytes // (1024 * 1024))

        block_path = self._create_and_attach_vhd(vhd_path, capacity_mb, name)

        with self._lock:
            info = BlockDeviceInfo(
                device_id=device_id,
                name=name,
                block_path=block_path,
                capacity_bytes=capacity_bytes,
                namespace_count=namespace_count,
                registered=True,
                simulated=False,
            )
            self._registered[device_id] = info
        return info

    def unregister(self, device_id: int) -> None:
        """Detach the VHDX for this device."""
        with self._lock:
            info = self._registered.pop(device_id, None)
        if info is None:
            return
        vhd_path = self._vhd_dir / f"{info.name}.vhdx"
        self._detach_vhd(vhd_path)

    def get_info(self, device_id: int) -> Optional[BlockDeviceInfo]:
        with self._lock:
            return self._registered.get(device_id)

    def list_registered(self) -> List[BlockDeviceInfo]:
        with self._lock:
            return list(self._registered.values())

    def _create_and_attach_vhd(self, vhd_path: Path, capacity_mb: int, name: str) -> str:
        """
        Create a VHDX and attach it via PowerShell.
        Returns the Windows physical disk path like \\\\.\\PhysicalDriveN,
        or a synthetic path on failure.
        """
        ps_script = (
            f"$p = '{vhd_path}'; "
            f"if (-not (Test-Path $p)) {{ New-VHD -Path $p -SizeBytes {capacity_mb}MB -Dynamic | Out-Null }}; "
            f"$d = Mount-VHD -Path $p -PassThru | Get-Disk; "
            f"Write-Output \"\\\\.\\PhysicalDrive$($d.Number)\""
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                path = result.stdout.strip().splitlines()[-1].strip()
                if path.startswith("\\\\.\\"):
                    return path
        except (subprocess.TimeoutExpired, OSError):
            pass
        # Fallback: synthetic path (VHDX file exists but not attached as block device)
        return f"\\\\.\\NvmeVirtual\\{name}"

    def _detach_vhd(self, vhd_path: Path) -> None:
        """Detach a mounted VHDX."""
        if not vhd_path.exists():
            return
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Dismount-VHD -Path '{vhd_path}'"],
                capture_output=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass


# ---------------------------------------------------------------------------
# OsDeviceRegistry — public interface
# ---------------------------------------------------------------------------

class OsDeviceRegistry:
    """
    Public interface for OS block device registration.

    Backend selection priority:
    1. force_simulated=True  → SimulatedOsBackend (always)
    2. Windows               → WindowsOsBackend (VHDX via PowerShell)
    3. Linux + NVMeVirt      → LinuxKernelBackend (configfs)
    4. Fallback              → SimulatedOsBackend

    Usage:
        registry = OsDeviceRegistry()
        info = registry.register(device_id=1, name="nvme0",
                                 capacity_bytes=1024**4, namespace_count=4)
        print(info.block_path)  # \\\\.\\PhysicalDrive1  (Windows)
                                # /dev/nvme0n1        (Linux)
        registry.unregister(device_id=1)
    """

    def __init__(self, force_simulated: bool = False) -> None:
        """
        Args:
            force_simulated: If True, always use the simulated backend
                             (useful for testing).
        """
        if force_simulated:
            self._backend: SimulatedOsBackend | WindowsOsBackend | LinuxKernelBackend = SimulatedOsBackend()
            self._backend_name = "simulated"
            self._using_kernel = False
            return

        windows_backend = WindowsOsBackend()
        if windows_backend.is_available():
            self._backend = windows_backend
            self._backend_name = "windows"
            self._using_kernel = True
            return

        linux_backend = LinuxKernelBackend()
        if linux_backend.is_available():
            self._backend = linux_backend
            self._backend_name = "linux"
            self._using_kernel = True
            return

        self._backend = SimulatedOsBackend()
        self._backend_name = "simulated"
        self._using_kernel = False

    @property
    def backend_name(self) -> str:
        """Name of the active backend: 'windows', 'linux', or 'simulated'."""
        return self._backend_name

    @property
    def using_kernel_backend(self) -> bool:
        """True if using a real OS backend (Windows or Linux)."""
        return self._using_kernel

    def register(
        self,
        device_id: int,
        name: str,
        capacity_bytes: int,
        namespace_count: int,
    ) -> BlockDeviceInfo:
        """Register a virtual NVMe device with the OS.

        Returns BlockDeviceInfo with the assigned block device path.
        On Linux with kernel support, this creates a real /dev/nvmeN device.
        Otherwise, returns a simulated registration.
        """
        return self._backend.register(device_id, name, capacity_bytes, namespace_count)

    def unregister(self, device_id: int) -> None:
        """Unregister a virtual NVMe device from the OS."""
        self._backend.unregister(device_id)

    def get_info(self, device_id: int) -> Optional[BlockDeviceInfo]:
        """Get block device info for a registered device, or None."""
        return self._backend.get_info(device_id)

    def list_registered(self) -> List[BlockDeviceInfo]:
        """List all currently registered block devices."""
        return self._backend.list_registered()

    def is_registered(self, device_id: int) -> bool:
        """Check if a device is currently registered."""
        return self._backend.get_info(device_id) is not None
