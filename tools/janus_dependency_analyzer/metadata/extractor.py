"""
Application metadata extraction for the Janus Dependency Analyzer.

Provides cross-platform extraction of application metadata including
file size, version information, and digital signature verification.
"""

import hashlib
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.models import ApplicationMetadata, Platform

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts metadata from application executables across platforms.

    Handles file size, version info (vendor, description), digital
    signature verification, and stable unique ID generation.
    """

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def extract(self, app_path: Path, platform: Platform) -> ApplicationMetadata:
        """
        Extract full metadata for an application.

        Args:
            app_path: Path to the application executable.
            platform: The platform the application runs on.

        Returns:
            ApplicationMetadata populated with all available information.
        """
        metadata = ApplicationMetadata()

        # File size — always attempt
        metadata.file_size = self.extract_file_size(app_path)

        # Install date from filesystem ctime
        try:
            if app_path.exists():
                stat = app_path.stat()
                metadata.install_date = datetime.fromtimestamp(stat.st_ctime)
        except (OSError, PermissionError) as exc:
            logger.warning("Could not read ctime for %s: %s", app_path, exc)

        # Version info (vendor, description)
        version_info = self.extract_version_info(app_path, platform)
        metadata.vendor = version_info.get("vendor")
        metadata.description = version_info.get("description")

        # Digital signature
        metadata.digital_signature = self.verify_digital_signature(app_path, platform)

        return metadata

    def extract_file_size(self, path: Path) -> int:
        """
        Return the file size in bytes, or 0 if the file is inaccessible.

        Args:
            path: Path to the file.

        Returns:
            File size in bytes, or 0 on error.
        """
        try:
            return path.stat().st_size
        except (OSError, PermissionError, FileNotFoundError) as exc:
            logger.debug("Could not get file size for %s: %s", path, exc)
            return 0

    def extract_version_info(self, exe_path: Path, platform: Platform) -> dict:
        """
        Extract version information (description, vendor, version) from an executable.

        On Windows, queries the file's VersionInfo via PowerShell.
        On other platforms, returns an empty dict (no portable mechanism).

        Args:
            exe_path: Path to the executable.
            platform: Target platform.

        Returns:
            Dict with optional keys: 'vendor', 'description', 'version'.
        """
        if platform != Platform.WINDOWS:
            return {}

        if not exe_path.exists():
            return {}

        try:
            ps_script = (
                "$info = (Get-Item -LiteralPath '{path}').VersionInfo; "
                "Write-Output ($info.CompanyName + '|' + $info.FileDescription + '|' + $info.FileVersion)"
            ).format(path=str(exe_path).replace("'", "''"))

            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2,
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|")
                info: dict = {}
                if len(parts) >= 1 and parts[0].strip():
                    info["vendor"] = parts[0].strip()
                if len(parts) >= 2 and parts[1].strip():
                    info["description"] = parts[1].strip()
                if len(parts) >= 3 and parts[2].strip():
                    info["version"] = parts[2].strip()
                return info

        except subprocess.TimeoutExpired:
            logger.debug("Timeout extracting version info for %s", exe_path)
        except FileNotFoundError:
            logger.debug("PowerShell not found; cannot extract version info")
        except Exception as exc:
            logger.debug("Error extracting version info for %s: %s", exe_path, exc)

        return {}

    def verify_digital_signature(
        self, exe_path: Path, platform: Platform
    ) -> Optional[str]:
        """
        Verify the digital signature of an executable.

        On Windows, uses PowerShell's Get-AuthenticodeSignature.
        On other platforms, returns None (not supported).

        Args:
            exe_path: Path to the executable.
            platform: Target platform.

        Returns:
            'Valid', 'Invalid', or None if verification is not possible.
        """
        if platform != Platform.WINDOWS:
            return None

        if not exe_path.exists():
            return None

        try:
            ps_script = (
                "$sig = Get-AuthenticodeSignature -LiteralPath '{path}'; "
                "Write-Output $sig.Status"
            ).format(path=str(exe_path).replace("'", "''"))

            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2,
            )

            if result.returncode == 0:
                status = result.stdout.strip()
                if status == "Valid":
                    return "Valid"
                elif status in ("HashMismatch", "NotSigned", "UnknownError",
                                "NotTrusted", "Incompatible"):
                    return "Invalid"

        except subprocess.TimeoutExpired:
            logger.debug("Timeout verifying signature for %s", exe_path)
        except FileNotFoundError:
            logger.debug("PowerShell not found; cannot verify signature")
        except Exception as exc:
            logger.debug("Error verifying signature for %s: %s", exe_path, exc)

        return None

    def generate_app_id(
        self, name: str, exe_path: Path, platform: Platform
    ) -> str:
        """
        Generate a stable, unique 16-character hex ID for an application.

        The ID is derived from the platform, lowercased name, and lowercased
        executable path, so the same application always gets the same ID.

        Args:
            name: Application name.
            exe_path: Path to the executable.
            platform: Target platform.

        Returns:
            First 16 hex characters of the SHA-256 hash of the key.
        """
        key = f"{platform.value}:{name.lower()}:{str(exe_path).lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
