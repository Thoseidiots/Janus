"""
Audit logging for the Janus Dependency Analyzer.

Provides comprehensive audit logging of all system access attempts,
permission events, sensitive path skips, and encryption operations
for security review.

Requirement 9.4: The System_Scanner SHALL provide audit logs of all
system access attempts for security review.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Comprehensive audit logger for system access attempts.

    Requirement 9.4: The System_Scanner SHALL provide audit logs of all
    system access attempts for security review.

    Each audit entry is a dict with:
        - timestamp (str): ISO-format datetime string
        - event_type (str): Type of event (e.g. "scan_attempt", "permission_denied")
        - path (str): File system path involved in the event
        - success (bool): Whether the operation succeeded
        - details (str): Additional context about the event
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        logger_name: str = "janus.audit",
    ) -> None:
        """
        Initialise the AuditLogger.

        Args:
            log_file: Optional path to write audit log file. If None, logs to
                      Python logging only.
            logger_name: Logger name for Python logging integration.
        """
        self._log: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(logger_name)
        self._log_file: Optional[Path] = log_file

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_scan_attempt(self, path: str, success: bool, reason: str = "") -> None:
        """
        Log a scan access attempt.

        Args:
            path: The file system path that was accessed.
            success: Whether the scan attempt succeeded.
            reason: Optional reason or context for the attempt.
        """
        entry = self._make_entry(
            event_type="scan_attempt",
            path=path,
            success=success,
            details=reason,
        )
        self._record(entry)

    def log_permission_denied(self, path: str, error: str = "") -> None:
        """
        Log a permission denied event.

        Args:
            path: The file system path that was denied.
            error: Optional error message from the OS.
        """
        entry = self._make_entry(
            event_type="permission_denied",
            path=path,
            success=False,
            details=error,
        )
        self._record(entry)

    def log_sensitive_path_skipped(self, path: str, pattern: str = "") -> None:
        """
        Log that a sensitive path was skipped.

        Args:
            path: The sensitive path that was skipped.
            pattern: The pattern that matched the path.
        """
        details = f"matched pattern: {pattern}" if pattern else ""
        entry = self._make_entry(
            event_type="sensitive_path_skipped",
            path=path,
            success=True,
            details=details,
        )
        self._record(entry)

    def log_encryption_event(
        self, operation: str, field: str, success: bool
    ) -> None:
        """
        Log an encryption or decryption event.

        Args:
            operation: The operation performed (e.g. "encrypt", "decrypt").
            field: The field name that was encrypted/decrypted.
            success: Whether the operation succeeded.
        """
        entry = self._make_entry(
            event_type="encryption",
            path="",
            success=success,
            details=f"operation={operation} field={field}",
        )
        self._record(entry)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Return all audit log entries as a list of dicts.

        Returns:
            A copy of all audit entries recorded so far.
        """
        return list(self._log)

    def get_recent_events(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Return audit events since the given datetime.

        Args:
            since: Only events with a timestamp >= this value are returned.

        Returns:
            Filtered list of audit entries.
        """
        result = []
        for entry in self._log:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= since:
                    result.append(entry)
            except (KeyError, ValueError):
                pass
        return result

    # ------------------------------------------------------------------
    # Export / clear
    # ------------------------------------------------------------------

    def export_audit_log(self, output_path: Path, format: str = "json") -> None:
        """
        Export the audit log to a JSON or CSV file.

        Args:
            output_path: Destination file path.
            format: "json" or "csv" (case-insensitive).

        Raises:
            ValueError: If the format is not supported.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = format.lower()

        if fmt == "json":
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(self._log, fh, indent=2, default=str)
        elif fmt == "csv":
            fieldnames = ["timestamp", "event_type", "path", "success", "details"]
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self._log:
                    writer.writerow({k: entry.get(k, "") for k in fieldnames})
        else:
            raise ValueError(
                f"Unsupported audit log format: '{format}'. "
                "Supported formats are: json, csv"
            )

    def clear(self) -> None:
        """Clear the in-memory audit log."""
        self._log.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        event_type: str,
        path: str,
        success: bool,
        details: str,
    ) -> Dict[str, Any]:
        """Build a standardised audit entry dict."""
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "path": path,
            "success": success,
            "details": details,
        }

    def _record(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the in-memory log and emit a Python log message."""
        self._log.append(entry)
        level = logging.INFO if entry["success"] else logging.WARNING
        self._logger.log(
            level,
            "AUDIT event_type=%s path=%r success=%s details=%r",
            entry["event_type"],
            entry["path"],
            entry["success"],
            entry["details"],
        )
