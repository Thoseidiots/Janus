"""
Application catalog storage for the Janus Dependency Analyzer.

Provides persistent, indexed storage for discovered applications with
optional encryption of sensitive metadata fields.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import Application, ApplicationMetadata, ChangeRecord, Platform

logger = logging.getLogger(__name__)

# Sensitive fields that are encrypted when encrypt=True
_SENSITIVE_FIELDS = ("vendor", "description", "digital_signature")

# Default storage location
_DEFAULT_CATALOG_PATH = Path.home() / ".janus" / "catalog.json"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _app_to_dict(app: Application) -> dict:
    """Serialise an Application to a JSON-compatible dict."""
    meta = app.metadata
    return {
        "id": app.id,
        "name": app.name,
        "version": app.version,
        "installation_path": str(app.installation_path),
        "executable_path": str(app.executable_path),
        "platform": app.platform.value,
        "metadata": {
            "vendor": meta.vendor,
            "description": meta.description,
            "file_size": meta.file_size,
            "install_date": meta.install_date.isoformat() if meta.install_date else None,
            "digital_signature": meta.digital_signature,
            "dependencies": meta.dependencies,
            "file_associations": meta.file_associations,
            "registry_keys": meta.registry_keys,
            "environment_variables": meta.environment_variables,
        },
        "discovered_at": app.discovered_at.isoformat(),
        "last_analyzed": app.last_analyzed.isoformat() if app.last_analyzed else None,
        "is_accessible": app.is_accessible,
        "access_error": app.access_error,
    }


def _dict_to_app(data: dict) -> Application:
    """Deserialise an Application from a dict."""
    meta_data = data.get("metadata", {})
    install_date = None
    if meta_data.get("install_date"):
        install_date = datetime.fromisoformat(meta_data["install_date"])

    metadata = ApplicationMetadata(
        vendor=meta_data.get("vendor"),
        description=meta_data.get("description"),
        file_size=meta_data.get("file_size", 0),
        install_date=install_date,
        digital_signature=meta_data.get("digital_signature"),
        dependencies=meta_data.get("dependencies", []),
        file_associations=meta_data.get("file_associations", []),
        registry_keys=meta_data.get("registry_keys", []),
        environment_variables=meta_data.get("environment_variables", {}),
    )

    last_analyzed = None
    if data.get("last_analyzed"):
        last_analyzed = datetime.fromisoformat(data["last_analyzed"])

    return Application(
        id=data["id"],
        name=data["name"],
        version=data.get("version", ""),
        installation_path=Path(data.get("installation_path", "")),
        executable_path=Path(data.get("executable_path", "")),
        platform=Platform(data["platform"]),
        metadata=metadata,
        discovered_at=datetime.fromisoformat(data["discovered_at"]),
        last_analyzed=last_analyzed,
        is_accessible=data.get("is_accessible", True),
        access_error=data.get("access_error"),
    )


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _try_import_fernet():
    """Return the Fernet class or None if cryptography is not installed."""
    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
        return Fernet
    except ImportError:
        return None


class _EncryptionManager:
    """Manages Fernet symmetric encryption for sensitive catalog fields."""

    def __init__(self, key_path: Path):
        Fernet = _try_import_fernet()
        if Fernet is None:
            raise ImportError("cryptography package is required for encryption")

        self._Fernet = Fernet
        self._key_path = key_path
        self._fernet = self._load_or_create_key()

    def _load_or_create_key(self):
        if self._key_path.exists():
            key = self._key_path.read_bytes()
        else:
            key = self._Fernet.generate_key()
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            self._key_path.write_bytes(key)
        return self._Fernet(key)

    def encrypt(self, value: str) -> str:
        """Encrypt a string value and return a base64-encoded token."""
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt a base64-encoded token and return the original string."""
        return self._fernet.decrypt(token.encode()).decode()


# ---------------------------------------------------------------------------
# ApplicationCatalog
# ---------------------------------------------------------------------------

class ApplicationCatalog:
    """
    Persistent, indexed catalog of discovered applications.

    Maintains three in-memory indexes for O(1) / O(n) lookups:
    - by ID
    - by lowercase name
    - by platform

    Optionally encrypts sensitive metadata fields (vendor, description,
    digital_signature) before writing to disk.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encrypt: bool = False,
    ):
        """
        Initialise the catalog.

        Args:
            storage_path: Path to the JSON catalog file.
                          Defaults to ~/.janus/catalog.json.
            encrypt: If True, encrypt sensitive fields before storage.
        """
        self._storage_path: Path = storage_path or _DEFAULT_CATALOG_PATH
        self._encrypt = encrypt
        self._encryption_manager: Optional[_EncryptionManager] = None

        # Indexes
        self._by_id: Dict[str, Application] = {}
        self._by_name: Dict[str, List[Application]] = defaultdict(list)
        self._by_platform: Dict[Platform, List[Application]] = defaultdict(list)

        # Change history
        self._change_history: List[ChangeRecord] = []

        if encrypt:
            key_path = self._storage_path.parent / ".catalog.key"
            try:
                self._encryption_manager = _EncryptionManager(key_path)
            except ImportError:
                logger.warning(
                    "cryptography package not installed; "
                    "falling back to unencrypted storage."
                )
                self._encrypt = False

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(self, app: Application) -> None:
        """
        Add or update an application in the catalog.

        If an application with the same ID already exists it is replaced.

        Args:
            app: The application to add.
        """
        existing = self._by_id.get(app.id)
        if existing is not None:
            self._remove_from_indexes(existing)

        self._by_id[app.id] = app
        self._by_name[app.name.lower()].append(app)
        self._by_platform[app.platform].append(app)

    def get(self, app_id: str) -> Optional[Application]:
        """
        Look up an application by its unique ID.

        Args:
            app_id: The application ID.

        Returns:
            The Application, or None if not found.
        """
        return self._by_id.get(app_id)

    def get_by_name(self, name: str) -> List[Application]:
        """
        Find applications by name (case-insensitive).

        Args:
            name: Application name to search for.

        Returns:
            List of matching applications (may be empty).
        """
        return list(self._by_name.get(name.lower(), []))

    def get_by_platform(self, platform: Platform) -> List[Application]:
        """
        Get all applications for a given platform.

        Args:
            platform: The platform to filter by.

        Returns:
            List of applications for that platform.
        """
        return list(self._by_platform.get(platform, []))

    def remove(self, app_id: str) -> bool:
        """
        Remove an application from the catalog.

        Args:
            app_id: The application ID to remove.

        Returns:
            True if the application was found and removed, False otherwise.
        """
        app = self._by_id.pop(app_id, None)
        if app is None:
            return False
        self._remove_from_indexes(app)
        return True

    def all(self) -> List[Application]:
        """Return all applications in the catalog."""
        return list(self._by_id.values())

    def count(self) -> int:
        """Return the total number of applications in the catalog."""
        return len(self._by_id)

    def clear(self) -> None:
        """Remove all applications from the catalog."""
        self._by_id.clear()
        self._by_name.clear()
        self._by_platform.clear()

    # ------------------------------------------------------------------
    # Change history
    # ------------------------------------------------------------------

    def record_change(self, record: ChangeRecord) -> None:
        """Append a change record to the history.

        Args:
            record: The ChangeRecord to append.
        """
        self._change_history.append(record)

    def get_change_history(self, app_id: Optional[str] = None) -> List[ChangeRecord]:
        """Return change history, optionally filtered by app_id.

        Args:
            app_id: If provided, only records for this application are returned.

        Returns:
            List of ChangeRecord objects (may be empty).
        """
        if app_id is None:
            return list(self._change_history)
        return [r for r in self._change_history if r.app_id == app_id]

    def get_recent_changes(self, since: datetime) -> List[ChangeRecord]:
        """Return change records since the given datetime.

        Args:
            since: Only records with timestamp >= this value are returned.

        Returns:
            List of ChangeRecord objects (may be empty).
        """
        return [r for r in self._change_history if r.timestamp >= since]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Persist the catalog to the storage path as JSON.

        Creates parent directories if they do not exist.
        """
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for app in self._by_id.values():
            record = _app_to_dict(app)
            if self._encrypt and self._encryption_manager is not None:
                record = self._encrypt_record(record)
            records.append(record)

        change_history = []
        for cr in self._change_history:
            change_history.append({
                "app_id": cr.app_id,
                "app_name": cr.app_name,
                "change_type": cr.change_type,
                "timestamp": cr.timestamp.isoformat(),
                "previous_version": cr.previous_version,
                "new_version": cr.new_version,
                "details": cr.details,
            })

        with self._storage_path.open("w", encoding="utf-8") as fh:
            json.dump({"applications": records, "change_history": change_history}, fh, indent=2)

        logger.debug("Catalog saved to %s (%d apps)", self._storage_path, len(records))

    def load(self) -> None:
        """
        Load the catalog from the storage path.

        Replaces the current in-memory state. If the file does not exist,
        the catalog is left empty.
        """
        if not self._storage_path.exists():
            logger.debug("Catalog file not found at %s; starting empty", self._storage_path)
            return

        with self._storage_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.clear()
        self._change_history.clear()

        for record in data.get("applications", []):
            if self._encrypt and self._encryption_manager is not None:
                record = self._decrypt_record(record)
            app = _dict_to_app(record)
            self.add(app)

        for cr_data in data.get("change_history", []):
            self._change_history.append(ChangeRecord(
                app_id=cr_data["app_id"],
                app_name=cr_data["app_name"],
                change_type=cr_data["change_type"],
                timestamp=datetime.fromisoformat(cr_data["timestamp"]),
                previous_version=cr_data.get("previous_version"),
                new_version=cr_data.get("new_version"),
                details=cr_data.get("details", ""),
            ))

        logger.debug("Catalog loaded from %s (%d apps)", self._storage_path, self.count())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_from_indexes(self, app: Application) -> None:
        """Remove an application from the name and platform indexes."""
        name_key = app.name.lower()
        name_list = self._by_name.get(name_key, [])
        try:
            name_list.remove(app)
        except ValueError:
            pass
        if not name_list:
            self._by_name.pop(name_key, None)

        platform_list = self._by_platform.get(app.platform, [])
        try:
            platform_list.remove(app)
        except ValueError:
            pass
        if not platform_list:
            self._by_platform.pop(app.platform, None)

    def _encrypt_record(self, record: dict) -> dict:
        """Encrypt sensitive fields in a serialised application record."""
        meta = record.get("metadata", {})
        for field in _SENSITIVE_FIELDS:
            value = meta.get(field)
            if value is not None:
                meta[field] = self._encryption_manager.encrypt(value)
        return record

    def _decrypt_record(self, record: dict) -> dict:
        """Decrypt sensitive fields in a serialised application record."""
        meta = record.get("metadata", {})
        for field in _SENSITIVE_FIELDS:
            value = meta.get(field)
            if value is not None:
                try:
                    meta[field] = self._encryption_manager.decrypt(value)
                except Exception as exc:
                    logger.warning("Failed to decrypt field '%s': %s", field, exc)
                    meta[field] = None
        return record
