"""
AppxManifest analysis strategy.

Parses Windows Store AppxManifest.xml files to extract capabilities from
declared file type associations, extensions, and app metadata.

This is the most reliable source of capability data for Store apps — the
manifest is the ground truth for what file types and protocols an app handles.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType


logger = logging.getLogger(__name__)

# Namespace map used in AppxManifest.xml
_NS = {
    "default": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
    "uap":     "http://schemas.microsoft.com/appx/manifest/uap/windows10",
    "uap10":   "http://schemas.microsoft.com/appx/manifest/uap/windows10/10",
    "rescap":  "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
}

# Map file extensions to capability categories
_EXT_CATEGORY: Dict[str, CapabilityCategory] = {
    # Code / text editing
    ".py": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".py3": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".js": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".ts": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".cs": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".java": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".kt": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".cpp": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".c": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".h": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".hpp": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".rs": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".go": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".rb": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".php": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".swift": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".vb": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".asm": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".ino": CapabilityCategory.DEVELOPMENT_TOOLS,
    ".lua": CapabilityCategory.DEVELOPMENT_TOOLS,
    # Data / config
    ".json": CapabilityCategory.DATA_TRANSFORMATION,
    ".xml": CapabilityCategory.DATA_TRANSFORMATION,
    ".yaml": CapabilityCategory.DATA_TRANSFORMATION,
    ".yml": CapabilityCategory.DATA_TRANSFORMATION,
    ".csv": CapabilityCategory.DATA_TRANSFORMATION,
    ".toml": CapabilityCategory.DATA_TRANSFORMATION,
    ".ini": CapabilityCategory.DATA_TRANSFORMATION,
    ".cfg": CapabilityCategory.DATA_TRANSFORMATION,
    ".properties": CapabilityCategory.DATA_TRANSFORMATION,
    # Documents / markup
    ".md": CapabilityCategory.FILE_PROCESSING,
    ".markdown": CapabilityCategory.FILE_PROCESSING,
    ".txt": CapabilityCategory.FILE_PROCESSING,
    ".log": CapabilityCategory.FILE_PROCESSING,
    ".html": CapabilityCategory.FILE_PROCESSING,
    ".htm": CapabilityCategory.FILE_PROCESSING,
    ".css": CapabilityCategory.FILE_PROCESSING,
    ".tex": CapabilityCategory.FILE_PROCESSING,
    # Media
    ".jpg": CapabilityCategory.MULTIMEDIA,
    ".jpeg": CapabilityCategory.MULTIMEDIA,
    ".png": CapabilityCategory.MULTIMEDIA,
    ".gif": CapabilityCategory.MULTIMEDIA,
    ".mp4": CapabilityCategory.MULTIMEDIA,
    ".mp3": CapabilityCategory.MULTIMEDIA,
    ".wav": CapabilityCategory.MULTIMEDIA,
}


class AppxManifestStrategy(BaseAnalysisStrategy):
    """
    Extracts capabilities from Windows Store AppxManifest.xml.

    Covers:
    - File type associations (the most information-dense section)
    - Declared restricted capabilities (e.g. runFullTrust, broadFileSystemAccess)
    - Protocol activations
    - App display name and description from Properties
    """

    def __init__(self):
        super().__init__("appx_manifest", confidence_factor=0.95)

    def can_analyze(self, app: Application) -> bool:
        return self._find_manifest(app) is not None

    def extract_capabilities(self, app: Application) -> List[Capability]:
        manifest_path = self._find_manifest(app)
        if not manifest_path:
            return []

        try:
            tree = ET.parse(manifest_path)
        except ET.ParseError as exc:
            self.logger.debug(f"Could not parse {manifest_path}: {exc}")
            return []

        root = tree.getroot()
        capabilities: List[Capability] = []

        capabilities.extend(self._extract_file_type_associations(app, root))
        capabilities.extend(self._extract_restricted_capabilities(app, root))
        capabilities.extend(self._extract_protocol_activations(app, root))

        return capabilities

    # ------------------------------------------------------------------ #
    # Manifest location                                                    #
    # ------------------------------------------------------------------ #

    def _find_manifest(self, app: Application) -> Optional[Path]:
        """Return the AppxManifest.xml path if it exists under the install dir."""
        candidate = app.installation_path / "AppxManifest.xml"
        if candidate.exists():
            return candidate

        # Some Store apps nest the manifest one level deeper
        for child in app.installation_path.iterdir() if app.installation_path.exists() else []:
            if child.is_dir():
                nested = child / "AppxManifest.xml"
                if nested.exists():
                    return nested

        return None

    # ------------------------------------------------------------------ #
    # File type associations                                               #
    # ------------------------------------------------------------------ #

    def _extract_file_type_associations(
        self, app: Application, root: ET.Element
    ) -> List[Capability]:
        """
        Parse every <uap:FileTypeAssociation> block and produce one capability
        per association group, listing all the extensions it covers.
        """
        capabilities: List[Capability] = []

        for fta in root.iter(f"{{{_NS['uap']}}}FileTypeAssociation"):
            assoc_name = fta.get("Name", "unknown")
            display_name_el = fta.find(f"{{{_NS['uap']}}}DisplayName")
            display_name = (
                display_name_el.text.strip()
                if display_name_el is not None and display_name_el.text
                else assoc_name.replace("-", " ").title()
            )

            # Collect all extensions declared in this association
            extensions: List[str] = []
            for ft in fta.iter(f"{{{_NS['uap']}}}FileType"):
                if ft.text:
                    extensions.append(ft.text.strip().lower())
            # uap10 wildcard
            for ft in fta.iter(f"{{{_NS['uap10']}}}FileType"):
                if ft.text and ft.text.strip() != "*":
                    extensions.append(ft.text.strip().lower())

            if not extensions:
                continue

            category = self._category_for_extensions(extensions)
            description = (
                f"Opens and edits {display_name} files "
                f"({', '.join(extensions)})"
            )

            cap = self._create_capability(
                app=app,
                name=f"{display_name} File Handling",
                category=category,
                interface_type=InterfaceType.GUI,
                description=description,
                confidence=0.95,
                supported_formats=extensions,
            )
            capabilities.append(cap)

        return capabilities

    # ------------------------------------------------------------------ #
    # Restricted capabilities                                              #
    # ------------------------------------------------------------------ #

    def _extract_restricted_capabilities(
        self, app: Application, root: ET.Element
    ) -> List[Capability]:
        """
        Map <rescap:Capability> declarations to security-relevant capabilities.
        """
        _RESCAP_MAP = {
            "runFullTrust": (
                "Full Trust Execution",
                CapabilityCategory.SYSTEM_INTEGRATION,
                "Runs outside the Store sandbox with full system access",
            ),
            "broadFileSystemAccess": (
                "Broad File System Access",
                CapabilityCategory.FILE_PROCESSING,
                "Can read and write anywhere on the file system",
            ),
            "backgroundMediaPlayback": (
                "Background Media Playback",
                CapabilityCategory.MULTIMEDIA,
                "Plays media while the app is in the background",
            ),
            "userDataTasks": (
                "User Data / Tasks Access",
                CapabilityCategory.SYSTEM_INTEGRATION,
                "Can access calendar and task data",
            ),
        }

        capabilities: List[Capability] = []

        for cap_el in root.iter(f"{{{_NS['rescap']}}}Capability"):
            cap_name = cap_el.get("Name", "")
            if cap_name in _RESCAP_MAP:
                friendly, category, description = _RESCAP_MAP[cap_name]
                cap = self._create_capability(
                    app=app,
                    name=friendly,
                    category=category,
                    interface_type=InterfaceType.LIBRARY,
                    description=description,
                    confidence=1.0,
                )
                capabilities.append(cap)

        return capabilities

    # ------------------------------------------------------------------ #
    # Protocol activations                                                 #
    # ------------------------------------------------------------------ #

    def _extract_protocol_activations(
        self, app: Application, root: ET.Element
    ) -> List[Capability]:
        """Extract <uap:Protocol> declarations (custom URI scheme handlers)."""
        capabilities: List[Capability] = []

        for proto in root.iter(f"{{{_NS['uap']}}}Protocol"):
            scheme = proto.get("Name", "")
            if not scheme:
                continue

            cap = self._create_capability(
                app=app,
                name=f"URI Protocol Handler: {scheme}://",
                category=CapabilityCategory.SYSTEM_INTEGRATION,
                interface_type=InterfaceType.REST_API,
                description=f"Handles '{scheme}://' URI activations",
                confidence=0.9,
            )
            capabilities.append(cap)

        return capabilities

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _category_for_extensions(self, extensions: List[str]) -> CapabilityCategory:
        """Pick the most common category across the given extensions."""
        counts: Dict[CapabilityCategory, int] = {}
        for ext in extensions:
            cat = _EXT_CATEGORY.get(ext, CapabilityCategory.FILE_PROCESSING)
            counts[cat] = counts.get(cat, 0) + 1

        if not counts:
            return CapabilityCategory.FILE_PROCESSING

        return max(counts, key=lambda c: counts[c])
