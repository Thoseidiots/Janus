"""
File Association analysis strategy.

Reads the Windows registry (HKEY_CLASSES_ROOT) to discover which file types
an application is registered to handle.  This covers all traditional Win32
and .NET apps — anything that registered itself via the standard shell
association mechanism (installers, Python, Office, 7-Zip, VLC, etc.).

Registry walk:
  HKCR\.<ext>
      (default)   -> ProgID  e.g. "Python.File"
      Content Type -> MIME   e.g. "text/x-python"
      PerceivedType -> category hint  e.g. "text", "audio", "video"

  HKCR\<ProgID>
      (default)   -> friendly name  e.g. "Python File"
      shell\<verb>\command
          (default) -> command string  e.g. '"py.exe" "%L" %*'

The strategy resolves the command string back to an executable path and
compares it against the Application's executable_path to decide ownership.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import threading

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType

# winreg is Windows-only; guard so the module can be imported on other platforms
try:
    import winreg
    _WINREG_AVAILABLE = True
except ImportError:
    _WINREG_AVAILABLE = False

logger = logging.getLogger(__name__)

# PerceivedType -> CapabilityCategory
_PERCEIVED_CATEGORY: Dict[str, CapabilityCategory] = {
    "text":        CapabilityCategory.FILE_PROCESSING,
    "document":    CapabilityCategory.FILE_PROCESSING,
    "image":       CapabilityCategory.MULTIMEDIA,
    "video":       CapabilityCategory.MULTIMEDIA,
    "audio":       CapabilityCategory.MULTIMEDIA,
    "compressed":  CapabilityCategory.FILE_PROCESSING,
    "system":      CapabilityCategory.SYSTEM_INTEGRATION,
    "application": CapabilityCategory.SYSTEM_INTEGRATION,
}

# MIME prefix -> CapabilityCategory (fallback when PerceivedType is absent)
_MIME_CATEGORY: Dict[str, CapabilityCategory] = {
    "text/":        CapabilityCategory.FILE_PROCESSING,
    "image/":       CapabilityCategory.MULTIMEDIA,
    "video/":       CapabilityCategory.MULTIMEDIA,
    "audio/":       CapabilityCategory.MULTIMEDIA,
    "application/": CapabilityCategory.FILE_PROCESSING,
}

# Extension sets for development-tool categorisation override
_CODE_EXTENSIONS: Set[str] = {
    ".py", ".py3", ".pyw", ".pyt", ".rpy",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".kts",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".cs", ".vb",
    ".rs", ".go", ".rb", ".php", ".lua", ".swift", ".zig",
    ".asm", ".ino", ".sql", ".sh", ".bat", ".ps1",
    ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".tex",
}

# Maximum number of HKCR extension keys to walk (keeps startup fast)
_MAX_EXTENSIONS = 1500
# Verbs to check for the handler command, in priority order
_HANDLER_VERBS = ("open", "edit", "play", "print", "view")


class FileAssociationStrategy(BaseAnalysisStrategy):
    """
    Discovers application capabilities via Windows shell file associations.

    For a given Application the strategy:
    1. Builds (or reuses) a cached index: exe_stem -> [(ext, friendly_name,
       perceived_type, mime_type)] by walking HKCR once.
    2. Looks up the application's executable stem in the index.
    3. Groups matched extensions by category and produces one Capability
       per group, listing all the file types the app handles.

    The index is class-level so it is built only once per process even when
    multiple applications are analysed in the same run.
    """

    # Class-level cache shared across all instances
    _index: Optional[Dict[str, List[Tuple[str, str, str, str]]]] = None
    _index_lock: threading.Lock = threading.Lock()

    def __init__(self):
        super().__init__("file_association", confidence_factor=0.9)

    # ------------------------------------------------------------------ #
    # Strategy interface                                                   #
    # ------------------------------------------------------------------ #

    def can_analyze(self, app: Application) -> bool:
        if not _WINREG_AVAILABLE:
            return False
        if not app.is_accessible:
            return False
        index = self._get_index()
        return bool(self._get_matching_associations(app, index))

    def extract_capabilities(self, app: Application) -> List[Capability]:
        if not _WINREG_AVAILABLE:
            return []

        index = self._get_index()
        associations = self._get_matching_associations(app, index)

        if not associations:
            return []

        # Group by category
        groups: Dict[CapabilityCategory, List[Tuple[str, str]]] = {}
        for ext, friendly_name, perceived_type, mime_type in associations:
            category = self._resolve_category(ext, perceived_type, mime_type)
            if category not in groups:
                groups[category] = []
            groups[category].append((ext, friendly_name))

        capabilities: List[Capability] = []
        for category, items in groups.items():
            extensions = sorted({ext for ext, _ in items})
            names = list(dict.fromkeys(name for _, name in items if name))
            description = self._build_description(app.name, extensions, names)

            cap = self._create_capability(
                app=app,
                name=f"{category.value.replace('_', ' ').title()} File Handler",
                category=category,
                interface_type=InterfaceType.GUI,
                description=description,
                confidence=0.9,
                supported_formats=extensions,
            )
            capabilities.append(cap)

        return capabilities

    def _get_matching_associations(
        self,
        app: Application,
        index: Dict[str, List[Tuple[str, str, str, str]]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Return all associations that belong to this application.

        Matches on:
        1. The primary executable stem (e.g. "notepad++")
        2. Any other .exe in the same install directory (e.g. "7z", "7zG"
           alongside "7zFM" for 7-Zip)
        3. A sanitised stem with non-alphanumeric chars stripped
           (handles "notepad++" -> "notepad")
        4. Launcher executables in the same directory tree
           (e.g. "py" / "pyw" launchers in Python\Launcher\ resolve back
           to the Python install)
        """
        matched: List[Tuple[str, str, str, str]] = []
        seen_exts: Set[str] = set()

        # Collect candidate stems
        stems: Set[str] = set()

        primary_stem = app.executable_path.stem.lower()
        stems.add(primary_stem)

        # Sanitised version (strip non-alphanumeric)
        sanitised = re.sub(r"[^a-z0-9]", "", primary_stem)
        if sanitised and sanitised != primary_stem:
            stems.add(sanitised)

        # Sibling executables in the same install directory
        try:
            for sibling in app.installation_path.glob("*.exe"):
                stems.add(sibling.stem.lower())
        except (OSError, PermissionError):
            pass

        # Also check subdirectories one level deep (catches launchers like
        # Python\Launcher\py.exe that live under the same install root)
        try:
            for subdir in app.installation_path.iterdir():
                if subdir.is_dir():
                    for sibling in subdir.glob("*.exe"):
                        stems.add(sibling.stem.lower())
        except (OSError, PermissionError):
            pass

        # Check sibling directories of the install path (catches Python\Launcher\
        # which lives next to Python\Python311\ rather than inside it)
        try:
            parent = app.installation_path.parent
            for sibling_dir in parent.iterdir():
                if sibling_dir.is_dir() and sibling_dir != app.installation_path:
                    for exe in sibling_dir.glob("*.exe"):
                        stems.add(exe.stem.lower())
        except (OSError, PermissionError):
            pass

        for stem in stems:
            for assoc in index.get(stem, []):
                ext = assoc[0]
                if ext not in seen_exts:
                    seen_exts.add(ext)
                    matched.append(assoc)

        return matched

    # ------------------------------------------------------------------ #
    # Index construction                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _get_index(cls) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Return the cached exe-stem -> associations index, building it if needed.
        Thread-safe: only one thread builds the index; others wait.
        """
        if cls._index is not None:
            return cls._index
        
        with cls._index_lock:
            # Double-checked locking — another thread may have built it while we waited
            if cls._index is None:
                cls._index = cls._build_index()
        
        return cls._index

    @classmethod
    def _build_index(cls) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Walk HKCR once and build:
            { exe_stem_lower: [(ext, friendly_name, perceived_type, mime)] }
        """
        index: Dict[str, List[Tuple[str, str, str, str]]] = {}

        if not _WINREG_AVAILABLE:
            return index

        HKCR = winreg.HKEY_CLASSES_ROOT

        try:
            with winreg.OpenKey(HKCR, "") as root:
                i = 0
                while i < _MAX_EXTENSIONS:
                    try:
                        key_name = winreg.EnumKey(root, i)
                        i += 1
                    except OSError:
                        break

                    if not key_name.startswith("."):
                        continue

                    ext = key_name.lower()

                    try:
                        prog_id, perceived_type, mime_type = cls._read_extension_key(
                            HKCR, key_name
                        )
                    except Exception:
                        continue

                    if not prog_id:
                        continue

                    try:
                        friendly_name, exe_stem = cls._read_prog_id(HKCR, prog_id)
                    except Exception:
                        continue

                    if not exe_stem:
                        continue

                    entry = (ext, friendly_name, perceived_type, mime_type)
                    if exe_stem not in index:
                        index[exe_stem] = []
                    index[exe_stem].append(entry)

        except OSError as exc:
            logger.debug(f"Could not walk HKCR: {exc}")

        logger.debug(
            f"FileAssociation index built: {len(index)} executables, "
            f"{sum(len(v) for v in index.values())} associations"
        )
        return index

    @staticmethod
    def _read_extension_key(
        hkcr, ext_name: str
    ) -> Tuple[str, str, str]:
        """Read ProgID, PerceivedType, and Content Type from an extension key."""
        prog_id = ""
        perceived_type = ""
        mime_type = ""

        with winreg.OpenKey(hkcr, ext_name) as k:
            i = 0
            while True:
                try:
                    name, data, _ = winreg.EnumValue(k, i)
                    i += 1
                    name_lower = (name or "").lower()
                    if name_lower == "" and isinstance(data, str):
                        prog_id = data.strip()
                    elif name_lower == "perceivedtype" and isinstance(data, str):
                        perceived_type = data.strip().lower()
                    elif name_lower == "content type" and isinstance(data, str):
                        mime_type = data.strip().lower()
                except OSError:
                    break

        return prog_id, perceived_type, mime_type

    @staticmethod
    def _read_prog_id(hkcr, prog_id: str) -> Tuple[str, str]:
        """
        Read the friendly name and resolve the handler executable stem.
        Returns (friendly_name, exe_stem_lower).
        """
        friendly_name = ""
        exe_stem = ""

        try:
            with winreg.OpenKey(hkcr, prog_id) as k:
                try:
                    val, _ = winreg.QueryValueEx(k, "")
                    if isinstance(val, str):
                        friendly_name = val.strip()
                except FileNotFoundError:
                    pass
        except OSError:
            return friendly_name, exe_stem

        # Try each verb in priority order
        for verb in _HANDLER_VERBS:
            cmd_path = f"{prog_id}\\shell\\{verb}\\command"
            try:
                with winreg.OpenKey(hkcr, cmd_path) as cmd_key:
                    val, _ = winreg.QueryValueEx(cmd_key, "")
                    if isinstance(val, str) and val.strip():
                        exe_stem = FileAssociationStrategy._extract_exe_stem(val)
                        if exe_stem:
                            break
            except OSError:
                continue

        return friendly_name, exe_stem

    @staticmethod
    def _extract_exe_stem(command: str) -> str:
        """
        Pull the executable stem out of a shell command string.

        Handles forms like:
          "C:\\Python311\\python.exe" "%1"
          C:\\Program Files\\7-Zip\\7zFM.exe "%1"
          %SystemRoot%\\system32\\notepad.exe %1
          rundll32.exe zipfldr.dll,RouteTheCall %1   <- skip rundll32
        """
        command = command.strip()

        # Skip system launchers that don't represent a real app
        _SKIP_STEMS = {"rundll32", "dllhost", "msiexec", "regsvr32", "wscript", "cscript"}

        # Extract the first token (quoted or unquoted)
        if command.startswith('"'):
            match = re.match(r'"([^"]+)"', command)
            exe_path = match.group(1) if match else ""
        else:
            exe_path = command.split()[0] if command.split() else ""

        if not exe_path:
            return ""

        # Expand environment variables
        exe_path = re.sub(
            r"%(\w+)%",
            lambda m: __import__("os").environ.get(m.group(1), m.group(0)),
            exe_path,
        )

        stem = Path(exe_path).stem.lower()

        if stem in _SKIP_STEMS:
            return ""

        return stem

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_category(
        ext: str, perceived_type: str, mime_type: str
    ) -> CapabilityCategory:
        """Determine the best CapabilityCategory for an extension."""
        # Code extensions always win
        if ext in _CODE_EXTENSIONS:
            return CapabilityCategory.DEVELOPMENT_TOOLS

        # PerceivedType is the most reliable hint
        if perceived_type in _PERCEIVED_CATEGORY:
            return _PERCEIVED_CATEGORY[perceived_type]

        # Fall back to MIME prefix
        for prefix, cat in _MIME_CATEGORY.items():
            if mime_type.startswith(prefix):
                return cat

        return CapabilityCategory.FILE_PROCESSING

    @staticmethod
    def _build_description(
        app_name: str, extensions: List[str], friendly_names: List[str]
    ) -> str:
        ext_str = ", ".join(extensions[:10])
        if len(extensions) > 10:
            ext_str += f" (+{len(extensions) - 10} more)"

        if friendly_names:
            # Show up to 3 friendly names
            name_str = ", ".join(friendly_names[:3])
            if len(friendly_names) > 3:
                name_str += f" (+{len(friendly_names) - 3} more)"
            return (
                f"{app_name} is the registered shell handler for "
                f"{name_str} files ({ext_str})"
            )

        return f"{app_name} handles files: {ext_str}"
