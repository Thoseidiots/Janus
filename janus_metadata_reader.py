"""
janus_metadata_reader.py
=========================
Metadata Reader for Janus.

Parses and extracts metadata from any file Janus encounters — before reading
the content, during job processing, or when inspecting client-provided files.

What "metadata" means here:
  - Filesystem facts: size, timestamps, permissions, path info
  - Format identity: detected MIME type, magic bytes, encoding
  - Embedded metadata: EXIF (images), ID3 (audio), PDF info dict,
    Office document properties, ZIP/archive manifests
  - Content statistics: line count, word count, record count, entropy
  - Integrity: SHA-256 / MD5 checksums

Reuses helpers from janus_corrupt_file_reader so format detection
stays consistent across both modules.

Supported embedded metadata:
  - Images  : JPEG/PNG/GIF/BMP/TIFF/WEBP via Pillow (EXIF, ICC, XMP)
  - Audio   : MP3/FLAC/OGG/WAV via mutagen (ID3, Vorbis, RIFF)
  - PDF     : info dict + page count via pypdf (no heavy deps)
  - ZIP/JAR : manifest, file list, compression stats via zipfile
  - Office  : DOCX/XLSX/PPTX (ZIP-based) property XML
  - SQLite  : page size, page count, schema summary
  - Text    : line/word/char counts, detected language hint
  - Any     : filesystem stats + SHA-256 always present
"""

import os
import io
import re
import math
import json
import stat
import struct
import hashlib
import logging
import zipfile
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Reuse helpers from the corrupt file reader
from janus_corrupt_file_reader import _read_raw, _detect_encoding, _sniff_format

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL HEAVY DEPS — graceful degradation if not installed
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from PIL import Image, ExifTags
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

try:
    import mutagen
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    try:
        import PyPDF2 as pypdf          # older alias
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False


# ═══════════════════════════════════════════════════════════════════════════════
# MIME / MAGIC BYTE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

# (magic_bytes_prefix, mime_type)
_MAGIC: List[Tuple[bytes, str]] = [
    (b"\xff\xd8\xff",               "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n",          "image/png"),
    (b"GIF87a",                     "image/gif"),
    (b"GIF89a",                     "image/gif"),
    (b"BM",                         "image/bmp"),
    (b"II\x2a\x00",                 "image/tiff"),
    (b"MM\x00\x2a",                 "image/tiff"),
    (b"RIFF",                       "audio/wav"),   # also AVI — disambiguate below
    (b"ID3",                        "audio/mpeg"),
    (b"\xff\xfb",                   "audio/mpeg"),
    (b"fLaC",                       "audio/flac"),
    (b"OggS",                       "audio/ogg"),
    (b"%PDF",                       "application/pdf"),
    (b"PK\x03\x04",                 "application/zip"),
    (b"PK\x05\x06",                 "application/zip"),
    (b"SQLite format 3\x00",        "application/x-sqlite3"),
    (b"\x1f\x8b",                   "application/gzip"),
    (b"BZh",                        "application/x-bzip2"),
    (b"\xfd7zXZ\x00",               "application/x-xz"),
    (b"Rar!\x1a\x07",               "application/x-rar"),
    (b"\x7fELF",                    "application/x-elf"),
    (b"MZ",                         "application/x-dosexec"),
    (b"\xca\xfe\xba\xbe",           "application/x-mach-binary"),
    (b"\xef\xbb\xbf",               "text/plain; charset=utf-8-bom"),
]

_EXT_MIME: Dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv":  "text/csv",
    ".tsv":  "text/tab-separated-values",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".html": "text/html",
    ".xml":  "application/xml",
    ".yaml": "application/yaml",
    ".yml":  "application/yaml",
    ".toml": "application/toml",
    ".ini":  "text/plain",
    ".cfg":  "text/plain",
    ".py":   "text/x-python",
    ".js":   "application/javascript",
    ".ts":   "application/typescript",
    ".log":  "text/plain",
    ".pdf":  "application/pdf",
    ".zip":  "application/zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".mp3":  "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg":  "audio/ogg",
    ".wav":  "audio/wav",
    ".mp4":  "video/mp4",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".webp": "image/webp",
    ".db":   "application/x-sqlite3",
    ".sqlite":  "application/x-sqlite3",
    ".sqlite3": "application/x-sqlite3",
    ".pt":   "application/octet-stream",  # PyTorch weights
    ".pth":  "application/octet-stream",
    ".bak":  "application/octet-stream",
    ".gz":   "application/gzip",
}


def _detect_mime(raw: bytes, path: str) -> str:
    """Detect MIME type from magic bytes, falling back to extension."""
    for magic, mime in _MAGIC:
        if raw[:len(magic)] == magic:
            # Disambiguate RIFF: WAV vs AVI
            if magic == b"RIFF" and len(raw) >= 12:
                form = raw[8:12]
                if form == b"AVI ":
                    return "video/avi"
            return mime
    ext = Path(path).suffix.lower()
    return _EXT_MIME.get(ext, "application/octet-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilesystemMeta:
    """Standard filesystem attributes."""
    path: str
    filename: str
    extension: str
    size_bytes: int
    created_at: Optional[str]       # ISO-8601 or None if unavailable
    modified_at: Optional[str]
    accessed_at: Optional[str]
    is_readable: bool
    is_writable: bool
    is_symlink: bool
    absolute_path: str


@dataclass
class FormatMeta:
    """Detected format and encoding information."""
    mime_type: str
    detected_format: str            # matches _sniff_format output
    encoding: Optional[str]         # text encoding, None for binary
    is_text: bool
    is_binary: bool
    magic_bytes_hex: str            # first 16 bytes as hex


@dataclass
class IntegrityMeta:
    """Checksums and integrity information."""
    sha256: str
    md5: str
    size_bytes: int
    is_empty: bool


@dataclass
class ContentMeta:
    """Content-level statistics (text files)."""
    line_count: int = 0
    word_count: int = 0
    char_count: int = 0
    blank_lines: int = 0
    avg_line_length: float = 0.0
    entropy_bits: float = 0.0       # Shannon entropy of raw bytes


@dataclass
class EmbeddedMeta:
    """
    Format-specific embedded metadata.

    The `data` dict holds whatever the format provides:
      - Images  : {"width", "height", "mode", "exif": {...}}
      - Audio   : {"duration_seconds", "bitrate", "tags": {...}}
      - PDF     : {"page_count", "info": {...}}
      - ZIP     : {"file_count", "total_uncompressed", "files": [...]}
      - Office  : {"title", "author", "created", "modified", ...}
      - SQLite  : {"page_size", "page_count", "tables": [...]}
    """
    format_name: str = "none"
    data: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class FileMetadata:
    """Complete metadata profile for a single file."""
    filesystem: FilesystemMeta
    format: FormatMeta
    integrity: IntegrityMeta
    content: Optional[ContentMeta]      # None for binary files
    embedded: EmbeddedMeta
    read_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        def _asdict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _asdict(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [_asdict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _asdict(v) for k, v in obj.items()}
            return obj
        return _asdict(self)

    def summary(self) -> str:
        lines = [
            f"File     : {self.filesystem.filename}",
            f"Path     : {self.filesystem.absolute_path}",
            f"Size     : {self.filesystem.size_bytes:,} bytes",
            f"MIME     : {self.format.mime_type}",
            f"Format   : {self.format.detected_format}",
            f"Encoding : {self.format.encoding or 'binary'}",
            f"Modified : {self.filesystem.modified_at}",
            f"SHA-256  : {self.integrity.sha256[:16]}...",
        ]
        if self.content:
            lines += [
                f"Lines    : {self.content.line_count:,}",
                f"Words    : {self.content.word_count:,}",
                f"Entropy  : {self.content.entropy_bits:.2f} bits/byte",
            ]
        if self.embedded.format_name != "none":
            lines.append(f"Embedded : {self.embedded.format_name}")
            for k, v in self.embedded.data.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED METADATA PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

class _ImageMetaParser:
    @staticmethod
    def parse(path: str, raw: bytes) -> EmbeddedMeta:
        meta = EmbeddedMeta(format_name="image")
        if not HAS_PILLOW:
            meta.parse_errors.append("Pillow not installed — EXIF unavailable")
            # Still get basic dimensions from PNG/JPEG headers without Pillow
            meta.data.update(_ImageMetaParser._header_only(raw))
            return meta
        try:
            img = Image.open(io.BytesIO(raw))
            meta.data["width"] = img.width
            meta.data["height"] = img.height
            meta.data["mode"] = img.mode
            meta.data["format"] = img.format

            # EXIF
            exif_raw = img._getexif() if hasattr(img, "_getexif") else None
            if exif_raw:
                exif = {
                    ExifTags.TAGS.get(k, str(k)): str(v)
                    for k, v in exif_raw.items()
                    if k in ExifTags.TAGS
                }
                meta.data["exif"] = exif

            # ICC profile presence
            if "icc_profile" in (img.info or {}):
                meta.data["icc_profile"] = True

        except Exception as e:
            meta.parse_errors.append(f"Pillow parse error: {e}")
            meta.data.update(_ImageMetaParser._header_only(raw))
        return meta

    @staticmethod
    def _header_only(raw: bytes) -> Dict[str, Any]:
        """Extract dimensions from PNG/JPEG headers without Pillow."""
        result: Dict[str, Any] = {}
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
            w = struct.unpack(">I", raw[16:20])[0]
            h = struct.unpack(">I", raw[20:24])[0]
            result["width"] = w
            result["height"] = h
        elif raw[:2] == b"\xff\xd8":
            # Scan JPEG SOF markers
            i = 2
            while i < len(raw) - 8:
                if raw[i] != 0xff:
                    break
                marker = raw[i + 1]
                length = struct.unpack(">H", raw[i + 2 : i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2):
                    result["height"] = struct.unpack(">H", raw[i + 5 : i + 7])[0]
                    result["width"]  = struct.unpack(">H", raw[i + 7 : i + 9])[0]
                    break
                i += 2 + length
        return result


class _AudioMetaParser:
    @staticmethod
    def parse(path: str) -> EmbeddedMeta:
        meta = EmbeddedMeta(format_name="audio")
        if not HAS_MUTAGEN:
            meta.parse_errors.append("mutagen not installed — audio tags unavailable")
            return meta
        try:
            audio = mutagen.File(path, easy=True)
            if audio is None:
                meta.parse_errors.append("mutagen could not identify audio format")
                return meta
            if hasattr(audio, "info"):
                info = audio.info
                meta.data["duration_seconds"] = round(
                    getattr(info, "length", 0), 2
                )
                meta.data["bitrate"] = getattr(info, "bitrate", None)
                meta.data["sample_rate"] = getattr(info, "sample_rate", None)
                meta.data["channels"] = getattr(info, "channels", None)
            # Tags (EasyID3 / Vorbis / etc.)
            tags = {k: list(v) for k, v in audio.items()} if audio else {}
            if tags:
                meta.data["tags"] = tags
        except Exception as e:
            meta.parse_errors.append(f"mutagen error: {e}")
        return meta


class _PdfMetaParser:
    @staticmethod
    def parse(path: str, raw: bytes) -> EmbeddedMeta:
        meta = EmbeddedMeta(format_name="pdf")
        if not HAS_PYPDF:
            meta.parse_errors.append("pypdf not installed — PDF metadata unavailable")
            # Fallback: count pages via regex on raw bytes
            pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
            if pages:
                meta.data["page_count_estimate"] = pages
            return meta
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            meta.data["page_count"] = len(reader.pages)
            info = reader.metadata
            if info:
                meta.data["info"] = {
                    k.lstrip("/"): str(v)
                    for k, v in info.items()
                    if v is not None
                }
        except Exception as e:
            meta.parse_errors.append(f"pypdf error: {e}")
            pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
            if pages:
                meta.data["page_count_estimate"] = pages
        return meta


class _ZipMetaParser:
    @staticmethod
    def parse(path: str, raw: bytes) -> EmbeddedMeta:
        """Handles ZIP, DOCX, XLSX, PPTX (all ZIP-based)."""
        ext = Path(path).suffix.lower()
        is_office = ext in (".docx", ".xlsx", ".pptx")
        meta = EmbeddedMeta(format_name="office" if is_office else "zip")

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                infos = zf.infolist()
                meta.data["file_count"] = len(infos)
                meta.data["total_uncompressed"] = sum(
                    i.file_size for i in infos
                )
                meta.data["total_compressed"] = sum(
                    i.compress_size for i in infos
                )
                if meta.data["total_uncompressed"] > 0:
                    ratio = (
                        1 - meta.data["total_compressed"]
                        / meta.data["total_uncompressed"]
                    )
                    meta.data["compression_ratio"] = round(ratio, 3)

                meta.data["files"] = [i.filename for i in infos[:50]]
                if len(infos) > 50:
                    meta.data["files_truncated"] = True

                # Office document properties
                if is_office:
                    _ZipMetaParser._parse_office_props(zf, meta)

        except zipfile.BadZipFile as e:
            meta.parse_errors.append(f"Bad ZIP: {e}")
        except Exception as e:
            meta.parse_errors.append(f"ZIP parse error: {e}")
        return meta

    @staticmethod
    def _parse_office_props(zf: zipfile.ZipFile, meta: EmbeddedMeta) -> None:
        """Extract core properties from Office Open XML."""
        prop_files = [
            "docProps/core.xml",
            "docProps/app.xml",
        ]
        for pf in prop_files:
            try:
                xml = zf.read(pf).decode("utf-8", errors="replace")
                # Pull out simple tag values
                for tag in (
                    "dc:title", "dc:creator", "dc:description",
                    "cp:lastModifiedBy", "dcterms:created", "dcterms:modified",
                    "Pages", "Words", "Characters", "Application",
                ):
                    m = re.search(
                        rf"<{re.escape(tag)}[^>]*>(.*?)</{re.escape(tag)}>",
                        xml,
                        re.DOTALL,
                    )
                    if m:
                        key = tag.split(":")[-1].lower()
                        meta.data[key] = m.group(1).strip()
            except KeyError:
                pass
            except Exception as e:
                meta.parse_errors.append(f"Office props error ({pf}): {e}")


class _SqliteMetaParser:
    @staticmethod
    def parse(path: str) -> EmbeddedMeta:
        meta = EmbeddedMeta(format_name="sqlite")
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            cursor = conn.cursor()

            # Page size and count from header
            page_size = cursor.execute("PRAGMA page_size").fetchone()[0]
            page_count = cursor.execute("PRAGMA page_count").fetchone()[0]
            meta.data["page_size"] = page_size
            meta.data["page_count"] = page_count
            meta.data["size_bytes"] = page_size * page_count

            # Schema summary
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_info = {}
            for (tname,) in tables:
                try:
                    count = cursor.execute(
                        f"SELECT COUNT(*) FROM [{tname}]"
                    ).fetchone()[0]
                    cols = cursor.execute(
                        f"PRAGMA table_info([{tname}])"
                    ).fetchall()
                    table_info[tname] = {
                        "row_count": count,
                        "columns": [c[1] for c in cols],
                    }
                except sqlite3.DatabaseError:
                    table_info[tname] = {"error": "unreadable"}

            meta.data["tables"] = table_info
            meta.data["table_count"] = len(tables)

            # SQLite version
            version = cursor.execute("SELECT sqlite_version()").fetchone()[0]
            meta.data["sqlite_version"] = version

            conn.close()
        except sqlite3.DatabaseError as e:
            meta.parse_errors.append(f"SQLite meta error: {e}")
        return meta


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence (bits per byte)."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    n = len(data)
    return -sum(
        (f / n) * math.log2(f / n) for f in freq if f > 0
    )


def _filesystem_meta(path: str) -> FilesystemMeta:
    p = Path(path)
    try:
        st = p.stat()
        size = st.st_size
        modified = datetime.fromtimestamp(st.st_mtime).isoformat()
        accessed = datetime.fromtimestamp(st.st_atime).isoformat()
        # ctime is creation time on Windows, metadata-change on Unix
        created = datetime.fromtimestamp(st.st_ctime).isoformat()
        readable = os.access(path, os.R_OK)
        writable = os.access(path, os.W_OK)
        is_symlink = p.is_symlink()
    except OSError:
        size = 0
        modified = accessed = created = None
        readable = writable = is_symlink = False

    return FilesystemMeta(
        path=str(path),
        filename=p.name,
        extension=p.suffix.lower(),
        size_bytes=size,
        created_at=created,
        modified_at=modified,
        accessed_at=accessed,
        is_readable=readable,
        is_writable=writable,
        is_symlink=is_symlink,
        absolute_path=str(p.resolve()),
    )


def _content_meta(raw: bytes, encoding: str) -> ContentMeta:
    """Compute text-level statistics."""
    try:
        text = raw.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")

    lines = text.splitlines()
    words = re.findall(r"\S+", text)
    blank = sum(1 for l in lines if not l.strip())
    avg_len = sum(len(l) for l in lines) / max(len(lines), 1)

    return ContentMeta(
        line_count=len(lines),
        word_count=len(words),
        char_count=len(text),
        blank_lines=blank,
        avg_line_length=round(avg_len, 1),
        entropy_bits=round(_entropy(raw), 3),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN READER
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataReader:
    """
    Reads and parses metadata from any file without modifying it.

    Usage
    -----
    reader = MetadataReader()
    meta = reader.read("some_file.pdf")
    print(meta.summary())
    data = meta.to_dict()   # JSON-serialisable

    # Batch
    results = reader.read_many(["a.jpg", "b.csv", "c.db"])
    """

    def read(self, path: str) -> FileMetadata:
        """
        Extract full metadata from a file.

        Always returns a FileMetadata object — never raises.
        """
        import time
        t0 = time.monotonic()
        path = str(path)

        # ── Filesystem ─────────────────────────────────────────────────────
        fs = _filesystem_meta(path)

        # ── Raw bytes ──────────────────────────────────────────────────────
        raw, _ = _read_raw(path)

        # ── Format detection ───────────────────────────────────────────────
        mime = _detect_mime(raw, path)
        fmt_key = _sniff_format(raw, path)
        is_text = mime.startswith("text/") or fmt_key in (
            "json", "jsonl", "csv", "tsv", "text", "python"
        )

        encoding: Optional[str] = None
        if is_text and raw:
            encoding = _detect_encoding(raw)

        fmt = FormatMeta(
            mime_type=mime,
            detected_format=fmt_key,
            encoding=encoding,
            is_text=is_text,
            is_binary=not is_text,
            magic_bytes_hex=raw[:16].hex() if raw else "",
        )

        # ── Integrity ──────────────────────────────────────────────────────
        integrity = IntegrityMeta(
            sha256=hashlib.sha256(raw).hexdigest() if raw else "",
            md5=hashlib.md5(raw).hexdigest() if raw else "",
            size_bytes=len(raw),
            is_empty=len(raw) == 0,
        )

        # ── Content stats (text only) ──────────────────────────────────────
        content: Optional[ContentMeta] = None
        if is_text and raw and encoding:
            try:
                content = _content_meta(raw, encoding)
                content.entropy_bits = round(_entropy(raw), 3)
            except Exception:
                pass
        elif raw and not is_text:
            # Still compute entropy for binary files
            pass

        # ── Embedded metadata ──────────────────────────────────────────────
        embedded = self._parse_embedded(path, raw, mime, fmt_key)

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        result = FileMetadata(
            filesystem=fs,
            format=fmt,
            integrity=integrity,
            content=content,
            embedded=embedded,
            read_duration_ms=elapsed_ms,
        )

        logger.info(
            f"[MetadataReader] {path}: {mime} | "
            f"{fs.size_bytes:,} bytes | {elapsed_ms}ms"
        )
        return result

    def read_many(self, paths: List[str]) -> Dict[str, FileMetadata]:
        """Read metadata for multiple files. Returns path -> FileMetadata."""
        return {p: self.read(p) for p in paths}

    def to_json(self, path: str, indent: int = 2) -> str:
        """Read metadata and return as a JSON string."""
        return json.dumps(self.read(path).to_dict(), indent=indent, default=str)

    # ── Internal ────────────────────────────────────────────────────────────

    def _parse_embedded(
        self, path: str, raw: bytes, mime: str, fmt_key: str
    ) -> EmbeddedMeta:
        """Dispatch to the right embedded parser based on MIME / format."""
        try:
            if mime.startswith("image/"):
                return _ImageMetaParser.parse(path, raw)

            if mime.startswith("audio/"):
                return _AudioMetaParser.parse(path)

            if mime == "application/pdf":
                return _PdfMetaParser.parse(path, raw)

            if mime == "application/zip" or mime.startswith(
                "application/vnd.openxmlformats"
            ):
                return _ZipMetaParser.parse(path, raw)

            if fmt_key == "sqlite" or mime == "application/x-sqlite3":
                return _SqliteMetaParser.parse(path)

        except Exception as e:
            logger.warning(f"[MetadataReader] Embedded parse failed for {path}: {e}")
            return EmbeddedMeta(
                format_name="error",
                parse_errors=[str(e)],
            )

        return EmbeddedMeta(format_name="none")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def read_metadata(path: str) -> FileMetadata:
    """
    Module-level shortcut.

    Example
    -------
    meta = read_metadata("client_data.xlsx")
    print(meta.summary())
    print(meta.embedded.data.get("title"))
    """
    return MetadataReader().read(path)
