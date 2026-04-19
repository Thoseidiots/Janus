"""
janus_corrupt_file_reader.py
=============================
Corrupt File Reader for Janus.

Attempts to salvage usable data from damaged, truncated, or malformed files.
Works across all contexts Janus operates in:
  - Recovering Janus's own state files (SQLite, JSON configs)
  - Parsing corrupt job deliverables or client-provided files
  - Handling malformed downloads during computer-use tasks

Supported formats:
  - JSON / JSONL (truncated, unbalanced brackets, bad escapes)
  - CSV / TSV (missing quotes, inconsistent columns, encoding issues)
  - SQLite (page corruption, incomplete transactions, WAL issues)
  - Plain text / logs (null bytes, mixed encodings, binary noise)
  - Python source files (syntax errors, truncation)
  - Binary / unknown (hex dump + entropy analysis)

Recovery philosophy:
  - Always return SOMETHING rather than raising
  - Report exactly what was salvaged and what was lost
  - Never modify the original file
"""

import os
import io
import re
import csv
import json
import sqlite3
import struct
import logging
import hashlib
import chardet
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class CorruptionType(Enum):
    """Categories of corruption detected"""
    TRUNCATED        = "truncated"        # File ends unexpectedly
    ENCODING_ERROR   = "encoding_error"   # Bad byte sequences
    SYNTAX_ERROR     = "syntax_error"     # Structural/parse errors
    MISSING_DATA     = "missing_data"     # Gaps or null regions
    CHECKSUM_FAIL    = "checksum_fail"    # Hash/checksum mismatch
    PARTIAL_WRITE    = "partial_write"    # Incomplete last record
    BINARY_NOISE     = "binary_noise"     # Non-text bytes in text file
    UNKNOWN          = "unknown"          # Could not classify


class RecoveryStatus(Enum):
    """Overall recovery outcome"""
    FULL        = "full"        # All data recovered
    PARTIAL     = "partial"     # Some data recovered, some lost
    MINIMAL     = "minimal"     # Very little recovered
    FAILED      = "failed"      # Nothing usable found


@dataclass
class RecoveryReport:
    """Result of a recovery attempt"""
    file_path: str
    file_size_bytes: int
    detected_format: str
    corruption_types: List[CorruptionType] = field(default_factory=list)
    status: RecoveryStatus = RecoveryStatus.FAILED

    # Recovered data
    data: Any = None                        # Parsed/recovered content
    raw_text: Optional[str] = None          # Best-effort decoded text
    recovered_records: int = 0              # For tabular/JSONL data
    total_records_estimated: int = 0

    # Diagnostics
    bytes_readable: int = 0
    bytes_lost: int = 0
    encoding_detected: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovery_notes: List[str] = field(default_factory=list)

    @property
    def recovery_rate(self) -> float:
        """Fraction of estimated records recovered (0.0 – 1.0)"""
        if self.total_records_estimated == 0:
            return 1.0 if self.status == RecoveryStatus.FULL else 0.0
        return self.recovered_records / self.total_records_estimated

    def summary(self) -> str:
        lines = [
            f"File      : {self.file_path}",
            f"Format    : {self.detected_format}",
            f"Status    : {self.status.value}",
            f"Corruption: {[c.value for c in self.corruption_types]}",
            f"Recovered : {self.recovered_records}/{self.total_records_estimated} records "
            f"({self.recovery_rate:.0%})",
            f"Bytes OK  : {self.bytes_readable} / {self.file_size_bytes}",
        ]
        if self.errors:
            lines.append(f"Errors    : {self.errors}")
        if self.warnings:
            lines.append(f"Warnings  : {self.warnings}")
        if self.recovery_notes:
            lines.append(f"Notes     : {self.recovery_notes}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_encoding(raw: bytes) -> str:
    """Detect the most likely text encoding for raw bytes."""
    result = chardet.detect(raw[:65536])  # sample first 64 KB
    encoding = result.get("encoding") or "utf-8"
    # Normalise common aliases
    encoding = encoding.lower().replace("-", "_")
    if encoding in ("ascii", "utf_8_sig"):
        encoding = "utf-8"
    return encoding


def _safe_decode(raw: bytes, encoding: str) -> Tuple[str, List[str]]:
    """Decode bytes, replacing undecodable sequences and reporting them."""
    warnings: List[str] = []
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        text = raw.decode(encoding, errors="replace")
        warnings.append(
            f"Encoding errors replaced with \ufffd using '{encoding}'"
        )
    # Strip null bytes that sneak into some corrupt files
    if "\x00" in text:
        text = text.replace("\x00", "")
        warnings.append("Null bytes stripped from decoded text")
    return text, warnings


def _read_raw(path: str) -> Tuple[bytes, int]:
    """Read as many bytes as possible from a file, even if partially readable."""
    file_size = 0
    try:
        file_size = os.path.getsize(path)
    except OSError:
        pass
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return raw, file_size
    except OSError as e:
        # Try reading in chunks to get whatever is accessible
        chunks: List[bytes] = []
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
        except OSError:
            pass
        return b"".join(chunks), file_size


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT-SPECIFIC RECOVERY STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class _JsonRecovery:
    """Recover data from corrupt JSON / JSONL files."""

    @staticmethod
    def recover(text: str, report: RecoveryReport) -> None:
        report.detected_format = "json"

        # ── Try 1: standard parse ──────────────────────────────────────────
        try:
            report.data = json.loads(text)
            report.status = RecoveryStatus.FULL
            report.recovered_records = 1
            report.total_records_estimated = 1
            report.recovery_notes.append("JSON parsed successfully without repair")
            return
        except json.JSONDecodeError:
            pass

        # ── Try 2: JSONL (one JSON object per line) ────────────────────────
        records: List[Any] = []
        bad_lines = 0
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
        if records:
            report.detected_format = "jsonl"
            report.data = records
            report.recovered_records = len(records)
            report.total_records_estimated = len(records) + bad_lines
            report.corruption_types.append(CorruptionType.SYNTAX_ERROR)
            report.warnings.append(
                f"{bad_lines} JSONL lines could not be parsed"
            )
            report.status = (
                RecoveryStatus.FULL if bad_lines == 0 else RecoveryStatus.PARTIAL
            )
            report.recovery_notes.append(
                f"Recovered {len(records)} JSONL records"
            )
            return

        # ── Try 3: bracket-balance repair ─────────────────────────────────
        repaired = _JsonRecovery._balance_brackets(text)
        try:
            report.data = json.loads(repaired)
            report.status = RecoveryStatus.PARTIAL
            report.recovered_records = 1
            report.total_records_estimated = 1
            report.corruption_types.append(CorruptionType.TRUNCATED)
            report.recovery_notes.append(
                "JSON recovered by closing unbalanced brackets/braces"
            )
            return
        except json.JSONDecodeError:
            pass

        # ── Try 4: extract any valid JSON fragments ────────────────────────
        fragments = _JsonRecovery._extract_fragments(text)
        if fragments:
            report.data = fragments
            report.recovered_records = len(fragments)
            report.total_records_estimated = len(fragments)
            report.status = RecoveryStatus.MINIMAL
            report.corruption_types.append(CorruptionType.SYNTAX_ERROR)
            report.recovery_notes.append(
                f"Extracted {len(fragments)} JSON fragments from corrupt file"
            )
            return

        report.errors.append("Could not recover any valid JSON from file")
        report.status = RecoveryStatus.FAILED

    @staticmethod
    def _balance_brackets(text: str) -> str:
        """Close any unclosed brackets/braces/strings at end of text."""
        stack: List[str] = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()

        closing = ""
        if in_string:
            closing += '"'
        closing += "".join(reversed(stack))
        return text.rstrip() + closing

    @staticmethod
    def _extract_fragments(text: str) -> List[Any]:
        """Pull out any valid JSON objects or arrays from arbitrary text."""
        fragments: List[Any] = []
        # Match top-level {...} or [...] blocks
        for match in re.finditer(r'(\{[^{}]*\}|\[[^\[\]]*\])', text):
            try:
                fragments.append(json.loads(match.group()))
            except json.JSONDecodeError:
                pass
        return fragments


class _CsvRecovery:
    """Recover data from corrupt CSV / TSV files."""

    @staticmethod
    def recover(text: str, path: str, report: RecoveryReport) -> None:
        ext = Path(path).suffix.lower()
        delimiter = "\t" if ext in (".tsv", ".tab") else ","
        report.detected_format = "tsv" if delimiter == "\t" else "csv"

        rows: List[List[str]] = []
        bad_rows = 0
        expected_cols: Optional[int] = None

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for i, row in enumerate(reader):
            try:
                if expected_cols is None and row:
                    expected_cols = len(row)
                if row:
                    rows.append(row)
            except csv.Error:
                bad_rows += 1

        if not rows:
            report.errors.append("No CSV rows could be parsed")
            report.status = RecoveryStatus.FAILED
            return

        # Detect and flag rows with wrong column count
        inconsistent = sum(
            1 for r in rows if expected_cols and len(r) != expected_cols
        )
        if inconsistent:
            report.warnings.append(
                f"{inconsistent} rows have inconsistent column counts "
                f"(expected {expected_cols})"
            )
            report.corruption_types.append(CorruptionType.MISSING_DATA)

        report.data = rows
        report.recovered_records = len(rows)
        report.total_records_estimated = len(rows) + bad_rows
        report.status = (
            RecoveryStatus.FULL
            if bad_rows == 0 and inconsistent == 0
            else RecoveryStatus.PARTIAL
        )
        report.recovery_notes.append(
            f"Recovered {len(rows)} CSV rows"
            + (f", {bad_rows} rows skipped" if bad_rows else "")
        )


class _SqliteRecovery:
    """Recover data from corrupt SQLite databases."""

    @staticmethod
    def recover(path: str, report: RecoveryReport) -> None:
        report.detected_format = "sqlite"
        recovered: Dict[str, List[Dict]] = {}
        errors: List[str] = []

        # ── Try 1: normal connection ───────────────────────────────────────
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            tables = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]

            for table in tables:
                try:
                    rows = cursor.execute(f"SELECT * FROM [{table}]").fetchall()
                    recovered[table] = [dict(r) for r in rows]
                    report.recovered_records += len(rows)
                except sqlite3.DatabaseError as e:
                    errors.append(f"Table '{table}' unreadable: {e}")

            conn.close()

            if errors:
                report.corruption_types.append(CorruptionType.MISSING_DATA)
                report.warnings.extend(errors)
                report.status = RecoveryStatus.PARTIAL
            else:
                report.status = RecoveryStatus.FULL

            report.data = recovered
            report.recovery_notes.append(
                f"Read {len(tables)} tables, {report.recovered_records} rows total"
            )
            return

        except sqlite3.DatabaseError as e:
            errors.append(f"Normal open failed: {e}")

        # ── Try 2: recover mode (SQLite 3.29+) ────────────────────────────
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("PRAGMA writable_schema=ON")
            # Dump whatever pages are readable
            dump_lines: List[str] = []
            try:
                src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                for line in src.iterdump():
                    dump_lines.append(line)
                src.close()
            except sqlite3.DatabaseError as dump_err:
                errors.append(f"Dump failed: {dump_err}")

            if dump_lines:
                for line in dump_lines:
                    try:
                        conn.execute(line)
                    except sqlite3.OperationalError:
                        pass
                conn.commit()

                cursor = conn.cursor()
                tables = [
                    row[0]
                    for row in cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                for table in tables:
                    try:
                        rows = cursor.execute(
                            f"SELECT * FROM [{table}]"
                        ).fetchall()
                        recovered[table] = [
                            dict(zip([d[0] for d in cursor.description], r))
                            for r in rows
                        ]
                        report.recovered_records += len(rows)
                    except sqlite3.DatabaseError as te:
                        errors.append(f"Dump-recovered table '{table}' failed: {te}")

                conn.close()

                if recovered:
                    report.data = recovered
                    report.status = RecoveryStatus.PARTIAL
                    report.corruption_types.append(CorruptionType.PARTIAL_WRITE)
                    report.recovery_notes.append(
                        f"Recovered via dump: {len(tables)} tables, "
                        f"{report.recovered_records} rows"
                    )
                    report.warnings.extend(errors)
                    return

        except Exception as e:
            errors.append(f"Dump-recovery failed: {e}")

        report.errors.extend(errors)
        report.status = RecoveryStatus.FAILED
        report.corruption_types.append(CorruptionType.UNKNOWN)


class _TextRecovery:
    """Recover plain text, logs, and Python source files."""

    @staticmethod
    def recover(text: str, path: str, report: RecoveryReport) -> None:
        ext = Path(path).suffix.lower()
        report.detected_format = "python" if ext == ".py" else "text"

        lines = text.splitlines()
        clean_lines: List[str] = []
        bad_lines = 0

        for line in lines:
            # Drop lines that are mostly non-printable (binary noise)
            printable = sum(1 for c in line if c.isprintable() or c in "\t ")
            if len(line) == 0 or printable / max(len(line), 1) >= 0.7:
                clean_lines.append(line)
            else:
                bad_lines += 1

        if bad_lines:
            report.corruption_types.append(CorruptionType.BINARY_NOISE)
            report.warnings.append(
                f"{bad_lines} lines dropped due to binary noise"
            )

        report.data = "\n".join(clean_lines)
        report.raw_text = report.data
        report.recovered_records = len(clean_lines)
        report.total_records_estimated = len(lines)
        report.status = (
            RecoveryStatus.FULL if bad_lines == 0 else RecoveryStatus.PARTIAL
        )
        report.recovery_notes.append(
            f"Recovered {len(clean_lines)}/{len(lines)} lines"
        )


class _BinaryRecovery:
    """Last-resort recovery for unknown binary files."""

    @staticmethod
    def recover(raw: bytes, report: RecoveryReport) -> None:
        report.detected_format = "binary"

        # Hex dump of first 512 bytes
        preview = raw[:512]
        hex_lines: List[str] = []
        for i in range(0, len(preview), 16):
            chunk = preview[i : i + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "." for b in chunk
            )
            hex_lines.append(f"{i:04x}  {hex_part:<48}  {ascii_part}")

        # Entropy estimate (rough)
        if raw:
            freq = [0] * 256
            for b in raw:
                freq[b] += 1
            import math
            entropy = -sum(
                (f / len(raw)) * math.log2(f / len(raw))
                for f in freq
                if f > 0
            )
        else:
            entropy = 0.0

        # Extract printable ASCII strings (min length 4)
        strings = re.findall(rb'[ -~]{4,}', raw)
        decoded_strings = [s.decode("ascii", errors="replace") for s in strings]

        report.data = {
            "hex_preview": "\n".join(hex_lines),
            "entropy_bits": round(entropy, 3),
            "extracted_strings": decoded_strings[:100],  # cap at 100
            "file_size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        report.corruption_types.append(CorruptionType.UNKNOWN)
        report.status = RecoveryStatus.MINIMAL
        report.recovery_notes.append(
            f"Binary file: entropy={entropy:.2f} bits/byte, "
            f"{len(decoded_strings)} ASCII strings extracted"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN READER
# ═══════════════════════════════════════════════════════════════════════════════

# SQLite magic bytes
_SQLITE_MAGIC = b"SQLite format 3\x00"

# Mapping of extensions to format hints
_EXT_FORMAT: Dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".txt": "text",
    ".log": "text",
    ".md": "text",
    ".py": "python",
    ".js": "text",
    ".ts": "text",
    ".html": "text",
    ".xml": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
}


def _sniff_format(raw: bytes, path: str) -> str:
    """Determine file format from magic bytes and extension."""
    if raw[:16] == _SQLITE_MAGIC:
        return "sqlite"
    ext = Path(path).suffix.lower()
    return _EXT_FORMAT.get(ext, "binary")


class CorruptFileReader:
    """
    Attempts to read and recover data from corrupt or damaged files.

    Usage
    -----
    reader = CorruptFileReader()
    report = reader.read("path/to/broken.json")
    print(report.summary())
    if report.data:
        # use report.data — format depends on file type
        ...
    """

    def read(self, path: str) -> RecoveryReport:
        """
        Attempt to recover data from a potentially corrupt file.

        Parameters
        ----------
        path : str
            Path to the file to recover.

        Returns
        -------
        RecoveryReport
            Always returns a report. Check `report.status` and `report.data`.
        """
        path = str(path)
        report = RecoveryReport(
            file_path=path,
            file_size_bytes=0,
            detected_format="unknown",
        )

        # ── Read raw bytes ─────────────────────────────────────────────────
        raw, file_size = _read_raw(path)
        report.file_size_bytes = file_size
        report.bytes_readable = len(raw)
        report.bytes_lost = max(0, file_size - len(raw))

        if not raw:
            report.errors.append("File is empty or completely unreadable")
            report.status = RecoveryStatus.FAILED
            logger.error(f"[CorruptFileReader] {path}: empty/unreadable")
            return report

        if report.bytes_lost > 0:
            report.corruption_types.append(CorruptionType.TRUNCATED)
            report.warnings.append(
                f"{report.bytes_lost} bytes could not be read"
            )

        # ── Detect format ──────────────────────────────────────────────────
        fmt = _sniff_format(raw, path)

        # ── SQLite: handle without decoding ───────────────────────────────
        if fmt == "sqlite":
            _SqliteRecovery.recover(path, report)
            logger.info(
                f"[CorruptFileReader] {path}: {report.status.value} "
                f"({report.recovered_records} records)"
            )
            return report

        # ── Binary: no text decoding possible ─────────────────────────────
        if fmt == "binary":
            _BinaryRecovery.recover(raw, report)
            logger.info(
                f"[CorruptFileReader] {path}: binary, "
                f"entropy={report.data.get('entropy_bits')}"
            )
            return report

        # ── Text-based formats: decode first ──────────────────────────────
        encoding = _detect_encoding(raw)
        report.encoding_detected = encoding
        text, decode_warnings = _safe_decode(raw, encoding)
        report.raw_text = text
        report.warnings.extend(decode_warnings)
        if decode_warnings:
            report.corruption_types.append(CorruptionType.ENCODING_ERROR)

        if fmt in ("json", "jsonl"):
            _JsonRecovery.recover(text, report)
        elif fmt in ("csv", "tsv"):
            _CsvRecovery.recover(text, path, report)
        else:
            # text, python, yaml, etc.
            _TextRecovery.recover(text, path, report)

        logger.info(
            f"[CorruptFileReader] {path}: {report.status.value} "
            f"({report.recovered_records} records, "
            f"format={report.detected_format})"
        )
        return report

    def read_many(self, paths: List[str]) -> Dict[str, RecoveryReport]:
        """
        Attempt recovery on multiple files.

        Returns a dict mapping each path to its RecoveryReport.
        """
        return {p: self.read(p) for p in paths}

    def is_readable(self, path: str) -> bool:
        """Quick check — returns True if at least minimal data was recovered."""
        report = self.read(path)
        return report.status != RecoveryStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def recover_file(path: str) -> RecoveryReport:
    """
    Module-level shortcut for one-off recovery.

    Example
    -------
    report = recover_file("broken_state.json")
    if report.data:
        process(report.data)
    """
    return CorruptFileReader().read(path)
