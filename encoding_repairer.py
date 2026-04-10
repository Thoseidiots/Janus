"""
EncodingRepairer — Oxpecker RepairPlugin that replaces Unicode smart quotes
with their ASCII equivalents in Python source files.

Targets:
  U+2018 ' → '  (LEFT SINGLE QUOTATION MARK)
  U+2019 ' → '  (RIGHT SINGLE QUOTATION MARK)
  U+201C " → "  (LEFT DOUBLE QUOTATION MARK)
  U+201D " → "  (RIGHT DOUBLE QUOTATION MARK)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Minimal RepairPlugin base class (defined inline so Oxpecker need not be
# installed in the test environment).
# ---------------------------------------------------------------------------

class RepairPlugin:
    """Base interface for Oxpecker repair plugins."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def supported_languages(self) -> list[str]:
        raise NotImplementedError

    def repair(self, source: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Smart-quote replacement table
# ---------------------------------------------------------------------------

_REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK  →  apostrophe
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK →  apostrophe
    "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK  →  quotation mark
    "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK →  quotation mark
}

_SMART_QUOTES = frozenset(_REPLACEMENTS)


# ---------------------------------------------------------------------------
# EncodingRepairer
# ---------------------------------------------------------------------------

class EncodingRepairer(RepairPlugin):
    """
    Oxpecker plugin that replaces Unicode smart-quote characters with their
    ASCII equivalents throughout a Python source file.

    Smart quotes in Python source are always erroneous in code positions
    (they cause SyntaxError when used as string delimiters or operators).
    Triple-quoted docstrings that contain prose with intentional typographic
    quotes are an edge case, but in practice any smart quote inside a Python
    source file should be replaced — the file cannot be imported as-is if
    smart quotes appear outside of string literals, and string literals that
    use smart quotes as delimiters are themselves syntax errors.

    Therefore this plugin replaces ALL occurrences of the four smart-quote
    code points unconditionally, which is the correct behaviour for Python
    source repair.
    """

    @property
    def name(self) -> str:
        return "encoding_repairer"

    @property
    def supported_languages(self) -> list[str]:
        return ["python"]

    def repair(self, source: str) -> str:
        """Replace all smart-quote characters with ASCII equivalents.

        Parameters
        ----------
        source:
            The full text of a Python source file.

        Returns
        -------
        str
            The repaired source with U+2018, U+2019, U+201C, U+201D replaced
            by their ASCII counterparts.  If the source contains none of those
            characters the original string is returned unchanged (no copy).
        """
        # Fast path: skip the replacement loop when no smart quotes present.
        if not any(ch in source for ch in _SMART_QUOTES):
            return source

        result = source
        for smart, ascii_equiv in _REPLACEMENTS.items():
            result = result.replace(smart, ascii_equiv)
        return result
