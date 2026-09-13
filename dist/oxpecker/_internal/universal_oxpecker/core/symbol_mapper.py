from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

@dataclass(frozen=True)
class Symbol:
    name: str
    language: str
    source_path: str
    line: Optional[int] = None

@dataclass
class SymbolLink:
    source: Symbol
    target: Symbol
    kind: str  # 'ffi', 'rpc', 'binding'

class CrossLanguageSymbolMapper:
    """
    Tracks and maps symbols across different languages (XLANG-FFI-009).
    """
    def __init__(self) -> None:
        self._symbols: Dict[str, Set[Symbol]] = {}
        self._links: List[SymbolLink] = []

    def register_symbol(self, symbol: Symbol) -> None:
        if symbol.name not in self._symbols:
            self._symbols[symbol.name] = set()
        self._symbols[symbol.name].add(symbol)

    def link_symbols(self, source: Symbol, target: Symbol, kind: str) -> None:
        self.register_symbol(source)
        self.register_symbol(target)
        self._links.append(SymbolLink(source, target, kind))

    def find_linked_symbols(self, symbol: Symbol) -> List[Symbol]:
        linked = []
        for link in self._links:
            if link.source == symbol:
                linked.append(link.target)
            elif link.target == symbol:
                linked.append(link.source)
        return linked

    def resolve_ffi_calls(self, symbol_name: str) -> List[SymbolLink]:
        """Find all FFI links for a given symbol name."""
        return [link for link in self._links if (link.source.name == symbol_name or link.target.name == symbol_name) and link.kind == 'ffi']
