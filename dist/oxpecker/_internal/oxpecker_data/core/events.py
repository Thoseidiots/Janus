from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DebugEvent:
    kind: str                                     # 'continued', 'paused', 'stopped', 'exited'
    message: str = ''
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StackFrame:
    id: str
    name: str
    source_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

@dataclass
class Variable:
    name: str
    value: Any
    type_name: Optional[str] = None

@dataclass
class Issue:
    severity: str                                 # 'error' | 'warning' | 'info'
    language: str
    message: str
    source_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    frame: Optional[StackFrame] = None
    variables: List[Variable] = field(default_factory=list)
    fix_suggestion: Optional[str] = None         # human-readable fix hint
    fix_patch: Optional[str] = None              # unified-diff patch when available
    complexity_tier: Optional[str] = None        # Tier 1 | Tier 2 | Tier 3
    repair_stage: str = "localized"              # localized | proposed | validated | applied
    approval: str = "pending"                    # pending | approved | needs-review
