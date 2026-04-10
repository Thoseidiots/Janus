"""
janus_synthetic_sql.py
=======================
Synthetic SQL for Janus — a query layer over Janus's own data.

Instead of a real database, this lets Janus (and you) query all of
Janus's internal state using SQL-like syntax:

    SELECT * FROM work_log WHERE status = 'delivered' ORDER BY revenue DESC
    SELECT SUM(revenue) FROM work_log WHERE category = 'code'
    SELECT * FROM identity WHERE key LIKE 'heuristic%'
    SELECT * FROM escalations WHERE status = 'open'
    SELECT * FROM messages WHERE from = 'janus' LIMIT 10

The "tables" are backed by Janus's JSON/JSONL files — no real DB needed.
This is also a training data generator for teaching Avus SQL reasoning.

Usage:
    from janus_synthetic_sql import JanusSQL
    db = JanusSQL()
    results = db.query("SELECT * FROM work_log WHERE status = 'delivered'")
    print(results)

    # Generate training pairs for Avus
    pairs = db.generate_training_pairs(1000)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Virtual table definitions ─────────────────────────────────────────────────

class VirtualTable:
    """A table backed by a JSON/JSONL file."""

    def __init__(self, name: str, file_path: Path, is_jsonl: bool = False):
        self.name      = name
        self.file_path = file_path
        self.is_jsonl  = is_jsonl

    def load(self) -> List[dict]:
        if not self.file_path.exists():
            return []
        try:
            if self.is_jsonl:
                rows = []
                for line in self.file_path.read_text().strip().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
                return rows
            else:
                data = json.loads(self.file_path.read_text())
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    # Flatten nested dicts into rows
                    rows = []
                    for key, val in data.items():
                        if isinstance(val, dict):
                            rows.append({"key": key, **val})
                        else:
                            rows.append({"key": key, "value": val})
                    return rows
                return [data]
        except Exception:
            return []


# ── SQL parser (minimal) ──────────────────────────────────────────────────────

@dataclass
class ParsedQuery:
    select_cols: List[str]    # ["*"] or ["col1", "col2"]
    table:       str
    where:       Optional[str]
    order_by:    Optional[str]
    order_dir:   str          # "ASC" | "DESC"
    limit:       Optional[int]
    aggregates:  Dict[str, str]  # {"SUM(revenue)": "revenue"}


def parse_sql(sql: str) -> ParsedQuery:
    """Parse a minimal SQL SELECT statement."""
    sql = sql.strip().rstrip(";")

    # Extract LIMIT
    limit = None
    m = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
    if m:
        limit = int(m.group(1))
        sql   = sql[:m.start()].strip()

    # Extract ORDER BY
    order_by  = None
    order_dir = "ASC"
    m = re.search(r'\bORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', sql, re.IGNORECASE)
    if m:
        order_by  = m.group(1)
        order_dir = (m.group(2) or "ASC").upper()
        sql       = sql[:m.start()].strip()

    # Extract WHERE
    where = None
    m = re.search(r'\bWHERE\s+(.+)', sql, re.IGNORECASE)
    if m:
        where = m.group(1).strip()
        sql   = sql[:m.start()].strip()

    # Extract FROM
    m = re.search(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    if not m:
        raise ValueError(f"No FROM clause in: {sql}")
    table = m.group(1).lower()
    sql   = sql[:m.start()].strip()

    # Extract SELECT columns
    m = re.match(r'SELECT\s+(.+)', sql, re.IGNORECASE)
    if not m:
        raise ValueError(f"No SELECT clause in: {sql}")
    cols_str = m.group(1).strip()

    # Detect aggregates
    aggregates = {}
    agg_pattern = re.compile(r'(SUM|COUNT|AVG|MAX|MIN)\((\w+|\*)\)', re.IGNORECASE)
    for agg_m in agg_pattern.finditer(cols_str):
        key = agg_m.group(0)
        aggregates[key] = (agg_m.group(1).upper(), agg_m.group(2))

    select_cols = [c.strip() for c in cols_str.split(",")]

    return ParsedQuery(
        select_cols = select_cols,
        table       = table,
        where       = where,
        order_by    = order_by,
        order_dir   = order_dir,
        limit       = limit,
        aggregates  = aggregates,
    )


# ── WHERE evaluator ───────────────────────────────────────────────────────────

def eval_where(row: dict, where: str) -> bool:
    """Evaluate a WHERE clause against a row."""
    if not where:
        return True

    # Handle AND / OR by splitting (simple, left-to-right)
    if re.search(r'\bAND\b', where, re.IGNORECASE):
        parts = re.split(r'\bAND\b', where, flags=re.IGNORECASE)
        return all(eval_where(row, p.strip()) for p in parts)

    if re.search(r'\bOR\b', where, re.IGNORECASE):
        parts = re.split(r'\bOR\b', where, flags=re.IGNORECASE)
        return any(eval_where(row, p.strip()) for p in parts)

    # LIKE
    m = re.match(r"(\w+)\s+LIKE\s+'([^']*)'", where, re.IGNORECASE)
    if m:
        col, pattern = m.group(1), m.group(2)
        val = str(row.get(col, "")).lower()
        pat = pattern.replace("%", ".*").replace("_", ".").lower()
        return bool(re.search(pat, val))

    # IS NULL / IS NOT NULL
    m = re.match(r"(\w+)\s+IS\s+(NOT\s+)?NULL", where, re.IGNORECASE)
    if m:
        col    = m.group(1)
        is_not = bool(m.group(2))
        val    = row.get(col)
        return (val is not None) if is_not else (val is None)

    # IN (...)
    m = re.match(r"(\w+)\s+IN\s+\(([^)]+)\)", where, re.IGNORECASE)
    if m:
        col    = m.group(1)
        values = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        return str(row.get(col, "")) in values

    # Comparison operators
    m = re.match(r"(\w+)\s*(=|!=|<>|>=|<=|>|<)\s*'?([^']*)'?", where)
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3).strip("'\"")
        row_val = row.get(col, "")

        # Try numeric comparison
        try:
            rv, v = float(str(row_val)), float(val)
            if op == "=":  return rv == v
            if op in ("!=", "<>"): return rv != v
            if op == ">":  return rv > v
            if op == "<":  return rv < v
            if op == ">=": return rv >= v
            if op == "<=": return rv <= v
        except (ValueError, TypeError):
            pass

        # String comparison
        rv = str(row_val).lower()
        v  = val.lower()
        if op == "=":  return rv == v
        if op in ("!=", "<>"): return rv != v
        if op == ">":  return rv > v
        if op == "<":  return rv < v

    return True


# ── Main SQL engine ───────────────────────────────────────────────────────────

class JanusSQL:
    """
    SQL query engine over Janus's internal data files.
    No real database — pure Python over JSON/JSONL.
    """

    def __init__(self, base_dir: Path = Path(".")):
        self._tables: Dict[str, VirtualTable] = {}
        self._register_tables(base_dir)

    def _register_tables(self, base: Path):
        """Register all known Janus data files as virtual tables."""
        defs = [
            ("work_log",      base / "work_log.jsonl",              True),
            ("messages",      base / "janus_messages.jsonl",        True),
            ("escalations",   base / "escalations.json",            False),
            ("finance",       base / "finance_ledger.json",         False),
            ("identity",      base / "janus_identity.json",         False),
            ("scheduler",     base / "scheduler_state.json",        False),
            ("ceo_state",     base / "ceo_state.json",              False),
            ("payment_ledger",base / "payment_ledger.json",         False),
            ("heuristics",    base / "janus_identity.json",         False),
            ("decisions",     base / "janus_identity.json",         False),
        ]
        for name, path, is_jsonl in defs:
            self._tables[name] = VirtualTable(name, path, is_jsonl)

    def register_table(self, name: str, file_path: Path, is_jsonl: bool = False):
        """Register a custom table."""
        self._tables[name] = VirtualTable(name, file_path, is_jsonl)

    def list_tables(self) -> List[str]:
        return list(self._tables.keys())

    def describe(self, table: str) -> List[str]:
        """Return column names for a table (from first row)."""
        tbl = self._tables.get(table)
        if not tbl:
            return []
        rows = tbl.load()
        if not rows:
            return []
        return list(rows[0].keys())

    def query(self, sql: str) -> List[dict]:
        """Execute a SQL SELECT query and return results."""
        try:
            q = parse_sql(sql)
        except ValueError as e:
            return [{"error": str(e)}]

        # Load table
        tbl = self._tables.get(q.table)
        if not tbl:
            return [{"error": f"Unknown table: {q.table}. Available: {', '.join(self._tables)}"}]

        rows = tbl.load()

        # Special handling for nested tables
        rows = self._flatten_table(q.table, rows)

        # WHERE filter
        if q.where:
            rows = [r for r in rows if eval_where(r, q.where)]

        # Aggregates
        if q.aggregates:
            return self._apply_aggregates(rows, q)

        # ORDER BY
        if q.order_by:
            reverse = q.order_dir == "DESC"
            rows.sort(
                key=lambda r: (r.get(q.order_by) is None,
                               _coerce(r.get(q.order_by, ""))),
                reverse=reverse,
            )

        # LIMIT
        if q.limit is not None:
            rows = rows[:q.limit]

        # SELECT columns
        if q.select_cols != ["*"]:
            rows = [{c: r.get(c) for c in q.select_cols} for r in rows]

        return rows

    def _flatten_table(self, table: str, rows: List[dict]) -> List[dict]:
        """Flatten nested structures for specific tables."""
        if table == "heuristics":
            # Extract heuristics from identity
            flat = []
            for row in rows:
                if row.get("key") == "heuristics" or "heuristics" in row:
                    heuristics = row.get("value") or row.get("heuristics", [])
                    if isinstance(heuristics, list):
                        flat.extend(heuristics)
            return flat if flat else rows

        if table == "decisions":
            flat = []
            for row in rows:
                if row.get("key") == "decisions" or "decisions" in row:
                    decisions = row.get("value") or row.get("decisions", [])
                    if isinstance(decisions, list):
                        flat.extend(decisions)
            return flat if flat else rows

        if table == "finance":
            # finance_ledger has entries and approvals
            flat = []
            for row in rows:
                entries = row.get("entries", [])
                if isinstance(entries, list):
                    flat.extend(entries)
            return flat if flat else rows

        if table == "escalations":
            flat = []
            for row in rows:
                escs = row.get("escalations", [])
                if isinstance(escs, list):
                    flat.extend(escs)
            return flat if flat else rows

        return rows

    def _apply_aggregates(self, rows: List[dict], q: ParsedQuery) -> List[dict]:
        result = {}
        for agg_expr, (func, col) in q.aggregates.items():
            values = []
            for row in rows:
                val = row.get(col) if col != "*" else 1
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    if func == "COUNT":
                        values.append(1)

            if func == "COUNT":
                result[agg_expr] = len(rows) if col == "*" else len(values)
            elif func == "SUM":
                result[agg_expr] = round(sum(values), 2)
            elif func == "AVG":
                result[agg_expr] = round(sum(values) / len(values), 2) if values else 0
            elif func == "MAX":
                result[agg_expr] = max(values) if values else None
            elif func == "MIN":
                result[agg_expr] = min(values) if values else None

        return [result]

    def query_pretty(self, sql: str) -> str:
        """Execute query and return formatted string output."""
        rows = self.query(sql)
        if not rows:
            return "No results."
        if "error" in rows[0]:
            return f"Error: {rows[0]['error']}"

        # Format as table
        cols = list(rows[0].keys())
        widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows))
                  for c in cols}
        widths = {c: min(w, 40) for c, w in widths.items()}

        header = " | ".join(str(c).ljust(widths[c]) for c in cols)
        sep    = "-+-".join("-" * widths[c] for c in cols)
        lines  = [header, sep]
        for row in rows:
            line = " | ".join(str(row.get(c, ""))[:widths[c]].ljust(widths[c]) for c in cols)
            lines.append(line)

        return "\n".join(lines) + f"\n\n({len(rows)} row{'s' if len(rows) != 1 else ''})"


# ── Training data generator ───────────────────────────────────────────────────

class SQLTrainingGenerator:
    """
    Generates (natural language question → SQL query) pairs
    for training Avus to understand SQL reasoning.
    """

    _TEMPLATES = [
        # Work log queries
        ("How many jobs has Janus completed?",
         "SELECT COUNT(*) FROM work_log WHERE status = 'delivered'"),
        ("What is the total revenue earned?",
         "SELECT SUM(actual_revenue) FROM work_log WHERE status = 'delivered'"),
        ("Show the most recent completed jobs",
         "SELECT title, actual_revenue, created_at FROM work_log WHERE status = 'delivered' ORDER BY created_at DESC LIMIT 5"),
        ("Which job category earns the most?",
         "SELECT category, SUM(actual_revenue) FROM work_log WHERE status = 'delivered' ORDER BY SUM(actual_revenue) DESC LIMIT 1"),
        ("Show all failed jobs",
         "SELECT title, notes, created_at FROM work_log WHERE status = 'failed'"),
        ("What jobs are currently active?",
         "SELECT title, category, bid_amount FROM work_log WHERE status = 'active'"),

        # Finance queries
        ("What is the current profit?",
         "SELECT SUM(amount) FROM finance WHERE type = 'income' AND status = 'confirmed'"),
        ("Show all pending invoices",
         "SELECT description, amount, counterparty FROM finance WHERE status = 'pending' AND type = 'income'"),
        ("What are the total expenses?",
         "SELECT SUM(amount) FROM finance WHERE type = 'expense' AND status = 'confirmed'"),

        # Escalation queries
        ("Are there any open escalations?",
         "SELECT title, trigger, created_at FROM escalations WHERE status = 'open'"),
        ("Show escalations that expired without response",
         "SELECT title, default_option FROM escalations WHERE status = 'expired'"),

        # Message queries
        ("Show recent messages from Janus",
         "SELECT message, timestamp FROM messages WHERE category != 'notification' ORDER BY timestamp DESC LIMIT 10"),
        ("Show all alert messages",
         "SELECT message, timestamp FROM messages WHERE category = 'escalation' ORDER BY timestamp DESC"),

        # Identity/heuristics
        ("What has Janus learned?",
         "SELECT lesson, confidence FROM heuristics ORDER BY confidence DESC LIMIT 10"),
        ("Show high-confidence heuristics",
         "SELECT lesson, confidence, used_count FROM heuristics WHERE confidence >= 0.7 ORDER BY confidence DESC"),
        ("What decisions has Janus made recently?",
         "SELECT decision, reasoning, decided_at FROM decisions ORDER BY decided_at DESC LIMIT 5"),
    ]

    def generate(self, samples: int = 1000) -> List[Tuple[str, str]]:
        """Generate (question, SQL) training pairs."""
        import random
        pairs = list(self._TEMPLATES)

        # Generate variations
        while len(pairs) < samples:
            base_q, base_sql = random.choice(self._TEMPLATES)
            # Add LIMIT variation
            n = random.randint(3, 20)
            if "LIMIT" not in base_sql:
                pairs.append((f"{base_q} (show {n})", f"{base_sql} LIMIT {n}"))
            # Add ORDER variation
            if "ORDER BY" not in base_sql and "FROM" in base_sql:
                pairs.append((
                    f"{base_q}, sorted by most recent",
                    f"{base_sql} ORDER BY created_at DESC"
                ))

        return pairs[:samples]

    def save(self, pairs: List[Tuple[str, str]], path: str = "training_data/sql_pairs.jsonl"):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for q, sql in pairs:
                f.write(json.dumps({"question": q, "sql": sql}) + "\n")
        print(f"[SQL Training] Saved {len(pairs)} pairs to {out}")


def _coerce(val: Any) -> Any:
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val).lower()


# ── Module-level singleton ────────────────────────────────────────────────────

_db: Optional[JanusSQL] = None

def get_db() -> JanusSQL:
    global _db
    if _db is None:
        _db = JanusSQL()
    return _db


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Janus Synthetic SQL")
    parser.add_argument("--query",   type=str, help="Run a SQL query")
    parser.add_argument("--tables",  action="store_true", help="List available tables")
    parser.add_argument("--describe",type=str, metavar="TABLE", help="Show table columns")
    parser.add_argument("--generate",type=int, metavar="N", help="Generate N SQL training pairs")
    args = parser.parse_args()

    db = JanusSQL()

    if args.tables:
        print("Available tables:")
        for t in db.list_tables():
            cols = db.describe(t)
            print(f"  {t} ({', '.join(cols[:5])}{'...' if len(cols) > 5 else ''})")

    elif args.describe:
        cols = db.describe(args.describe)
        print(f"Columns in {args.describe}:")
        for c in cols:
            print(f"  {c}")

    elif args.query:
        print(db.query_pretty(args.query))

    elif args.generate:
        gen   = SQLTrainingGenerator()
        pairs = gen.generate(args.generate)
        gen.save(pairs)
        print(f"Generated {len(pairs)} SQL training pairs")
        print("\nSample pairs:")
        import random
        for q, sql in random.sample(pairs, min(5, len(pairs))):
            print(f"  Q: {q}")
            print(f"  SQL: {sql}\n")

    else:
        # Interactive mode
        print("Janus SQL — type queries or 'exit' to quit")
        print("Tables:", ", ".join(db.list_tables()))
        print()
        while True:
            try:
                sql = input("sql> ").strip()
                if sql.lower() in ("exit", "quit"):
                    break
                if sql:
                    print(db.query_pretty(sql))
                    print()
            except (EOFError, KeyboardInterrupt):
                break
