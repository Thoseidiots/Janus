"""
synthetic_databases/generate_all.py
=====================================
Runs all synthetic database generators and prints a summary.

Usage:
    python synthetic_databases/generate_all.py
    python synthetic_databases/generate_all.py --records 1000
"""

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Janus synthetic databases")
    parser.add_argument("--records", type=int, default=500,
                        help="Records per category per database (default: 500)")
    args = parser.parse_args()

    generators = [
        ("Conversation DB",  "janus_conversation_db", "build_database"),
        ("Reasoning DB",     "janus_reasoning_db",    "build_database"),
        ("Skills DB",        "janus_skills_db",       "build_database"),
        ("Humanity DB",      "janus_humanity_db",     "build_database"),
    ]

    print(f"\n{'='*60}")
    print(f"  Janus Synthetic Database Generator")
    print(f"  Records per category: {args.records}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for name, module_name, func_name in generators:
        print(f"[{name}]")
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(f"synthetic_databases.{module_name}")
            fn = getattr(mod, func_name)
            fn(n_per_category=args.records)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"{'='*60}")
    print(f"  All databases generated in {total_elapsed:.1f}s")
    print(f"\n  Output files:")
    for f in sorted(Path("synthetic_databases").glob("*.db")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<40} {size_kb:>8.1f} KB")
    for f in sorted(Path("synthetic_databases").glob("*.jsonl")):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"    {f.name:<40} {lines:>8,} records")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
