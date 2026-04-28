"""
CLI interface for the Software NVMe Engine Configuration Manager.

Usage:
    python -m nvme_engine.control.config_cli validate <config_file>
    python -m nvme_engine.control.config_cli list-templates
    python -m nvme_engine.control.config_cli show-template <name>
    python -m nvme_engine.control.config_cli apply <config_file> --persist-to <output.json>
"""

from __future__ import annotations

import argparse
import json
import sys

from nvme_engine.control.config_manager import ConfigManager


def cmd_validate(args: argparse.Namespace, mgr: ConfigManager) -> int:
    try:
        configs = mgr.load_from_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    all_valid, error_map = mgr.validate_all(configs)
    if all_valid:
        print(f"OK: {len(configs)} device(s) validated successfully.")
        return 0
    else:
        for idx, errors in error_map.items():
            name = configs[idx].get("name", f"<device {idx}>")
            print(f"Device [{idx}] '{name}' has {len(errors)} error(s):")
            for err in errors:
                print(f"  - {err}")
        return 1


def cmd_list_templates(args: argparse.Namespace, mgr: ConfigManager) -> int:
    templates = mgr.list_templates()
    print("Available templates:")
    for t in templates:
        print(f"  {t}")
    return 0


def cmd_show_template(args: argparse.Namespace, mgr: ConfigManager) -> int:
    try:
        tmpl = mgr.get_template(args.name)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(tmpl, indent=2))
    return 0


def cmd_apply(args: argparse.Namespace, mgr: ConfigManager) -> int:
    try:
        configs = mgr.load_from_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading config: {exc}", file=sys.stderr)
        return 1

    all_valid, error_map = mgr.validate_all(configs)
    if not all_valid:
        for idx, errors in error_map.items():
            name = configs[idx].get("name", f"<device {idx}>")
            print(f"Validation error in device [{idx}] '{name}':")
            for err in errors:
                print(f"  - {err}")
        return 1

    if args.persist_to:
        mgr.persist(configs, args.persist_to)
        print(f"Config persisted to: {args.persist_to}")

    print(f"Applied {len(configs)} device configuration(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nvme-config",
        description="Software NVMe Engine configuration management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # validate
    p_validate = sub.add_parser("validate", help="Validate a config file")
    p_validate.add_argument("file", help="Path to JSON or YAML config file")

    # list-templates
    sub.add_parser("list-templates", help="List available device profile templates")

    # show-template
    p_show = sub.add_parser("show-template", help="Show a device profile template")
    p_show.add_argument("name", help="Template name")

    # apply
    p_apply = sub.add_parser("apply", help="Validate and optionally persist a config")
    p_apply.add_argument("file", help="Path to JSON or YAML config file")
    p_apply.add_argument("--persist-to", metavar="OUTPUT", help="Persist validated config to this JSON file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mgr = ConfigManager()

    dispatch = {
        "validate": cmd_validate,
        "list-templates": cmd_list_templates,
        "show-template": cmd_show_template,
        "apply": cmd_apply,
    }
    return dispatch[args.command](args, mgr)


if __name__ == "__main__":
    sys.exit(main())
