"""Explicit result operations; importing this module never launches inference."""
from __future__ import annotations

import argparse
from typing import Callable
from pathlib import Path
import json
from .validation import validate_run



def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate completed result artifacts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    details = validate_run(args.config.resolve(), args.run_dir.resolve(), args.dataset.resolve())
    print(json.dumps(details, ensure_ascii=False, sort_keys=True))
    return 0 if details["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "validate": "Validate config, query identity and metric coverage (not SIA).",
        "status": "Inspect configured non-SIA runs for completion and stale outputs.",
        "publish": "Explicitly publish a run or mirror an artifact.",
    }
    for command, description in descriptions.items():
        subparsers.add_parser(command, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    handler: Callable[[list[str]], int]
    if args.command == "validate":
        handler = validate_main
    elif args.command == "status":
        from .status import status_main as handler
    else:
        from .sync import main as handler
    return handler(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
