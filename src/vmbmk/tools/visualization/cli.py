"""Render curves through one entry point with explicit query-history modes."""
from __future__ import annotations

import argparse
from typing import Callable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "value": "Sample one task's episodes and run value inference.",
        "selection": "Run inference for trajectories selected in a JSON file.",
        "cycle": "Run continuous forward/reverse inference and render Cycle-VOC curves.",
    }
    for command, description in descriptions.items():
        subparsers.add_parser(command, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    handler: Callable[[list[str]], int]
    if args.command == "value":
        from .value_curves import main as handler
    elif args.command == "selection":
        from .selected_curves import main as handler
    else:
        from .cycle_curves import main as handler
    return handler(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
