from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from typing import Any

from vmbmk.data.dataset import Dataset
from .errors import VMBMKError
from vmbmk.runner.run import run_evaluation
from vmbmk.inference.dispatch import run_inference
from vmbmk.adapters.registry import ADAPTERS


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_secrets_file() -> None:
    """Load optional process secrets without persisting them in run artifacts.

    Only an explicit ``VMBMK_SECRETS_FILE`` is read. Existing environment
    values always win, so schedulers can explicitly supply a different key.
    """
    configured_path = os.environ.get("VMBMK_SECRETS_FILE")
    if not configured_path:
        return
    path = Path(configured_path)
    if not path.exists():
        raise VMBMKError(f"configured secrets file does not exist: {path}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise VMBMKError(f"secrets file is not a regular file: {path}")
    if metadata.st_mode & 0o077:
        raise VMBMKError(f"secrets file must not be group/world readable: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or not _ENV_NAME.fullmatch(name):
            raise VMBMKError(f"invalid secrets entry at {path}:{line_number}")
        try:
            fields = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise VMBMKError(f"invalid quoting at {path}:{line_number}") from exc
        if len(fields) != 1:
            raise VMBMKError(f"invalid secrets value at {path}:{line_number}")
        os.environ.setdefault(name, fields[0])


def build_parser() -> argparse.ArgumentParser:
    """Build the public command interface without loading model packages."""
    parser = argparse.ArgumentParser(prog="vmbmk")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a VMBMK dataset")
    validate.add_argument("dataset")
    infer = commands.add_parser("infer", help="execute value/compare/subtask queries")
    infer.add_argument("--data", required=True)
    infer.add_argument("--queries", required=True)
    infer.add_argument(
        "--model",
        required=True,
        choices=tuple(ADAPTERS),
    )
    infer.add_argument("--gpu", type=int)
    infer.add_argument("--python", required=True)
    infer.add_argument("--checkpoint")
    infer.add_argument("--use-lora", action="store_true")
    infer.add_argument("--shot-mode", choices=("zero_shot", "one_shot"))
    infer.add_argument("--reference-data")
    infer.add_argument(
        "--reference-view"
    )
    infer.add_argument("--ref-num", type=int)
    infer.add_argument("--source-root")
    infer.add_argument("--view")
    infer.add_argument("--output", required=True)
    run = commands.add_parser("run", help="run one configured evaluation")
    run.add_argument("config")
    run.add_argument("--sia-stage", choices=("all", "generate", "judge"))
    run.add_argument("--mode", choices=("normal", "rerun"), default="normal")
    commands.add_parser("batch", help="schedule configured evaluations with retries and incremental completion", add_help=False)
    commands.add_parser("data", help="inspect dataset assets", add_help=False)
    commands.add_parser("results", help="validate, inspect and publish results", add_help=False)
    commands.add_parser("visualize", help="render value and cycle curves", add_help=False)
    judge = commands.add_parser("sia-judge", help="judge saved SIA responses without GPU inference")
    judge.add_argument("responses")
    judge.add_argument(
        "--data",
        required=True,
        help="dataset root for episode selection and diverse subtask mapping",
    )
    judge.add_argument("--output", required=True)
    judge.add_argument("--model", help="judge model (default: DEEPSEEK_MODEL or deepseek-v4-pro)")
    judge.add_argument("--base-url")
    return parser


def _inference_model(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI model options without changing model-specific validation."""
    model = {
        "adapter": args.model,
        "python": args.python,
    }
    if args.checkpoint is not None:
        model["checkpoint"] = args.checkpoint
    if args.model == "vlac":
        if args.ref_num is not None:
            model["ref_num"] = args.ref_num
        if args.shot_mode is not None:
            model["shot_mode"] = args.shot_mode
        if args.reference_data is not None:
            model.update({
                "reference_data": args.reference_data,
            })
        if args.reference_view is not None:
            model["reference_view"] = args.reference_view
    elif (
        args.shot_mode is not None
        or args.reference_data is not None
        or args.reference_view is not None
        or args.ref_num is not None
    ):
        raise VMBMKError("VLAC options require --model vlac")
    if args.model == "liv":
        if args.source_root is None:
            raise VMBMKError("--source-root is required for --model liv")
        model["source_root"] = args.source_root
        if args.view is not None:
            model["view"] = args.view
    elif args.source_root is not None or args.view is not None:
        raise VMBMKError("--source-root/--view require --model liv")
    if args.use_lora:
        if args.model != "procvlm":
            raise VMBMKError("--use-lora requires --model procvlm")
        model["use_lora"] = True
    return model


def main(argv: list[str] | None = None) -> int:
    try:
        _load_secrets_file()
        arguments = sys.argv[1:] if argv is None else argv
        if arguments and arguments[0] in {"data", "results", "visualize"}:
            from importlib import import_module

            modules = {
                "data": "vmbmk.tools.data",
                "results": "vmbmk.tools.results.cli",
                "visualize": "vmbmk.tools.visualization.cli",
            }
            return import_module(modules[arguments[0]]).main(arguments[1:])
        if arguments and arguments[0] == "batch":
            from vmbmk.runner.batch import main as run_batch

            return run_batch(arguments[1:])
        args = build_parser().parse_args(arguments)
        if args.command == "validate":
            Dataset.load(args.dataset)
            print("valid")
            return 0
        if args.command == "infer":
            model = _inference_model(args)
            print(
                run_inference(
                    args.data,
                    args.queries,
                    model,
                    args.output,
                    gpu=args.gpu,
                )
            )
            return 0
        if args.command == "sia-judge":
            from vmbmk.metrics.sia.evaluation import judge_sia_responses, _save_json
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            metrics = judge_sia_responses(
                args.responses, output / "operations.jsonl",
                model={"sia_model": args.model, "sia_base_url": args.base_url},
                cache_dir=output / "judgements",
                dataset=Dataset.load(args.data, metrics=["sia"]),
            )
            _save_json(output / "metrics.json", {"sia": metrics})
            print(output)
            return 0
        print(
            run_evaluation(
                args.config,
                sia_stage=args.sia_stage,
                **({"execution_mode": "rerun"} if args.mode == "rerun" else {}),
            )
        )
        return 0
    except (OSError, ValueError, VMBMKError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
