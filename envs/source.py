"""Fetch pinned baseline source and validate the import roots used by setup."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / ".baseline-sources"


def model_spec(model: str) -> dict[str, Any]:
    """Read the shared source and runtime contract for one baseline."""
    models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    if model not in models:
        raise ValueError(f"unknown baseline {model!r}; choose from {', '.join(models)}")
    return models[model]


def validate_source(source: Path, row: dict[str, Any]) -> None:
    """Require the upstream files and additional import directories used at runtime."""
    for name in row["source"]:
        if not (source / name).is_file():
            raise ValueError(f"missing baseline source: {source / name}")
    for name in row.get("paths", []):
        if not (source / name).is_dir():
            raise ValueError(f"missing baseline import directory: {source / name}")


def git_output(source: Path, *args: str) -> str:
    """Read checkout metadata without changing the source."""
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True,
    ).strip()


def check_checkout(source: Path, row: dict[str, Any]) -> None:
    """Reuse only an unchanged checkout of the configured public revision."""
    if source.is_symlink() or not (source / ".git").is_dir():
        raise ValueError(f"not a managed Git checkout; refusing to overwrite: {source}")
    upstream = row["upstream"]
    if git_output(source, "remote", "get-url", "origin") != upstream["repository"]:
        raise ValueError(f"source repository differs from models.yaml: {source}")
    if git_output(source, "rev-parse", "HEAD") != upstream["revision"]:
        raise ValueError(f"source revision differs from models.yaml: {source}")
    if git_output(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"source has local changes; refusing to reuse: {source}")
    validate_source(source, row)


def resolve_source(
    model: str, row: dict[str, Any], source: Path | None, root: Path = SOURCE_ROOT,
) -> Path | None:
    """Resolve an explicit import root or the baseline downloader's default checkout."""
    if not row.get("source"):
        if source is not None:
            raise ValueError(f"{model}: --source is not used")
        return None
    if source is not None:
        source = source.expanduser().resolve()
        validate_source(source, row)
        return source
    source = root.expanduser().absolute() / model
    if not source.exists():
        raise ValueError(
            f"missing {model} source: {source}; run bash envs/source.sh {model} "
            f"--root {root} first, or supply --source IMPORT_ROOT"
        )
    check_checkout(source, row)
    return source.resolve()


def fetch_source(model: str, row: dict[str, Any], root: Path) -> Path:
    """Fetch source-only sparse paths, without LFS weights or recursive submodules."""
    upstream = row["upstream"]
    revision = upstream["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{model}: upstream revision must be a full Git commit ID")
    root = root.expanduser().resolve()
    destination = root / model
    if destination.exists() or destination.is_symlink():
        check_checkout(destination, row)
        return destination
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        environment.pop(name, None)

    def run(*args: str, patterns: str | None = None) -> None:
        subprocess.run(
            ["git", "-C", str(destination), *args], env=environment,
            input=patterns, text=True, check=True,
        )

    subprocess.run(["git", "init", str(destination)], env=environment, check=True)
    try:
        run("config", "core.autocrlf", "false")
        run("remote", "add", "origin", upstream["repository"])
        run("-c", "protocol.version=2", "fetch", "--filter=blob:none", "--depth=1",
            "origin", revision)
        patterns = ["/*", "!/*/"]
        patterns.extend(f"/{name}/" for name in upstream["directories"])
        patterns.extend([
            "!/liv/assets/", "!/evo_vlac/examples/", "!*.pt", "!*.pth",
            "!*.ckpt", "!*.safetensors", "!*.bin", "!*.mp4", "!*.mov",
            "!*.tar", "!*.zip",
            "!*.tar.gz", "!*.tgz", "!*.npy", "!*.npz", "!*.h5",
            "!*.hdf5", "!*.parquet", "!*.webm", "!*.avi", "!*.gif",
        ])
        run("sparse-checkout", "set", "--no-cone", "--stdin",
            patterns="\n".join(patterns) + "\n")
        run("checkout", "--detach", revision)
        check_checkout(destination, row)
    except (OSError, ValueError, subprocess.CalledProcessError):
        print(f"Incomplete source retained for inspection: {destination}")
        raise
    return destination


def main(argv: list[str] | None = None) -> int:
    """Download the baseline selected by name; --plan performs no network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT,
                        help="source checkout parent (default: repository/.baseline-sources)")
    parser.add_argument("--plan", action="store_true", help="show source inputs without downloading")
    args = parser.parse_args(argv)
    try:
        row = model_spec(args.baseline)
        if not row.get("source"):
            print(f"{args.baseline}: no external baseline source is needed; run envs/setup.sh.")
            return 0
        upstream = row["upstream"]
        destination = args.root.expanduser().resolve() / args.baseline
        print(json.dumps({
            "baseline": args.baseline, **upstream, "source_root": str(destination),
            "required_files": row["source"],
            "import_paths": [str(destination), *[
                str(destination / name) for name in row.get("paths", [])
            ]],
        }, indent=2), flush=True)
        if not args.plan:
            source = fetch_source(args.baseline, row, args.root)
            print(f"Verified source: {source}")
            print(f"Next: bash envs/setup.sh {args.baseline} --source \"{source}\" --plan")
            if args.baseline == "liv":
                print(f"model_options.source_root: {source}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
