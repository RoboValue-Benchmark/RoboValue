"""Create isolated baseline runtimes from repository locks, without model loading."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from source import SOURCE_ROOT, model_spec, resolve_source

ROOT = Path(__file__).resolve().parent


def check_ffmpeg(versions: list[int]) -> None:
    """Require a loadable shared-library set supported by the locked TorchCodec."""
    libraries = {
        4: (58, 58, 58, 56, 3, 5),
        5: (59, 59, 59, 57, 4, 6),
        6: (60, 60, 60, 58, 4, 7),
        7: (61, 61, 61, 59, 5, 8),
        8: (62, 62, 62, 60, 6, 9),
    }
    names = ("avcodec", "avformat", "avdevice", "avutil", "swresample", "swscale")
    for version in versions:
        try:
            for name, abi in zip(names, libraries[version]):
                ctypes.CDLL(f"lib{name}.so.{abi}")
        except OSError:
            continue
        return
    raise ValueError(
        f"requires loadable FFmpeg shared libraries from one of {versions}; "
        "an ffmpeg executable alone is insufficient"
    )


def build_code(builds: list[str]) -> str:
    """Check the installed Torch/toolkit pair before building native packages."""
    return fr"""
import json
import re
import subprocess
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME

if CUDA_HOME is None or torch.version.cuda is None:
    raise RuntimeError("native build requires a CUDA toolkit and CUDA-enabled Torch")
nvcc = Path(CUDA_HOME) / "bin/nvcc"
output = subprocess.check_output([str(nvcc), "--version"], text=True)
match = re.search(r"release (\d+)\.(\d+)", output)
if match is None:
    raise RuntimeError("cannot identify CUDA toolkit version: " + output)
toolkit = tuple(map(int, match.groups()))
torch_cuda = tuple(map(int, torch.version.cuda.split(".")[:2]))
if toolkit[0] != torch_cuda[0]:
    raise RuntimeError(f"CUDA toolkit {{toolkit}} and Torch CUDA {{torch_cuda}} differ in major version")
if "flash-attn" in {builds!r} and toolkit < (11, 7):
    raise RuntimeError("locked FlashAttention requires CUDA toolkit 11.7 or newer")
subprocess.run(["ninja", "--version"], check=True)
print(json.dumps({{"cuda_toolkit": toolkit, "torch_cuda": torch_cuda}}))
"""


def check_code(model: str, imports: list[str], version: str) -> str:
    """Build the offline probe executed by the selected model interpreter."""
    adapter = "topreward" if model == "topreward_molmo" else model
    return f"""
import importlib
import io
import json
import platform
import sys
from PIL import Image
import numpy as np
import torch
import yaml

assert platform.python_version().rsplit(".", 1)[0] == {version!r}, platform.python_version()
for name in {imports!r}:
    importlib.import_module(name)
importlib.import_module("vmbmk.adapters." + {adapter!r})
image = Image.new("RGB", (8, 8), (12, 34, 56))
stream = io.BytesIO()
image.save(stream, format="PNG")
stream.seek(0)
assert Image.open(stream).getpixel((0, 0)) == (12, 34, 56)
assert yaml.safe_load("value: 1") == {{"value": 1}}
assert np.array([1, 2]).sum() == 3
assert torch.tensor([1, 2]).sum().item() == 3
print(json.dumps({{"python": sys.executable, "python_version": platform.python_version(),
                  "torch": torch.__version__,
                  "cuda_build": torch.version.cuda, "imports": {imports!r}}}))
"""


def prepare(
    model: str, root: Path, source: Path | None, source_root: Path = SOURCE_ROOT,
) -> dict[str, Any]:
    """Check external boundaries before creating or changing an environment."""
    row = model_spec(model)
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError(
            "baseline locks target Linux x86_64; no alternative runtime is substituted"
        )
    source = resolve_source(model, row, source, source_root)
    if row.get("ffmpeg"):
        check_ffmpeg(row["ffmpeg"])
    if row.get("build"):
        for tool in ("nvcc", "c++"):
            if shutil.which(tool) is None:
                raise ValueError(f"{model} requires {tool} on PATH for its locked native build")
    project = ROOT / row["project"]
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        if not (project / name).is_file():
            raise ValueError(f"missing environment input: {project / name}")
    reference_version = (project / ".python-version").read_text().strip()
    if not re.fullmatch(r"3\.\d+(?:\.\d+)?", reference_version):
        raise ValueError(f"invalid Python version in {project / '.python-version'}")
    version = ".".join(reference_version.split(".")[:2])
    paths = [ROOT.parent / "src"]
    source_hashes = {}
    if source is not None:
        paths.extend([source, *(source / name for name in row.get("paths", []))])
        for name in row["source"]:
            path = source / name
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = root.resolve() / model
    if target.is_symlink():
        raise ValueError(f"runtime directory must not be a symlink: {target}")
    if source is not None and (
        source.is_relative_to(target) or target.is_relative_to(source)
    ):
        raise ValueError("source checkout and runtime directory must not contain one another")
    if target == project or project.is_relative_to(target):
        raise ValueError("runtime directory must not contain its locked project")
    identity = {
        "model": model,
        "project": row["project"],
        "python": version,
        "source": str(source) if source else None,
        "source_files": source_hashes,
        "locks": {
            name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in ("pyproject.toml", "uv.lock", ".python-version")
        },
        "setup": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mapping": hashlib.sha256((ROOT / "models.yaml").read_bytes()).hexdigest(),
        "build": hashlib.sha256((ROOT / "build.txt").read_bytes()).hexdigest(),
    }
    record = target / "setup.json"
    if target.exists():
        if not record.is_file():
            raise ValueError(f"refusing to overwrite an unowned runtime: {target}")
        if json.loads(record.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"runtime inputs changed: {target}; choose a new --root")
    return {
        "target": target, "project": project, "version": version,
        "paths": paths, "identity": identity, "row": row,
    }


def _runtime_lock_sections(text: str, builds: list[str]) -> tuple[str, list[str], list[str]]:
    """Partition the pinned uv export format while preserving package entries verbatim."""
    header, *packages = text.split("[[packages]]")
    native = []
    regular = []
    for package in packages:
        name = re.search(r'^name = "([^"]+)"', package, re.MULTILINE).group(1)
        if "\narchive = " in package:
            package = re.sub(r'^version = .*\n', "", package, flags=re.MULTILINE)
        (native if name in builds else regular).append("[[packages]]" + package)
    return header, regular, native


def deploy(
    plan: dict[str, Any], uv: str, python: str | None, probe_gpu: int | None = None,
) -> Path:
    """Export frozen dependencies, install them, attach source paths and probe."""
    if plan["row"].get("probe_gpu_required") and probe_gpu is None:
        raise ValueError(
            "RoboMeter's upstream Unsloth import requires a GPU; "
            "select --probe-gpu GPU explicitly before installing"
        )
    if probe_gpu is not None and probe_gpu < 0:
        raise ValueError("--probe-gpu must be a non-negative GPU index")
    target = plan["target"]
    if shutil.which(uv) is None:
        raise ValueError(f"uv executable not found: {uv}; install uv explicitly first")
    env = dict(os.environ)
    for key in (
        "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"
    ):
        env.pop(key, None)
    env.update(
        UV_LINK_MODE="copy",
        UV_PYTHON_DOWNLOADS="never", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="",
    )
    builds = plan["row"].get("build", [])
    if builds:
        for key in list(env):
            if key.startswith("DS_BUILD_"):
                env.pop(key)
        env.setdefault("MAX_JOBS", "4")
    if "deepspeed" in builds:
        env.update(DS_ACCELERATOR="cuda", DS_BUILD_OPS="0", DS_SKIP_CUDA_CHECK="0")
    if "flash-attn" in builds:
        env["FLASH_ATTENTION_FORCE_BUILD"] = "TRUE"
        env["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "FALSE"
    interpreter = python or plan["version"]
    probe = subprocess.check_output(
        [uv, "python", "find", "--no-python-downloads", interpreter], env=env, text=True,
    ).strip()
    version = subprocess.check_output(
        [probe, "-c", "import platform; print(platform.python_version())"], env=env, text=True,
    ).strip()
    if version.rsplit(".", 1)[0] != plan["version"]:
        raise ValueError(
            f"this locked runtime targets Python {plan['version']}.x, "
            f"found {version}: {probe}"
        )
    if not target.exists():
        subprocess.run(
            [uv, "venv", "--no-python-downloads", "--python", probe, str(target)],
            env=env, check=True,
        )
    (target / "setup.json").write_text(
        json.dumps({"identity": plan["identity"], "status": "installing"}, indent=2),
        encoding="utf-8",
    )
    runtime = target / "bin/python"
    env["PATH"] = str(runtime.parent) + os.pathsep + env.get("PATH", "")
    env["UV_CACHE_DIR"] = str(target / ".cache/uv")
    with tempfile.TemporaryDirectory(prefix=".setup-", dir=target) as directory:
        work = Path(directory)
        lock = work / "pylock.runtime.toml"
        command = [
            uv, "export", "--project", str(plan["project"]), "--frozen",
            "--no-emit-local", "--no-dev", "--no-python-downloads",
            "--format", "pylock.toml", "--output-file", str(lock),
        ]
        subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
        header, regular, native = _runtime_lock_sections(lock.read_text(encoding="utf-8"), builds)
        lock.write_text(header + "".join(regular + native), encoding="utf-8")
        if builds:
            if len(native) != len(builds):
                raise ValueError(f"native build packages missing from the lock: {builds}")
            base = work / "pylock.base.toml"
            base.write_text(header + "".join(regular), encoding="utf-8")
            subprocess.run(
                [uv, "pip", "sync", "--python", str(runtime), str(base)],
                env=env, check=True,
            )
            names = {
                re.search(r'^name = "([^"]+)"', package, re.MULTILINE).group(1)
                for package in regular
            }
            tools = [
                line for line in (ROOT / "build.txt").read_text().splitlines()
                if line.split("==")[0] not in names
            ]
            if tools:
                subprocess.run(
                    [uv, "pip", "install", "--python", str(runtime), "--no-deps", *tools],
                    env=env, check=True,
                )
            subprocess.run(
                [str(runtime), "-c", build_code(builds)], env=env, check=True,
            )
            native_file = work / "pylock.native.toml"
            native_file.write_text(header + "".join(native), encoding="utf-8")
            subprocess.run(
                [uv, "pip", "install", "--python", str(runtime), "--no-deps",
                 "--no-build-isolation", "--requirement", str(native_file)],
                env=env, check=True,
            )
        else:
            subprocess.run(
                [uv, "pip", "sync", "--python", str(runtime), str(lock)],
                env=env, check=True,
            )
    site = subprocess.check_output(
        [str(runtime), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        env=env, text=True,
    ).strip()
    site = Path(site).resolve()
    if not site.is_relative_to(target):
        raise ValueError(f"runtime site-packages escapes its directory: {site}")
    (site / "vmbmk_source.pth").write_text(
        "\n".join(map(str, plan["paths"])) + "\n", encoding="utf-8"
    )
    subprocess.run([uv, "pip", "check", "--python", str(runtime)], env=env, check=True)
    code = check_code(plan["identity"]["model"], plan["row"]["imports"], plan["version"])
    if probe_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(probe_gpu)
    result = subprocess.check_output(
        [str(runtime), "-c", code],
        env=env, text=True,
    )
    (target / "setup.json").write_text(
        json.dumps({
            "identity": plan["identity"], "status": "checked", "check": result.strip(),
        }, indent=2),
        encoding="utf-8",
    )
    return runtime


def main(argv: list[str] | None = None) -> int:
    """Run the single deployment entrypoint; --plan never installs anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("--root", type=Path, default=ROOT.parent / ".model-envs")
    parser.add_argument("--source", type=Path, help="upstream import root, not weights")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="parent used by source.sh; ignored when --source is explicit")
    parser.add_argument("--python", help="existing interpreter compatible with the runtime's Python series")
    parser.add_argument("--uv", default="uv", help="existing uv executable")
    parser.add_argument("--probe-gpu", type=int,
                        help="explicit GPU for the final import probe; required by RoboMeter")
    parser.add_argument("--plan", action="store_true", help="validate paths and show the plan only")
    args = parser.parse_args(argv)
    try:
        plan = prepare(args.baseline, args.root, args.source, args.source_root)
        print(json.dumps({
            "baseline": args.baseline, "project": str(plan["project"]),
            "python": plan["version"], "runtime": str(plan["target"]),
            "source_paths": list(map(str, plan["paths"])),
            "source_root": plan["identity"]["source"],
            "probe_gpu_required": bool(plan["row"].get("probe_gpu_required")),
            "steps": ["check", "create", "install locked", "probe"],
        }, indent=2))
        if not args.plan:
            print(f"python: {deploy(plan, args.uv, args.python, args.probe_gpu)}")
            if args.baseline == "liv":
                print(f"model_options.source_root: {plan['identity']['source']}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
