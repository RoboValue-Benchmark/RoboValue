# Baseline installation guides

[Back to README](../../README.md) | [Data preparation](../data.md) | [Environment internals](../environments.md)

These guides are for organizers running selected baselines in an evaluation
deployment. Submitting your own model does not require installing these
baselines; follow the [service and adapter guide](../api.md).

Install **one baseline at a time**. Each family guide follows the same sequence:
prepare its runtime, download the required model assets, configure local paths,
then validate and run. Setup scripts download dependencies, not weights or data.
LIV's required assets are distributed separately through Git LFS; see its guide.
FailSafe is an exception: reuse its original working source/runtime/checkpoint;
the original source commit is not currently verified as publicly downloadable.

| Family | Runtime selector | External source | Guide |
| --- | --- | --- | --- |
| RoboMeter | `robometer` | Required | [Install](robometer.md) |
| RoboDopamine | `robodopamine` | Not required | [Install](robodopamine.md) |
| ProcVLM | `procvlm` | Required | [Install](procvlm.md) |
| RoboReward | `roboreward` | Not required | [Install](roboreward.md) |
| VLAC | `vlac` | Required | [Install](vlac.md) |
| TOPReward | `topreward` / `topreward_molmo` | Qwen only | [Install](topreward.md) |
| RoboFAC | `robofac` | Not required | [Install](robofac.md) |
| RynnValue | `rynnvalue` | Not required | [Install](rynnvalue.md) |
| LIV | `liv` | Required | [Install](liv.md) |
| FailSafe-labeled integration (SIA only) | `failsafe` | Original local source retained; public fetch unverified | [Reuse/provenance](failsafe.md) |

GVL, ReWiND and FailSafe simulation/training environments are not supported setup
targets. A family's adapter does not imply that every metric is supported.

## Prerequisites

These examples follow the supplied Linux x86_64 lock-based deployment path.
That is the installer's current target, not a universal model platform restriction.
Windows users can read/edit configs locally; Windows model installation is not
validated by these locks. Run commands from the repository root.
On shared servers, use the authorized container or execution environment and
persistent storage; do not change system drivers or packages through these scripts.

Required tools:

- A launcher Python 3.10+ with PyYAML; `VMBMK_PYTHON` selects it. This is distinct
  from each model's YAML `python` interpreter.
- Python 3.10+ for the benchmark. Use Python **3.10.x** for the supplied model
  locks, or **3.11.x** for TOPReward; no specific patch version is required.
- uv with frozen `export --format pylock.toml` and `pip sync` support. No exact
  uv release is required; select your existing executable with `--uv` if needed.
- Git for source-dependent baselines; network access to the mapped public repos
  and locked dependency artifacts during installation.
- The baseline-specific CUDA/build/FFmpeg requirements listed in its guide.

## Compatible environments

Distinguish model requirements from this repository's reproducibility controls:

| Setting | Supplied setup path | Meaning |
| --- | --- | --- |
| Platform | Linux x86_64 | Lock/build target; other platforms are not validated |
| Python | 3.10.x; TOPReward 3.11.x | Compatible series for the supplied locks; patch versions are unrestricted |
| uv | Frozen pylock export and pip sync | Required functionality, not an exact release number |
| Torch/CUDA/FFmpeg | Baseline-specific | Preserve a compatible dependency/native-library combination |

Use your existing compatible Python and uv; setup does not install either
implicitly. Pass `--python /path/to/python` when needed. If an interpreter is
missing, explicitly provision the appropriate series, for example
`uv python install 3.10` or `uv python install 3.11`. Older uv releases that lack
the export functionality need an explicit tooling update, not an arbitrary exact
version match. Do not modify shared system tooling automatically.

The installer reads the Python series from `.python-version`, allowing different
patch versions while keeping the supplied dependency locks unchanged. Wider
Python-series support is not claimed: native wheels and dependency constraints
must be validated before using another series. The final probe records the actual
Python and Torch versions in the runtime's `setup.json`.

An already working model-compatible environment does not need to be recreated
solely to match a patch number in these guides. Configure its Python executable
directly and ensure the benchmark plus required upstream code are importable.
That is a custom integration path: run dependency/import and small inference
checks, record actual versions, and do not call it the same locked deployment.
Changing dependency versions/locks requires separate validation; upgrading all
baselines together is not a shortcut to compatibility.

Checkpoint examples use the Hugging Face `hf` CLI. Install it **separately from
model environments**, for example `uv tool install huggingface-hub`, and check
`hf --help`. See the [official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).
Authentication, if required by a model, belongs in `hf auth login` or your secure
environment, not YAML or this repository. Never paste tokens into committed files.

## Source, runtime and checkpoint roots

Default locations are `.baseline-sources/<baseline>`, `.model-envs/<runtime>`
and `checkpoints/<model-name>`. Source and runtime defaults are anchored to the
repository, even if the shell is elsewhere. `hf download --local-dir` destinations
are relative to the shell, so run download examples from the repository root.

For another persistent parent, use matching source roots:

```bash
bash envs/source.sh liv --root /path/to/project/sources
bash envs/setup.sh liv --source-root /path/to/project/sources \
  --root /path/to/project/model-envs --plan
```

`--source IMPORT_ROOT` can select an existing external checkout. Default managed
checkouts must match the pinned repository/commit and be clean. Do not move them
after installing: runtimes reference their original paths. Upstream code remains
outside `src/vmbmk`, so it is not accidentally packaged or redistributed.

Model cards and remote-code files are part of some checkpoints. Do not download
only `.safetensors` shards and omit tokenizer/configuration/Python files. Commands
use full checkpoint snapshots unless a guide explicitly selects inference assets.
Check the model license and available disk space before any large download.

## Revisions and offline execution

Download examples pin checkpoint revisions. Some were recorded from existing
benchmark assets; others were checked against public repository metadata on
**2026-10-06**. A newly checked revision is not proof that historical paper scores
used it. Guides mark the difference. Keep the selected revisions with your run
record; do not silently replace them with `main` to match a result.

`vmbmk.sh` defaults to Hugging Face offline mode. Complete explicit checkpoint
and auxiliary-model downloads before inference. If upstream configuration uses
a Hub model ID, pre-populate the Hub cache as instructed; a separate `--local-dir`
download alone does not make that ID resolvable offline. Do not confuse the Hub
model cache with uv's dependency-installation cache.

## Configure and verify

Copy the family YAML to a new config, preserving the template. Replace all
placeholders, select valid tasks/domains/metrics, and set:

| Field | Meaning |
| --- | --- |
| `python` | Full path to `.model-envs/<runtime>/bin/python` printed by setup |
| `checkpoint` | Local model directory from the guide, not the Hub repository ID |
| `data` | Supplied dataset root containing task directories |
| `output` | Result root, such as `<REPO>/output` |
| `gpu` | Explicit device(s) authorized and available on your machine |

`<REPO>` means your checkout's absolute path; it is not an environment variable
or literal YAML value. Top-level relative paths resolve from the config directory;
nested adapter paths should follow the guide. Use [data preparation](../data.md).

Installation checks must not be overstated:

1. `source.sh` success checks the pinned source and required files.
2. `setup.sh --plan` checks paths, locks and platform prerequisites, not downloads.
3. `setup.sh` success prints a runtime and writes `setup.json` with checked status
   after dependency/import and small CPU checks; it does not load weights.
4. `vmbmk.sh run CONFIG` is the actual model/data integration check. Start with
   a small eligible task/metric selection; scripts do not auto-limit inference.

On failure, read the traceback and retain the failed source/runtime for inspection.
Do not overwrite an unowned or changed runtime. Use a new `--root` for a clean
installation. All fresh-environment installations are **pending verification**
unless explicitly stated otherwise; existing-environment import evidence is narrower.
