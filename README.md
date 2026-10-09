# RoboValue

RoboValue diagnoses robotic value models through simulation and real-world
evaluation: success discrimination, instruction alignment, progress and memory,
failure/recovery, value stability and cross-solution consistency. It is an
evaluation package, not a single-model inference service or a leaderboard alone.

The repository provides adapters, metric implementations, configuration
templates and result tools. **Datasets and credentials are not included.** LIV's
three required assets are provided under `checkpoints/LIV` (weights via Git LFS);
other checkpoints must be supplied separately. Missing or unsupported evaluation
coverage is N/A, not zero.

## Install one baseline

Start with the **[baseline installation guides](docs/baselines/README.md)**.
Each guide specifies the environment scripts, checkpoint download locations,
local directories, configuration fields and verification limits.

| Baseline | Installation | Configuration |
| --- | --- | --- |
| RoboMeter | [Guide](docs/baselines/robometer.md) | [robometerconfigs.yaml](configs/robometerconfigs.yaml) |
| RoboDopamine | [Guide](docs/baselines/robodopamine.md) | [robodopamineconfigs.yaml](configs/robodopamineconfigs.yaml) |
| ProcVLM | [Guide](docs/baselines/procvlm.md) | [procvlmconfigs.yaml](configs/procvlmconfigs.yaml) |
| RoboReward | [Guide](docs/baselines/roboreward.md) | [roborewardconfigs.yaml](configs/roborewardconfigs.yaml) |
| VLAC | [Guide](docs/baselines/vlac.md) | [vlacconfigs.yaml](configs/vlacconfigs.yaml) |
| TOPReward (Qwen / Molmo) | [Guide](docs/baselines/topreward.md) | [toprewardconfigs.yaml](configs/toprewardconfigs.yaml) |
| RoboFAC | [Guide](docs/baselines/robofac.md) | [robofacconfigs.yaml](configs/robofacconfigs.yaml) |
| RynnValue | [Guide](docs/baselines/rynnvalue.md) | [rynnvalueconfigs.yaml](configs/rynnvalueconfigs.yaml) |
| LIV | [Guide](docs/baselines/liv.md) | [livconfigs.yaml](configs/livconfigs.yaml) |
| FailSafe-labeled SIA integration (official equivalence unverified) | [Provenance/setup](docs/baselines/failsafe.md) | [failsafeconfigs.yaml](configs/failsafeconfigs.yaml) |

The provided lock-based installer targets **Linux x86_64**; this is its current
validation scope, not a claim that each model only runs on that platform.
The lightweight CLI supports Python 3.10+. Keep model environments separate.
[Shared prerequisites](docs/baselines/README.md#prerequisites)
cover Python, uv and checkpoint download tooling. GVL and ReWiND are excluded.

## Put data and weights here

Recommended layout, relative to your checkout:

```text
vmbmk_codebase/
  data/
    dataset_sim/                 supplied simulation dataset
    dataset_real/                supplied real-world dataset
    reference/                   separate one-shot reference dataset
  checkpoints/<model-name>/      weights, tokenizer and model configuration
  .baseline-sources/<baseline>/  downloaded upstream code
  .model-envs/<runtime>/         isolated Python environments
  .cache/huggingface/            optional installation/download cache
  configs/                      editable run configurations
  output/<config-stem>/          generated evaluation artifacts
```

`data:` must point to **one dataset root containing task directories**, not the
parent `data/` folder or a folder of unannotated videos. See the
**[data preparation guide](docs/data.md)** for the required structure and asset
paths. The small files already under `data/` are not a complete dataset release.

These storage locations are conventions, not hard-coded requirements. External
storage is supported through configuration paths. Downloaded code, datasets,
weights, environments, caches and outputs must stay out of Git, except for the
explicitly distributed LIV assets. See the [LIV guide](docs/baselines/liv.md)
for Git LFS checkout instructions and upstream download alternatives.

## Configure and run

After completing your baseline guide, copy its template and replace all
`/path/to/...` placeholders. Choose available GPUs, actual task IDs and eligible
metrics; the default task is only an example. For example:

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
# Edit python, checkpoint, data, output, gpu, tasks and metrics first.
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/roboreward-local.yaml
```

Validation checks dataset schema and assets; it does not load a model or score
anything. A successful evaluation writes `metrics.json` and a saved `config.yaml`
under the configured output root, with metric-specific operation/provenance
artifacts. See [configuration and commands](docs/configuration.md).

The shell entry points have separate responsibilities:

| Command | Responsibility |
| --- | --- |
| `bash envs/source.sh BASELINE` | Download required upstream code at a fixed commit; not weights |
| `bash envs/setup.sh BASELINE --plan` | Check source paths, lock inputs and platform prerequisites; no installation |
| `bash envs/setup.sh BASELINE` | Install locked dependencies and attach source import paths; no model loading |
| `bash vmbmk.sh run CONFIG` | Run the selected evaluation; workers use YAML `python` |

RoboMeter requires an explicit GPU for its final installation probe; LIV needs
an additional CLIP RN50 asset. Follow their guides rather than treating these
four commands as a universal copy-and-paste installation.

## Read next

- [Environment details](docs/environments.md): lockfiles, custom roots, native builds and installation boundaries.
- [Metric contracts](docs/metric_alignment.md): scoring, cohorts, units and protocol changes.
- [Result interfaces](docs/result_publication.md): validation, publication and SIA intermediate outputs.
- [Developer guide](docs/developer_guide.md): package layout, call flow and modification sites.
- [Testing and evidence](docs/testing.md): CPU tests and bounded model checks, not all-model reproduction.

## Verification and scope

Source acquisition and path checks were verified for the six source-dependent
baselines. Fresh-source imports passed in existing ProcVLM, VLAC, TOPReward Qwen
and FailSafe environments. Clean-machine installation for every baseline has
**not** been established; LIV and RoboMeter new-environment validation remains
pending. Individual guides distinguish available setup from tested execution.

CPU scoring/interface tests require no weights, videos or GPU:

```bash
PYTHONPATH=src:tests python -m pytest tests -q
```

See [testing](docs/testing.md) for prerequisites, inference evidence and known
limitations. For help, open an issue in the
[project repository](https://github.com/duzhengye-droid/vmbmk_codebase) with the
baseline, config without credentials, metric, environment versions and relevant
traceback. Do not attach private data, weights or API tokens.
License and third-party redistribution review remains required before a public
release; availability of a checkpoint does not grant redistribution rights.
