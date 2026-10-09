# Install VLAC

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/vlacconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites) for the chosen deployment path.
Keep VLAC's FlashAttention/Torch dependencies isolated; see [environment details](../environments.md).

```bash
bash envs/source.sh vlac
bash envs/setup.sh vlac --plan
bash envs/setup.sh vlac
```

The [official source](https://github.com/InternRobotics/VLAC) is downloaded to
`.baseline-sources/vlac` at the mapped commit; setup attaches `evo_vlac`.
The runtime is `.model-envs/vlac`. Source download excludes example videos and
does not download model weights.

## 2. Download a checkpoint

The primary example is [InternRobotics/VLAC](https://huggingface.co/InternRobotics/VLAC),
the 2B checkpoint:

```bash
hf download InternRobotics/VLAC \
  --revision f44e552324944f80a63ab0083879f0bfafe01516 \
  --local-dir checkpoints/VLAC-2B
```

The [8B variant](https://huggingface.co/InternRobotics/VLAC-8b) has recorded
revision `dd94c057b0274b88e7228e8eaa73c03a7e8fa1d5`; use
`checkpoints/VLAC-8B` for that variant. Keep the model's Python configuration,
modeling and tokenizer files: this checkpoint uses custom code. Review that
code and its license before running it; file availability is not a security audit.

## 3. Select references, configure and run

```bash
cp configs/vlacconfigs.yaml configs/vlac-local.yaml
```

Set `python: <REPO>/.model-envs/vlac/bin/python`,
`checkpoint: <REPO>/checkpoints/VLAC-2B`, plus `data`, `output`, `gpu`, `tasks`
and `metrics`. Omit `reference_data` for zero-shot; add an explicit disjoint
reference input for one-shot. The adapter selects this from reference presence,
not a second adapter or a separate runtime.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/vlac-local.yaml
```

## Verification and common failures

Source download, setup plan and fresh-source imports passed in an existing
runtime. This does not establish a clean installation or complete 2B/8B metric
coverage. Missing `evo_vlac` indicates a source/runtime path issue; FlashAttention
errors indicate environment compatibility. One-shot reference requirements are
not satisfied by downloading the model alone.
