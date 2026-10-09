# Install TOPReward

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/toprewardconfigs.yaml)

TOPReward has **one adapter**, selected with `model: topreward`. The YAML
`model_options.backend` switches Qwen/Molmo; runtime and checkpoint must match.
Do not install both model stacks into one environment or use a Qwen checkpoint
with the Molmo backend.

## 1. Prepare the chosen runtime

Both choices use the shared TOPReward lock project; its reference Python version
and current installer constraints are listed in the [shared prerequisites](README.md#prerequisites). Native
FlashAttention builds require `nvcc`, `c++` and a compatible Torch/CUDA pair;
TorchCodec requires loadable FFmpeg 4–7 shared libraries. Follow the
[shared prerequisites](README.md#prerequisites) and [environment details](../environments.md).

For Qwen:

```bash
bash envs/source.sh topreward
bash envs/setup.sh topreward --plan
bash envs/setup.sh topreward
```

The [TOPReward source](https://github.com/TOPReward/TOPReward) is downloaded to
`.baseline-sources/topreward`, and the runtime is `.model-envs/topreward`.

For Molmo, no TOPReward source checkout is required by this backend:

```bash
bash envs/setup.sh topreward_molmo --plan
bash envs/setup.sh topreward_molmo
```

The runtime is `.model-envs/topreward_molmo`; this is an environment selector,
not another YAML model name.

## 2. Download the corresponding checkpoint

Qwen uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct):

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir checkpoints/Qwen3-VL-8B-Instruct
```

Molmo uses [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B):

```bash
hf download allenai/Molmo2-8B \
  --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
  --local-dir checkpoints/Molmo2-8B
```

Both revisions were checked against public metadata on 2026-10-06, not certified
as archived benchmark revisions. Preserve tokenizer/processor files and any
custom model code, not just weight shards. Model download does not install
FlashAttention, TorchCodec or the runtime.

## 3. Configure and run

```bash
cp configs/toprewardconfigs.yaml configs/topreward-local.yaml
```

Select these four fields together:

| Field | Qwen | Molmo |
| --- | --- | --- |
| `model_options.backend` | `qwen` | `molmo` |
| `python` | `<REPO>/.model-envs/topreward/bin/python` | `<REPO>/.model-envs/topreward_molmo/bin/python` |
| `checkpoint` | `<REPO>/checkpoints/Qwen3-VL-8B-Instruct` | `<REPO>/checkpoints/Molmo2-8B` |
| Starting `batch_size` | `2` | `1` |

Keep `model: topreward` in both cases. These batch sizes are conservative
template choices, not GPU-memory guarantees. Fill `data`, `output`, `gpu`,
`tasks` and eligible `metrics` before starting.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/topreward-local.yaml
```

## Verification and common failures

The Qwen source and imports were checked in an existing runtime; both runtime
plans were checked. This does not establish new-environment installation or
Molmo checkpoint inference. The checkpoint supplies the model; TOPReward reward
logic remains in the selected implementation. Keep backend selection consistent
instead of restoring separate adapter files or renaming the public model key.
