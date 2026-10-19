# Install RoboReward

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/roborewardconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites). This adapter needs no external
baseline source checkout. Its locked TorchCodec requires loadable **FFmpeg 4–8
shared libraries**, not just an `ffmpeg` executable; see [environment details](../environments.md).

```bash
bash envs/setup.sh roboreward --plan
bash envs/setup.sh roboreward
```

The runtime is `.model-envs/roboreward`. If the FFmpeg check fails, resolve the
authorized environment prerequisite before installing; do not change host
libraries or drivers as an automatic fallback.

## 2. Download a checkpoint

Choose [teetone/RoboReward-4B](https://huggingface.co/teetone/RoboReward-4B):

```bash
hf download teetone/RoboReward-4B \
  --revision 4dec8af8d89d214de7028d70f9f9aee3b3a44c3d \
  --local-dir checkpoints/RoboReward-4B
```

or [teetone/RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B):

```bash
hf download teetone/RoboReward-8B \
  --revision 3a185b4fce2b1253643105be1f234ae618b9732f \
  --local-dir checkpoints/RoboReward-8B
```

Both choices use the same `model: roboreward` adapter and runtime. These are
fixed download revisions; see [checkpoint references](../checkpoints.md) for
the separate historical evaluation records.

Keep the tokenizer, chat template, image/video processor configurations and
all indexed weight shards. Do not copy just one shard or substitute another
model's processor.

## 3. Configure and run

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
```

Set `python: <REPO>/.model-envs/roboreward/bin/python` and
`checkpoint: <REPO>/checkpoints/RoboReward-4B` or
`checkpoint: <REPO>/checkpoints/RoboReward-8B` to match the download. Keep the
template's starting `batch_size: 1`, then fill `data`, `output`,
`gpu`, `tasks` and `metrics` for your input. Top-level relative paths resolve
from the config directory, as explained in [data preparation](../data.md).

The [runner policy](../../src/vmbmk/runner/policy.py) permits the VOC family
(`voc`, `voc_mem`, and `cycle_voc`). VS and CSVC (`csvc`, alias `cspc`) are
excluded for RoboReward's discrete outputs, matching the paper. Their benchmark
coverage is N/A, not a measured zero. Appendix F.3 separately assigns zero to
missing metrics when computing the paper's normalized leaderboard score.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/roboreward-local.yaml
```

## Verification and common failures

The environment plan was checked against available prerequisites. This is not
a completed fresh-install or checkpoint-inference test. A TorchCodec import
failure concerns native library compatibility; a missing tokenizer or shard
concerns checkpoint completeness. Keep model output semantics unchanged instead
of introducing common min-max normalization to work around unexpected outputs.
