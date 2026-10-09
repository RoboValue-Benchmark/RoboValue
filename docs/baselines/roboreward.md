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

Use [teetone/RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B):

```bash
hf download teetone/RoboReward-8B \
  --revision 3a185b4fce2b1253643105be1f234ae618b9732f \
  --local-dir checkpoints/RoboReward-8B
```

Keep the tokenizer, chat template, image/video processor configurations and
all indexed weight shards. Do not copy just one shard or substitute another
model's processor.

## 3. Configure and run

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
```

Set `python: <REPO>/.model-envs/roboreward/bin/python` and
`checkpoint: <REPO>/checkpoints/RoboReward-8B`, then fill `data`, `output`,
`gpu`, `tasks` and `metrics` for your input. Top-level relative paths resolve
from the config directory, as explained in [data preparation](../data.md).

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
