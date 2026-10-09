# Install RoboMeter

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/robometerconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites) for compatible tooling
and the supplied installer's reference settings. The official source imports Unsloth and therefore requires
a visible accelerator even for its final import check. Select one authorized GPU;
`0` below is an example, not an instruction to use an occupied device.

```bash
bash envs/source.sh robometer
bash envs/setup.sh robometer --plan
bash envs/setup.sh robometer --probe-gpu 0
```

Sources are downloaded from [robometer/robometer](https://github.com/robometer/robometer)
at the commit in `envs/models.yaml`, to `.baseline-sources/robometer`.
The runtime is `.model-envs/robometer`. Installation/builds hide GPUs; only the
final probe sees the selected device. No checkpoint is loaded during setup.

## 2. Download weights and the base model

The [Robometer-4B model card](https://huggingface.co/robometer/Robometer-4B)
identifies Qwen3-VL-4B as its base. Download the reward checkpoint into a local
directory, preserving its `config.yaml`, `config.json`, shard index and weights:

```bash
hf download robometer/Robometer-4B \
  --revision beef63bc914c5c189329d49c6d712d96d632aa34 \
  --local-dir checkpoints/Robometer-4B
```

**The reward checkpoint alone is not enough for offline loading.** Its
`config.yaml` uses `model.base_model_id: Qwen/Qwen3-VL-4B-Instruct`; the official
loader reads the base model and processor by that ID before loading reward
weights. Pre-populate the Hub cache using the same `HF_HOME` for download and run:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
```

This second command deliberately omits `--local-dir`: it prepares the Hub cache
for the ID used by upstream code. Keep that cache available during inference.
It is a **model asset**, not uv's disposable dependency cache. The base revision
was checked publicly on 2026-10-06; it is not certified as the archived benchmark
base revision. Do not silently modify the checkpoint's model settings.

## 3. Configure and run

```bash
cp configs/robometerconfigs.yaml configs/robometer-local.yaml
```

Set `python` to `<REPO>/.model-envs/robometer/bin/python`, `checkpoint` to
`<REPO>/checkpoints/Robometer-4B`, and `data`/`output` to your supplied dataset
root and result root. Replace `gpu`, `tasks` and `metrics` with an authorized,
eligible selection. Keep the `HF_HOME` above exported for offline execution.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/robometer-local.yaml
```

## Verification and common failures

Source download and setup plan were checked. The old server runtime lacked
Unsloth and did not match the lock; it is not a verified clean installation.
Build a fresh runtime using the lock rather than adding packages to that old
environment. Missing Unsloth, hidden GPUs and an absent Qwen Hub snapshot are
different errors; none is fixed by downloading the reward checkpoint again.
Full new-environment installation and checkpoint inference remain unverified.
