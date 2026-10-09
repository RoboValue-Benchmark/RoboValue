# Install RoboFAC

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/robofacconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites). No external RoboFAC source
checkout is required by the adapter; it loads through the locked Transformers
and Qwen utility dependencies. See [shared prerequisites](README.md#prerequisites).

```bash
bash envs/setup.sh robofac --plan
bash envs/setup.sh robofac
```

The runtime is `.model-envs/robofac`.

## 2. Download inference assets

Use [MINT-SJTU/RoboFAC-7B](https://huggingface.co/MINT-SJTU/RoboFAC-7B). The
repository also contains training/optimizer states that are unnecessary for
this adapter; exclude those directories rather than downloading the whole
training checkpoint tree:

```bash
hf download MINT-SJTU/RoboFAC-7B \
  --revision 7d55520ed6b44c500e7a844baae5a36e40186ecf \
  --local-dir checkpoints/RoboFAC-7B \
  --exclude 'global_step*/*' 'optimizer.pt' 'scheduler.pt'
```

Keep all indexed inference shards, configuration, tokenizer and processor files.
The revision was checked against public metadata on 2026-10-06; it is not
certified as the archived benchmark revision. Inspect sizes and model licenses
before downloading.

## 3. Configure and run

```bash
cp configs/robofacconfigs.yaml configs/robofac-local.yaml
```

Set `python: <REPO>/.model-envs/robofac/bin/python` and
`checkpoint: <REPO>/checkpoints/RoboFAC-7B`, then select `data`, `output`,
`gpu`, `tasks` and eligible `metrics`.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/robofac-local.yaml
```

## Verification and common failures

The adapter has a CPU interface test, not a certified clean-install or
checkpoint-inference test. Downloading only training `global_step` states does
not produce the Transformers checkpoint this adapter expects. Missing shard
indices/tokenizers should be resolved in model assets, not by changing the
evaluation's query or scoring rules.
