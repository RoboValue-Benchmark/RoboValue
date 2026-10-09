# Install RoboDopamine

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/robodopamineconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites) and preserve a
compatible Torch/vLLM environment. No separate baseline source
checkout is required by this adapter; its runtime uses the locked vLLM stack.

```bash
bash envs/setup.sh robodopamine --plan
bash envs/setup.sh robodopamine
```

The runtime is `.model-envs/robodopamine`. Do not combine its Torch/vLLM
environment with another baseline's environment.

## 2. Download a checkpoint

The primary example is [Robo-Dopamine-GRM-3B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B):

```bash
hf download tanhuajie2001/Robo-Dopamine-GRM-3B \
  --revision b75e1ea8381d384f580b41d04ee80fdff5369b34 \
  --local-dir checkpoints/RoboDopamine-GRM-3B
```

The [8B checkpoint](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B)
has recorded revision `773d72a9d20dac523968e43a6b0752ace434471b`; use a separate
`checkpoints/RoboDopamine-GRM-8B` directory if selecting it. Keep model, tokenizer,
processor and configuration files together. Other model releases are not implied
to have the same tested behavior; [checkpoint references](../checkpoints.md)
record additional historical repository identities.

## 3. Prepare references, configure and run

This adapter uses **one-shot references** and normally expects front,
wrist-left and wrist-right views. Prepare the disjoint reference input supplied
for this baseline; a random evaluation episode is not a valid substitute.
See [data/reference guidance](../data.md#one-shot-references).

```bash
cp configs/robodopamineconfigs.yaml configs/robodopamine-local.yaml
```

Set `python: <REPO>/.model-envs/robodopamine/bin/python` and
`checkpoint: <REPO>/checkpoints/RoboDopamine-GRM-3B`. Set `data`, `output`,
`reference_data`, `gpu`, `tasks` and `metrics` explicitly. Do not remove
`reference_data` just to bypass an unavailable reference dataset.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/robodopamine-local.yaml
```

## Verification and common failures

A bounded real-video 3B inference-to-score path was exercised in an existing
environment, including VOC, Cycle-VOC and VS; see [testing evidence](../testing.md).
This does not certify clean installation, every checkpoint, OOD coverage or
paper-table reproduction. Missing camera views/references are input problems;
vLLM memory errors require a genuinely available GPU and appropriate batch size,
not a silent change to metric sampling.
