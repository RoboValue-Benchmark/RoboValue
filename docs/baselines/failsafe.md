# Prepare FailSafe for SIA

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/failsafeconfigs.yaml)

This package retains FailSafe **only for SIA**. Simulation, training and other
metric interfaces are not installation targets. Environment availability does
not mean a complete, publicly downloadable FailSafe checkpoint is supplied.

**Supported metric: `sia` only.** Use
[`configs/failsafeconfigs.yaml`](../../configs/failsafeconfigs.yaml); do not copy
the all-metric general template for this adapter. Other metrics are unsupported,
not zero-scoring baselines.

## Provenance: not an official FailSafe installation

The author/project repository is [Jimntu/FailSafe_code](https://github.com/Jimntu/FailSafe_code),
which describes the FailSafe failure-generation/recovery pipeline in ManiSkill.
The `llava` source used by this benchmark instead comes from the original local
[LLaVA-NeXT fork](https://github.com/XCXJasonHsia/LLaVA-NeXT), at commit
`72999db7b210d23856a21b80a67bf52fe14b92e9`. We have not verified
that the FailSafe authors endorse this fork or the benchmark's SIA adaptation.

**The commands below prepare the current benchmark adapter's dependencies,
not the official FailSafe pipeline.** The `failsafe` key is retained for existing
configs, but official implementation/checkpoint equivalence remains unresolved.

## 1. Prepare the inference runtime

Follow the [shared prerequisites](README.md#prerequisites). The supplied DeepSpeed dependency
requires `nvcc`, `c++` and a toolkit compatible with Torch; see
[environment requirements](../environments.md).

**Reuse the original working source, runtime and checkpoint.** Do not replace
them with the publicly accessible fork revision previously suggested by this
guide. Set YAML `python` and `checkpoint` to those existing local paths; ensure
the runtime still imports `llava` from the original source.

The original commit is restored in `envs/models.yaml`. It could not be fetched
from the public fork during inspection, so `source.sh failsafe` is not currently
a verified public download route. It must not silently substitute another commit.
Keep the original checkout or obtain an approved copy; this is a delivery limitation.

If a new runtime is needed, use the existing original checkout explicitly:

```bash
bash envs/setup.sh failsafe --source /path/to/original/FailSafe_inference --plan
bash envs/setup.sh failsafe --source /path/to/original/FailSafe_inference
```

Confirm its commit with `git -C /path/to/original/FailSafe_inference rev-parse HEAD`.
Explicit `--source` checks required files but does not enforce the commit itself.
Do not upgrade the existing working runtime just to match the supplied setup
lock: a newly installed runtime still requires separate inference validation.

## 2. Obtain the actual evaluation checkpoint

**The identity and author-approved distribution of the checkpoint required for
an official FailSafe result have not been established.** Obtain confirmation
from the authors about the intended model, weights and inference procedure,
then place the approved assets at `checkpoints/FailSafe/`.
Do not fabricate a Hub download command or substitute unrelated weights.
Neither a local folder named `failsafe` nor a configuration's base-model name
proves that the weights are FailSafe-specific; we have not established that
separately trained FailSafe weights are necessarily required either.

The adapter expects a complete local LLaVA checkpoint: `config.json`,
`model.safetensors.index.json`, all referenced shards, tokenizer/processor
assets and a compatible `LlavaQwenForCausalLM` configuration. A standalone LoRA
directory or training-state directory does not meet that contract.

The inspected old checkpoint configuration names
[lmms-lab/llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)
as its base and [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
as its vision tower. These are **base assets, not certified FailSafe weights**.
For that configuration, pre-populate the referenced vision-tower Hub cache:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf download google/siglip-so400m-patch14-384 \
  --revision 9fdffc58afc957d1a03a25b10dba0329ab15c2a3
```

That revision was checked publicly on 2026-10-06, not matched to archived scores.
Inspect your supplied checkpoint's `mm_vision_tower` first; a different tower
requires its own approved assets. Keep the same `HF_HOME` exported during offline
inference. Downloading the base model/tower does not resolve missing FailSafe weights.

## 3. Configure SIA and run

```bash
cp configs/failsafeconfigs.yaml configs/failsafe-local.yaml
```

Set `python` to the original working runtime's executable and `checkpoint` to
the original evaluated checkpoint directory. Fill `data`, `output`, `gpu`
and actual task IDs. Keep only `metrics.sia` in this config.
SIA judge settings belong in the ignored local
`src/vmbmk/metrics/sia/config.yaml` or supported environment overrides, not a
committed baseline YAML. See [SIA settings](../configuration.md) for exact fields
and [SIA execution](../result_publication.md#sia-execution) for intermediate outputs.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/failsafe-local.yaml
```

## Verification and common failures

On 2026-10-06, the original checkout was clean at the restored commit and
`llava.model.builder` / `llava.mm_utils` imported from that checkout using the
original runtime. Existing SIA generation artifacts and sim/real score artifacts
were also located. These establish prior execution evidence, not a fresh full
inference test or a verified clean-machine installation. The replacement public
revision's earlier import check is not evidence for the restored version.
Until the intended source/model/procedure is confirmed, this integration is
**not verified for reporting official FailSafe results**. Generic LLaVA weights
must not be labeled FailSafe solely because the adapter or directory has that name.
