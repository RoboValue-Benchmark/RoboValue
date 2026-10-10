# Install ProcVLM

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/procvlmconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites). The supplied environment includes
a DeepSpeed build: `nvcc` and `c++` must be available, and the toolkit must be
compatible with locked Torch. See [native build requirements](../environments.md).
Do not install or change a shared system CUDA toolkit through this workflow.

```bash
bash envs/source.sh procvlm
bash envs/setup.sh procvlm --plan
bash envs/setup.sh procvlm
```

The [official source](https://github.com/RUCKBReasoning/ProcVLM) is pinned in
`envs/models.yaml` and downloaded to `.baseline-sources/procvlm`; setup attaches
its `evqa` and `core` packages. The runtime is `.model-envs/procvlm`.

## 2. Download the base checkpoint

Download [ce-amtic/ProcVLM-2B](https://huggingface.co/ce-amtic/ProcVLM-2B):

```bash
hf download ce-amtic/ProcVLM-2B \
  --revision a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5 \
  --local-dir checkpoints/ProcVLM-2B
```

This revision contains processor/tokenizer files and `procvlm_extra/procvlm_head.pt`
as well as model weights. Preserve the directory structure; do not point YAML
to `.cache/huggingface` metadata or copy only model shards. A full snapshot can
contain multiple weight formats, so inspect the download size before starting.

## 3. Select base or task LoRA, then run

```bash
cp configs/procvlmconfigs.yaml configs/procvlm-local.yaml
```

Set `python: <REPO>/.model-envs/procvlm/bin/python`,
`checkpoint: <REPO>/checkpoints/ProcVLM-2B`, plus `data`, `output`, `gpu`,
`tasks` and `metrics`. The base example does not require task LoRA weights.

For the one-shot/task-adapted variant, obtain the task LoRA assets separately
and set `model_options.checkpoint_by_task` to absolute paths. No public download
mapping for those experiment-specific assets is certified here. Missing task
weights must not be silently replaced with the base checkpoint.

For one-shot evaluation, VOC and Memory-VOC exclude `press_by_number` and
`swap_blocks` because their task-specific LoRA weights are unavailable. The
[runner policy](../../src/vmbmk/runner/policy.py) enforces this scope.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/procvlm-local.yaml
```

## Verification and common failures

Source download, setup plan and new-source imports passed in an existing runtime.
The older server checkout had local model/backend patches that are not copied
into the clean public source. This guide does not establish equivalence to those
archived runs, especially task LoRA loading. Fresh installation and checkpoint
inference must still be tested. Retained parsing retry/forward-fill behavior is
documented in [testing](../testing.md), not presented as a new scoring contract.
