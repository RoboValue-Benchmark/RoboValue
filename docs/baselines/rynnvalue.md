# Install RynnValue

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/rynnvalueconfigs.yaml)

## 1. Prepare the runtime

Follow the [shared prerequisites](README.md#prerequisites). This adapter needs no separate
baseline source checkout; model-specific code is shipped in the HF checkpoint.

```bash
bash envs/setup.sh rynnvalue --plan
bash envs/setup.sh rynnvalue
```

The runtime is `.model-envs/rynnvalue`.

## 2. Download a checkpoint

The smaller starting choice is
[Alibaba-DAMO-Academy/RynnValue-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-4B):

```bash
hf download Alibaba-DAMO-Academy/RynnValue-4B \
  --revision 3f73b5d2b5e53b21f248c8791004dde6a8cf2b92 \
  --local-dir checkpoints/RynnValue-4B
```

For [RynnValue-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B),
use revision `8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e` and a separate
`checkpoints/RynnValue-8B` directory. These revisions were checked publicly on
2026-10-06; neither is certified as the archived benchmark revision.

Preserve custom configuration/modeling/processing files, tokenizer files and
all indexed weights. This adapter executes checkpoint-provided custom code;
review its provenance/license before use. The absence of a Git source checkout
does not mean there is no third-party model code.

## 3. Configure and run

```bash
cp configs/rynnvalueconfigs.yaml configs/rynnvalue-local.yaml
```

Set `python: <REPO>/.model-envs/rynnvalue/bin/python` and
`checkpoint: <REPO>/checkpoints/RynnValue-4B`, then fill `data`, `output`,
`gpu`, `tasks` and eligible `metrics`. Change only the checkpoint path for a
different size, while checking its actual memory requirements.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/rynnvalue-local.yaml
```

## Verification and common failures

CPU interface tests do not establish clean installation, 4B/8B inference or
paper reproduction. Missing `modeling_rynn_value_lang.py`, processing code or
tokenizer assets is an incomplete-checkpoint issue; do not resolve it by using
another baseline's processor or deleting custom files from the snapshot.
