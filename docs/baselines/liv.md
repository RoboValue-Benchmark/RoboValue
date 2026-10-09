# Install LIV

[All baselines](README.md) | [Data](../data.md) | [Template](../../configs/livconfigs.yaml)

## 1. Prepare a fresh runtime

Follow the [shared prerequisites](README.md#prerequisites). The supplied reference lock preserves LIV's older
Torch `1.13.1+cu117` and torchvision `0.14.1+cu117` pair, with NumPy `1.24.4`
and ftfy `6.3.1`. Do not upgrade that stack to match another baseline.

```bash
bash envs/source.sh liv
bash envs/setup.sh liv --plan
bash envs/setup.sh liv
```

The [official LIV source](https://github.com/penn-pal-lab/LIV) is downloaded to
`.baseline-sources/liv` at `a12991f53aab01f3cecc1315a81068ba12e2bd6b`.
Setup attaches both that root and `liv/models/clip` to `.model-envs/liv`.
ftfy is installed from the dependency lock; no vendored `.runtime_liv_deps`
folder or independent CLIP clone is needed.

**LIV does not require uv's installation cache at runtime.** Setup uses copy
mode, not package symlinks into temporary caches. An old environment with broken
Pillow/Torch/NumPy links should be retained for inspection and replaced by a fresh
runtime, not repaired by installing Pillow alone. Use a new `--root` if necessary.

## 2. Retrieve all three required assets

The release includes `config.yaml` and distributes both weights through Git LFS.
From a Git checkout, with Git LFS available, run:

```bash
git lfs install --local
git lfs pull --include="checkpoints/LIV/model.pt,checkpoints/LIV/RN50.pt"
```

Each weight is about 244 MiB. A small text pointer is not a usable checkpoint;
do not assume a source ZIP or a clone with LFS downloads disabled includes the
weights. See [asset provenance](../../checkpoints/LIV/README.md) and
[checksums](../checkpoints.md#liv-asset-checksums). Source and environment setup
remain separate steps.

### Alternative: download from upstream

If Git LFS is unavailable, obtain the same assets directly instead:

[jasonyma/LIV](https://huggingface.co/jasonyma/LIV) supplies **`model.pt` and
`config.yaml`, but not `RN50.pt`**:

```bash
hf download jasonyma/LIV model.pt config.yaml \
  --revision 5a29adb8f796b0f91fdd9bc284b31a1829c1eb91 \
  --local-dir checkpoints/LIV
```

Download CLIP RN50 from the official URL recorded by LIV's bundled CLIP code:

```bash
curl --fail --location \
  'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt' \
  --output checkpoints/LIV/RN50.pt
echo 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762  checkpoints/LIV/RN50.pt' | sha256sum --check
```

Choose a fresh destination if assets already exist; the shell download command
does not protect against overwriting a local file. The completed directory is:

```text
checkpoints/LIV/
  model.pt
  config.yaml
  RN50.pt
```

The adapter loads these local files explicitly, not an implicit `~/.liv`
download. Source includes CLIP's BPE vocabulary; it is not the RN50 checkpoint.
For recorded checksums of all three assets, see [checkpoint references](../checkpoints.md#liv-asset-checksums).

## 3. Configure and run

```bash
cp configs/livconfigs.yaml configs/liv-local.yaml
```

Set `python: <REPO>/.model-envs/liv/bin/python`,
`checkpoint: <REPO>/checkpoints/LIV`, plus `data`, `output`, `gpu`, `tasks`
and eligible `metrics`. **Also set** `model_options.source_root` to the absolute
path printed by source.sh: `<REPO>/.baseline-sources/liv`. This adapter reads
its official model/CLIP files explicitly; the checkpoint path is not the source root.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/liv-local.yaml
```

## Verification and common failures

Official source download and automatic source-path setup plans passed. The old
server environment had broken links across multiple dependencies; its failed
Pillow import is not evidence that this fresh environment has been installed.
Full locked installation, tokenizer/CPU checks and real checkpoint inference
remain pending. A missing `RN50.pt` is a model-asset issue, while missing ftfy or
broken PIL/Torch files is an environment issue; downloading LIV source alone
does not fix either.
