# LIV checkpoint assets

`model.pt` and `RN50.pt` use Git LFS; `config.yaml` is a regular Git file.
These assets are kept separate from source, environments and caches.

```bash
git lfs install --local
git lfs pull --include="checkpoints/LIV/model.pt,checkpoints/LIV/RN50.pt"
```

See the [installation guide](../../docs/baselines/liv.md) for environment setup
and direct downloads, and [checksums](../../docs/checkpoints.md#liv-asset-checksums).
These assets alone do not establish score reproduction.

## Provenance and upstream notices

- `model.pt` and `config.yaml`: [jasonyma/LIV](https://huggingface.co/jasonyma/LIV),
  recorded revision `5a29adb8f796b0f91fdd9bc284b31a1829c1eb91`.
- `RN50.pt`: [OpenAI CLIP](https://github.com/openai/CLIP), downloaded from its
  official Azure asset URL recorded in the installation guide.
- LIV source: [penn-pal-lab/LIV](https://github.com/penn-pal-lab/LIV),
  commit `a12991f53aab01f3cecc1315a81068ba12e2bd6b`.

Upstream source license notices are preserved in `LICENSE-LIV` and `LICENSE-CLIP`.
The inspected LIV Hub metadata does not specify a separate checkpoint license.
These notices do not invent a checkpoint license or grant additional rights;
consult the upstream authors if your use requires clarification. No
repository-wide license is selected here.
