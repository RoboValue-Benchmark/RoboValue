# Checkpoints

For executable download/setup instructions, use the
[per-baseline guides](baselines/README.md). This page preserves recorded
checkpoint identities; it is not a substitute for the installation guides.
Public revisions newly checked on 2026-10-06 are labeled in those guides rather
than silently replacing the historical identities below.

Use the local path in the YAML `checkpoint` field. Revisions are recorded when
available locally; `unrecorded` means the local snapshot is usable but its
exact upstream revision has not been pinned in this repository.

| Adapter | Pretrained checkpoint | Hugging Face revision | Local checkpoint path |
| --- | --- | --- | --- |
| Robometer | [`robometer/Robometer-4B`](https://huggingface.co/robometer/Robometer-4B) | `beef63bc914c5c189329d49c6d712d96d632aa34` | `/path/to/checkpoints/robometer` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-3B`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B) | `b75e1ea8381d384f580b41d04ee80fdff5369b34` | `/path/to/checkpoints/robo-dopamine-grm-3b` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-8B`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B) | `773d72a9d20dac523968e43a6b0752ace434471b` | `/path/to/checkpoints/robo-dopamine-grm-8b` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) | `68c8268cfb5f72f828e652bd40cb8b3324e902a5` | `/path/to/checkpoints/robo-dopamine-grm-2.0-4b-preview` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) | `980f3d1819c870f62c36169a9486e971049bb09a` | `/path/to/checkpoints/robo-dopamine-grm-2.0-8b-preview` |
| ProcVLM | [`ce-amtic/ProcVLM-2B`](https://huggingface.co/ce-amtic/ProcVLM-2B) | `a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5` | `/path/to/checkpoints/ProcVLM-2B` |
| RoboReward | [`teetone/RoboReward-8B`](https://huggingface.co/teetone/RoboReward-8B) | `3a185b4fce2b1253643105be1f234ae618b9732f` | `/path/to/checkpoints/roboreward` |
| TOPReward | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `unrecorded` | `/path/to/checkpoints/Qwen3-VL-8B-Instruct` |
| RoboFAC | [`MINT-SJTU/RoboFAC-7B`](https://huggingface.co/MINT-SJTU/RoboFAC-7B) | `unrecorded` | `/path/to/checkpoints/robofac` |
| RynnValue 4B | [`Alibaba-DAMO-Academy/RynnValue-4B`](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-4B) | `unrecorded` | `/path/to/checkpoints/RynnValue-4B` |
| RynnValue 8B | [`Alibaba-DAMO-Academy/RynnValue-8B`](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B) | `unrecorded` | `/path/to/checkpoints/RynnValue-8B` |
| VLAC 2B | [`InternRobotics/VLAC`](https://huggingface.co/InternRobotics/VLAC) | `f44e552324944f80a63ab0083879f0bfafe01516` | `/path/to/checkpoints/vlac-official` |
| VLAC 8B | [`InternRobotics/VLAC-8b`](https://huggingface.co/InternRobotics/VLAC-8b) | `dd94c057b0274b88e7228e8eaa73c03a7e8fa1d5` | `/path/to/checkpoints/VLAC-8B` |
| LIV-EPIC RN50 | [`jasonyma/LIV`](https://huggingface.co/jasonyma/LIV) | `5a29adb8f796b0f91fdd9bc284b31a1829c1eb91` | `/path/to/checkpoints/LIV` |

TOPReward uses model=topreward with model_options.backend=qwen or molmo.
The Qwen checkpoint above is not interchangeable with a Molmo2 checkpoint;
provide the corresponding externally obtained weights. No Molmo checkpoint
revision is certified by this document. See configuration.md for backend selection.

## LIV asset checksums

The LIV directory contains the following distributed runtime assets. The two
weights use Git LFS; config.yaml is a regular tracked file. Other checkpoints
remain ignored. HF provides model.pt and config.yaml; RN50.pt is a separate
official CLIP asset. Follow the [LIV guide](baselines/liv.md) for LFS checkout or
upstream downloads, and see [asset provenance](../checkpoints/LIV/README.md).
The recommended persistent location is checkpoints/LIV, separate from downloaded
source and uv caches. Preserve all three files together; reinstalling dependencies
must not delete or relocate the checkpoint directory.

| File | SHA256 |
| --- | --- |
| `model.pt` | `0387b1ce397ffc67c03e0945ce3ecce1dc037191ea7e680a721618e6fed1bc1c` |
| `RN50.pt` | `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` |
| `config.yaml` | `0a859e48f94d9a3e1300a64dfa65556f5bbe48a90e4a2f03bbfabbbc0409f5de` |

The matching official LIV source is stored at
`.baseline-sources/liv` by source.sh and pinned to commit
`a12991f53aab01f3cecc1315a81068ba12e2bd6b`. The adapter reads
`config.yaml`, passes the local `RN50.pt` path into the official model, and
loads `model.pt` explicitly; it never uses `~/.liv` or an implicit download.
