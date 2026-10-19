# Checkpoints

## Model list

The table lists 17 checkpoint choices by adapter. Select a model, then follow
its setup guide for the runtime, download commands and YAML configuration.

| Adapter | Model / weights | Setup guide |
| --- | --- | --- |
| Robometer | [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) | [Setup](baselines/robometer.md) |
| Robo-Dopamine | [Robo-Dopamine-GRM-3B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B) | [Setup](baselines/robodopamine.md) |
| Robo-Dopamine | [Robo-Dopamine-GRM-8B](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B) | [Setup](baselines/robodopamine.md) |
| Robo-Dopamine | [Robo-Dopamine-GRM-2.0-4B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) | [Setup](baselines/robodopamine.md) |
| Robo-Dopamine | [Robo-Dopamine-GRM-2.0-8B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) | [Setup](baselines/robodopamine.md) |
| ProcVLM | [ProcVLM-2B](https://huggingface.co/ce-amtic/ProcVLM-2B) | [Setup](baselines/procvlm.md) |
| RoboReward | [RoboReward-4B](https://huggingface.co/teetone/RoboReward-4B) | [Setup](baselines/roboreward.md) |
| RoboReward | [RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B) | [Setup](baselines/roboreward.md) |
| TOPReward | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Setup](baselines/topreward.md) |
| TOPReward | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | [Setup](baselines/topreward.md) |
| TOPReward | [Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) | [Setup](baselines/topreward.md) |
| RoboFAC | [RoboFAC-7B](https://huggingface.co/MINT-SJTU/RoboFAC-7B) | [Setup](baselines/robofac.md) |
| RynnValue | [RynnValue-4B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-4B) | [Setup](baselines/rynnvalue.md) |
| RynnValue | [RynnValue-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B) | [Setup](baselines/rynnvalue.md) |
| VLAC | [VLAC-2B](https://huggingface.co/InternRobotics/VLAC) | [Setup](baselines/vlac.md) |
| VLAC | [VLAC-8B](https://huggingface.co/InternRobotics/VLAC-8b) | [Setup](baselines/vlac.md) |
| LIV | [LIV-EPIC RN50](https://huggingface.co/jasonyma/LIV) | [Setup](baselines/liv.md) |

TOPReward uses `model: topreward` with `model_options.backend: qwen` for either
Qwen3-VL size, or `model_options.backend: molmo` for Molmo2. Choose the matching
runtime and checkpoint in the [TOPReward guide](baselines/topreward.md).
See [configuration](configuration.md) for backend selection.

<details>
<summary><strong>Download references: revisions and local paths</strong></summary>

A revision identifies a snapshot of the model repository, including weights,
tokenizer and processor files. Use the references below to repeat a download
of the same files. These download references do not establish the versions
used in historical evaluations.

| Adapter | Pretrained checkpoint | Fixed download revision | Local checkpoint path |
| --- | --- | --- | --- |
| Robometer | [`robometer/Robometer-4B`](https://huggingface.co/robometer/Robometer-4B) | `beef63bc914c5c189329d49c6d712d96d632aa34` | `/path/to/checkpoints/Robometer-4B` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-3B`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B) | `b75e1ea8381d384f580b41d04ee80fdff5369b34` | `/path/to/checkpoints/RoboDopamine-GRM-3B` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-8B`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-8B) | `773d72a9d20dac523968e43a6b0752ace434471b` | `/path/to/checkpoints/RoboDopamine-GRM-8B` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) | `68c8268cfb5f72f828e652bd40cb8b3324e902a5` | `/path/to/checkpoints/RoboDopamine-GRM-2.0-4B-Preview` |
| Robo-Dopamine | [`tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview`](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview) | `980f3d1819c870f62c36169a9486e971049bb09a` | `/path/to/checkpoints/RoboDopamine-GRM-2.0-8B-Preview` |
| ProcVLM | [`ce-amtic/ProcVLM-2B`](https://huggingface.co/ce-amtic/ProcVLM-2B) | `a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5` | `/path/to/checkpoints/ProcVLM-2B` |
| RoboReward | [`teetone/RoboReward-4B`](https://huggingface.co/teetone/RoboReward-4B) | `4dec8af8d89d214de7028d70f9f9aee3b3a44c3d` | `/path/to/checkpoints/RoboReward-4B` |
| RoboReward | [`teetone/RoboReward-8B`](https://huggingface.co/teetone/RoboReward-8B) | `3a185b4fce2b1253643105be1f234ae618b9732f` | `/path/to/checkpoints/RoboReward-8B` |
| TOPReward | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `/path/to/checkpoints/Qwen3-VL-8B-Instruct` |
| TOPReward | [`Qwen/Qwen3-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | `/path/to/checkpoints/Qwen3-VL-32B-Instruct` |
| TOPReward | [`allenai/Molmo2-8B`](https://huggingface.co/allenai/Molmo2-8B) | `e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b` | `/path/to/checkpoints/Molmo2-8B` |
| RoboFAC | [`MINT-SJTU/RoboFAC-7B`](https://huggingface.co/MINT-SJTU/RoboFAC-7B) | `7d55520ed6b44c500e7a844baae5a36e40186ecf` | `/path/to/checkpoints/RoboFAC-7B` |
| RynnValue 4B | [`Alibaba-DAMO-Academy/RynnValue-4B`](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-4B) | `3f73b5d2b5e53b21f248c8791004dde6a8cf2b92` | `/path/to/checkpoints/RynnValue-4B` |
| RynnValue 8B | [`Alibaba-DAMO-Academy/RynnValue-8B`](https://huggingface.co/Alibaba-DAMO-Academy/RynnValue-8B) | `8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e` | `/path/to/checkpoints/RynnValue-8B` |
| VLAC 2B | [`InternRobotics/VLAC`](https://huggingface.co/InternRobotics/VLAC) | `f44e552324944f80a63ab0083879f0bfafe01516` | `/path/to/checkpoints/VLAC-2B` |
| VLAC 8B | [`InternRobotics/VLAC-8b`](https://huggingface.co/InternRobotics/VLAC-8b) | `dd94c057b0274b88e7228e8eaa73c03a7e8fa1d5` | `/path/to/checkpoints/VLAC-8B` |
| LIV-EPIC RN50 | [`jasonyma/LIV`](https://huggingface.co/jasonyma/LIV) | `5a29adb8f796b0f91fdd9bc284b31a1829c1eb91` | `/path/to/checkpoints/LIV` |

</details>

<details>
<summary><strong>Historical evaluation checkpoint records</strong></summary>

The original checkpoint identity table is retained below. `unrecorded` means
that the exact Hugging Face revision was not captured in the available historical
records. Current download pins, website source hashes and GitHub code commits
are not substitutes for those missing model revisions.

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

</details>

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
