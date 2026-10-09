# Model environments

For a first installation, start with the [baseline guides](baselines/README.md).
This page documents the shared deployment contract and validation boundaries;
checkpoint download steps live in each baseline guide.

Keep baseline dependencies isolated. The benchmark core supports Python 3.10
and later; supplied locks target Linux x86_64 and pin reference versions below.
These are installer/reproduction settings, not universal model compatibility bounds. Do not
combine or upgrade upstream Torch/CUDA stacks. The declarations and locks are
retained without dependency upgrades.

## Supported mapping

The authoritative mapping is envs/models.yaml. Runtime directory names are
adapter names, not necessarily lock-project names.

| Adapter / runtime directory | Lock project under envs/ | Python series | Upstream source required |
| --- | --- | --- | --- |
| robometer | robometer | 3.10.x | Robometer import root |
| robodopamine | robo-dopamine | 3.10.x | No |
| procvlm | procvlm | 3.10.x | ProcVLM import root containing evqa and core |
| roboreward | roboreward | 3.10.x | No |
| vlac | vlac | 3.10.x | VLAC import root containing evo_vlac |
| rynnvalue | rynnvalue | 3.10.x | No |
| topreward | topreward | 3.11.x | TOPReward import root |
| topreward_molmo | topreward | 3.11.x | No |
| robofac | robofac | 3.10.x | No |
| liv | liv | 3.10.x | Official LIV checkout |
| failsafe | failsafe_inference | 3.10.x | LLaVA fork; official FailSafe equivalence unverified |

TOPReward Qwen and Molmo share a lock project but have separate runtimes.
Both use model=topreward; model_options.backend selects qwen or molmo.
The setup target topreward_molmo is a runtime name, not a second adapter.
FailSafe uses only the SIA inference environment, not its simulation stack.
GVL and ReWiND are unsupported; their retained environment declarations and
result identities are historical only. ProcVLM's former minor-version Python
selector uses the 3.10 series; its dependency lock is unchanged.

## Prepare and deploy

### Shell entry points

Source acquisition, dependency installation and evaluation are separate steps.
Checkpoints and datasets are user-provided; none of these commands downloads them.
The source helper selects the repository and full commit ID from envs/models.yaml
by baseline name; it never fetches a moving branch. Default checkout locations
are repository/.baseline-sources/<baseline>, independent of the current working
directory. These are ignored third-party sources, not files copied into the
supported src/vmbmk package. Setup attaches the selected import roots through
the runtime's site-packages/vmbmk_source.pth. Start with LIV, for example:

    bash envs/source.sh liv --plan
    bash envs/source.sh liv
    bash envs/setup.sh liv --plan

The same source command supports robometer, procvlm, vlac, topreward (Qwen),
liv and failsafe. Other runtime names explicitly report that no external source
is needed. TOPReward Molmo uses source-free setup.sh topreward_molmo.
For a custom source parent, use the same root in both commands:

    bash envs/source.sh liv --root /path/to/sources
    bash envs/setup.sh liv --source-root /path/to/sources --root /path/to/model-envs --plan

Review the plan, then remove --plan to install locked dependencies. If a compatible
Python is not discoverable, add --python /path/to/python; select uv with
--uv /path/to/uv. VMBMK_PYTHON selects the launcher Python with PyYAML, not the
model-runtime Python. An existing custom checkout can still be selected with
--source /path/to/import-root. For a baseline without external source:

    bash envs/setup.sh robodopamine --root /path/to/model-envs --plan

Installation uses copy mode, avoiding package-file links to disposable uv caches.
The source helper fetches a shallow, blob-filtered commit and sparsely checks out
the mapped code directories. It skips LFS downloads, sample videos and checkpoint
archives. Public repositories must support Git partial fetches to avoid fetching
unselected blobs. It checks the actual commit, clean working tree and required
source files; an unchanged matching checkout is reusable. Modified, unrelated or
incomplete destinations are never overwritten. A failed download is retained for
inspection: choose another --root or explicitly resolve that checkout before
retrying. No mapped checkout requires a submodule; submodules are not fetched.
Download again on the target OS rather than transferring Windows Git checkouts
to Linux: file-mode and line-ending changes correctly fail the clean-tree check.
Source and runtime directories must remain at their configured
locations: source paths are attached to the runtime, not copied into the package.
No system packages, drivers or CUDA installations are changed by the helpers.

After setup succeeds, set the YAML python field to the printed runtime, checkpoint
to your weights and data to your dataset. For LIV also set model_options.source_root.
Then launch bash vmbmk.sh run CONFIG. Setup's checked status is an import/dependency
probe, not proof of successful model inference. Downloads and native builds can
consume significant storage/time and require explicit authorization on shared hosts.

Run the standalone envs/setup.py with an existing Python 3.10+ interpreter
that has PyYAML. Use a compatible Python series and uv with frozen pylock export
and pip sync support; neither a Python patch version nor an exact uv release is
required. See [compatible environments](baselines/README.md#compatible-environments)
for existing/custom environments. It does not install uv or download Python. Use --python to select an
existing interpreter and --uv to select an existing uv executable explicitly.
TOPReward (both runtimes), ProcVLM and FailSafe require nvcc and c++ on PATH for
their locked native builds. Do not change drivers or system CUDA to satisfy a
model lock implicitly.

RoboReward requires a loadable FFmpeg 4-8 shared-library set; TOPReward (both
runtimes) requires FFmpeg 4-7 for its locked TorchCodec. An ffmpeg executable
alone is insufficient. The plan checks these libraries without installing
system packages; the final TorchCodec import still checks its actual linkage.

First inspect the plan; --plan validates platform, source paths, lock inputs
shared-library prerequisites and runtime ownership without creating an
environment or installing anything.
It does not check uv/interpreter availability or resolve/install dependencies.
Run on Linux x86_64; --help remains usable on other platforms.

    python envs/setup.py robometer --root /path/to/model-envs --source /path/to/sources/Robometer --plan
    python envs/setup.py robodopamine --root /path/to/model-envs --plan

After reviewing the plan and authorizing dependency downloads/compilation,
remove --plan. Installation can consume substantial disk space and time.
RoboMeter's pinned upstream imports Unsloth, which requires an accelerator even
without loading a model. For that runtime, explicitly select one authorized
device with --probe-gpu GPU; setup rejects installation without it. All package
installation and builds still hide GPUs; only the final offline import probe
sees the selected device. It does not load checkpoints or start training:

    bash envs/setup.sh robometer --probe-gpu 0

On a shared Docker server, run inside the authorized container and keep code,
source checkouts and runtimes under an authorized persistent mount, not on the host.

    python envs/setup.py liv --root /path/to/model-envs --source /path/to/sources/LIV
    python envs/setup.py topreward_molmo --root /path/to/model-envs

The default root is the ignored .model-envs directory in this checkout. Each
runtime is ROOT/ADAPTER, with its interpreter at ROOT/ADAPTER/bin/python.
The script follows one path:

1. Validate the selected baseline, platform, lock files and required source files.
2. Find an existing Python in the compatible series and create a separate uv virtual environment.
3. Export frozen pylock dependencies without local editable-project entries,
   preserving artifact URLs, hashes and pinned git commits; install them into
   the selected runtime without rewriting repository locks. For native build
   packages, install locked prerequisites first, then build without isolation
   using envs/build.txt tools only where the lock does not already supply them.
   Check the installed Torch/toolkit CUDA major version and ninja executable
   before the native build. This is not a driver or GPU-kernel validation.
4. Write runtime-local import paths for the benchmark and required upstream
   source roots; default roots come from source.sh, not a server-wide search.
5. Run uv pip check, import probes, a PNG round trip, YAML/NumPy operations and
   a tiny CPU Torch operation, with Hugging Face offline and GPUs hidden unless
   --probe-gpu explicitly authorizes one device for accelerator-only imports.

Normal isolated source builds may still fetch build-system dependencies. This
is locked runtime deployment, not a claim of fully hermetic native builds.
Sources are linked, not copied or installed with their upstream dependency
resolvers. Automatically resolved checkouts must match the repository and commit
in models.yaml and have no local changes. Explicit --source roots are checked
for required files and hashed, but their repository provenance remains the
caller's responsibility.

### Source validation scope (2026-10-06)

The downloader fetched all six mapped public repositories from GitHub and
validated their required files at the recorded commits. This verifies source
acquisition and file layout, not model inference or paper-score reproduction.
All six were also downloaded on Linux through source.sh; setup --plan resolved
the selected roots and checked the available system prerequisites. Imports from
fresh source passed in existing ProcVLM, VLAC, TOPReward Qwen and FailSafe
runtimes. RoboMeter's existing runtime lacks unsloth, although its lock includes
it; LIV's old runtime has broken Pillow files. Neither is counted as a successful
fresh-source import, and no new full environment installation was performed.
ProcVLM's older server checkout has local model/backend patches; they are not
silently copied into a public download. FailSafe's older server commit
72999db7b210d23856a21b80a67bf52fe14b92e9 could not be fetched from the public
fork during inspection; the reason is not established. At the author's request,
the mapping now retains that original commit rather than substituting the
accessible fork revision. Reuse the original source with explicit --source and
the original runtime/checkpoint. Its current imports and saved SIA execution
artifacts were checked; a public download and fresh installation are not verified.
The public fork is LLaVA-NeXT, not a verified official FailSafe release. See
[FailSafe provenance](baselines/failsafe.md#provenance-not-an-official-failsafe-installation)
before using its name in reported results.

FlashAttention is built from its locked source archive, rather than allowing
its setup hook to guess and download an unrecorded wheel. This can take time;
MAX_JOBS defaults to four if not explicitly set. DeepSpeed installs without
prebuilding ops and selects its CUDA backend even while GPUs are hidden; later
inference may JIT-compile operators and has not been validated by the probe.
The runtime's bin directory precedes inherited PATH, and uv's package/build
cache stays under the selected runtime instead of a shared global cache.

## Configure and maintain

Set the configuration's python field to the printed runtime executable and
checkpoint to released weights. LIV additionally needs model_options.source_root
set to the same official checkout passed to --source. Other mapped source roots
are attached through the runtime's vmbmk_source.pth rather than new YAML fields.
Relocating linked source or benchmark code requires rebuilding its runtime.

setup.json records input fingerprints and installing/checked status. Only a
successful lightweight check produces checked. Unowned directories or changed
inputs are rejected instead of overwritten; choose a new --root. An interrupted
owned runtime can be retried with unchanged inputs. Do not edit linked sources
in place and treat an old checked record as proof that new code was tested.

The probe does not instantiate adapters, load checkpoints, access datasets,
call remote APIs or run model inference. Import checks can expose missing
upstream extras; failures are reported, not silently replaced with another
runtime. Successful deployment is not proof of all-model inference or archived
result reproduction. Keep model installs separate from ordinary CPU tests.

On 2026-10-06, existing remote environments ran the current reduced CPU suite
and a real RoboDopamine-3B VOC/Cycle-VOC/VS single-trajectory smoke evaluation.
This confirms that representative existing runtime, not installation from these
environment declarations. No deployment script was exercised and no packages
were installed. See testing.md for recorded scope, cache reuse and score differences.

VLAC one-shot still needs disjoint references; ProcVLM one-shot needs its task
checkpoint map. Do not substitute base checkpoints for missing trained models.
Preserve LIV's checkpoint and upstream tokenizer assets. Install ftfy==6.3.1
through envs/liv's pinned dependencies; CLIP imports it from the LIV environment's
site-packages. The unused top-level .runtime_liv_deps source copy is removed.
Missing ftfy is an environment error, not a reason to inject a private source path.

vmbmk.sh is a Linux convenience wrapper; use bash vmbmk.sh run CONFIG or pass
any other package CLI command. The package CLI also works directly
where the selected runtime supports it. The batch GPU scheduler uses Linux
locking and is not a Windows integration-test target.
