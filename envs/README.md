# Model environments

Start with the [baseline installation guides](../docs/baselines/README.md) for
model-specific source, checkpoint and run instructions.

Each baseline has an isolated runtime. envs/models.yaml maps supported adapters
to the existing lock projects, pinned public source repositories, required source
files and import checks. envs/source.sh and envs/setup.sh share this mapping.
The repository root pyproject.toml belongs to the lightweight benchmark core,
not a combined model environment.

Download source by baseline name, then deploy on Linux x86_64:

    bash envs/source.sh liv
    bash envs/setup.sh liv --plan
    bash envs/setup.sh liv

Source defaults to .baseline-sources/liv; setup resolves it without --source.
Use source.sh --root and setup.sh --source-root together for another parent,
or setup.sh --source for an existing checkout. For RoboDopamine, RoboReward,
RynnValue, RoboFAC and TOPReward Molmo, no baseline checkout is required.
The launcher Python needs PyYAML; VMBMK_PYTHON selects that interpreter.
RoboMeter requires --probe-gpu GPU for the final upstream Unsloth import;
other runtimes keep GPUs hidden by default. No model is loaded during setup.
Install only after reviewing --plan and authorizing dependency downloads.
See [the environment guide](../docs/environments.md) for the complete mapping,
compatible Python series, required uv features, source paths, native builds and check limits.
Keep each pyproject.toml, uv.lock and .python-version together when copying.

TOPReward Qwen/Molmo share declarations, not runtimes. FailSafe is SIA-only and
uses failsafe_inference. GVL, ReWiND and the FailSafe simulation environment are
retained historical inputs, not deployment choices.
