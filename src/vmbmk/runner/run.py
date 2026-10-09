from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vmbmk.data.dataset import Dataset
from vmbmk.errors import ConfigurationError
from .policy import evaluation_policy
from vmbmk.metrics.registry import (
    SUPPORTED_METRICS,
    canonical_metric,
    metric_runner,
    selected_task_ids,
)
from vmbmk.adapters.registry import ADAPTERS, adapter_class
from vmbmk.metrics.sia.scoring import SIA_AGGREGATION, SIA_EPISODE_SELECTION, SIA_PROTOCOL
from vmbmk.metrics.csvc import CSVC_PROTOCOL
from vmbmk.metrics.trr import TRR_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.vs import VS_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.forward import VOC_PROTOCOL
from vmbmk.metrics.cycle_voc_vs.planning import plan_cycle
from vmbmk.serialization import read_jsonl, read_yaml_object
from .provenance import run_identity


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_operations(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConfigurationError(f"{label} operations checkpoint is missing: {path}")
    try:
        rows = list(read_jsonl(path))
    except ValueError as exc:
        raise ConfigurationError(f"{label} operations checkpoint is invalid: {exc}") from exc
    if not rows:
        raise ConfigurationError(f"{label} operations checkpoint is empty: {path}")


def _result_config(path: Path, label: str) -> dict[str, Any]:
    try:
        value = read_yaml_object(path)
    except ValueError as exc:
        raise ConfigurationError(f"{label} config is invalid: {path}") from exc
    value = dict(value)
    value.pop("gpu", None)
    value.pop("batch_size", None)
    options = value.get("model_options")
    if isinstance(options, dict):
        options.pop("sia_workers", None)
        options.pop("sia_timeout", None)
    return value


def _validate_saved_config(source: Path, saved: Path, label: str) -> bool:
    if not saved.is_file():
        raise ConfigurationError(f"{label} config is missing: {saved}")
    if saved.read_bytes() == source.read_bytes():
        return False
    requested = _result_config(source, label)
    previous = _result_config(saved, label)
    if requested != previous:
        # An unfinished standalone judge can switch endpoints: each score's
        # cache identity already includes its endpoint. Never reuse final metrics.
        if (
            label == "resumable run"
            and requested.get("sia_stage") == previous.get("sia_stage") == "judge"
        ):
            partial = saved.parent / ".metrics.partial.json"
            has_scores = partial.exists() and bool(
                {"sia"} & set(json.loads(partial.read_text()))
            )
            requested.get("model_options", {}).pop("sia_base_url", None)
            previous.get("model_options", {}).pop("sia_base_url", None)
            if not has_scores and requested == previous:
                return True
        raise ConfigurationError(
            f"{label} config does not match the requested run: {saved}"
        )
    return True


def _read_metrics(path: Path, expected: set[str], label: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"{label} metrics checkpoint is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} metrics checkpoint must be an object")
    unknown = sorted(set(value) - expected)
    if "voc" in value and (
        not isinstance(value["voc"], dict) or value["voc"].get("protocol") != VOC_PROTOCOL
    ):
        raise ConfigurationError(f"{label} VOC checkpoint predates cycle-forward scoring; use a new output directory")
    if unknown:
        raise ConfigurationError(
            f"{label} metrics checkpoint contains unknown metrics: {unknown}"
        )
    if "sia" in value and (
        not isinstance(value["sia"], dict)
        or value["sia"].get("protocol") != SIA_PROTOCOL
        or value["sia"].get("aggregation") != SIA_AGGREGATION
    ):
        raise ConfigurationError(
            f"{label} SIA checkpoint uses an obsolete scoring protocol; "
            "run classification CE in a new output directory"
        )
    if "sia" in value and value["sia"].get("episode_selection") != SIA_EPISODE_SELECTION:
        raise ConfigurationError(
            f"{label} SIA checkpoint uses an obsolete trajectory selection; "
            "rejudge saved responses in a new output directory with diverse ordinal mapping"
        )
    if "csvc" in value and (
        not isinstance(value["csvc"], dict)
        or value["csvc"].get("protocol") != CSVC_PROTOCOL
    ):
        raise ConfigurationError(
            f"{label} CSVC checkpoint uses an obsolete scoring protocol; "
            "use a new output directory"
        )
    if "trr" in value and (
        not isinstance(value["trr"], dict)
        or value["trr"].get("protocol") != TRR_PROTOCOL
    ):
        raise ConfigurationError(
            f"{label} TRR checkpoint uses an obsolete protocol; use a new output directory"
        )
    if "vs" in value and (
        not isinstance(value["vs"], dict) or value["vs"].get("protocol") != VS_PROTOCOL
    ):
        raise ConfigurationError(
            f"{label} VS checkpoint uses an obsolete protocol; use a new output directory"
        )
    return value


def _cleanup_checkpoints(directory: Path, metric_names: tuple[str, ...]) -> None:
    for name in (".metrics.partial.json",) + tuple(
        name
        for metric in metric_names
        for name in (
            f".{metric}.operations.jsonl",
            f".{metric}.operations.jsonl.cycle.jsonl",
        )
    ):
        path = directory / name
        if path.exists():
            path.unlink()


def _validate_completed_run(
    source: Path,
    run_dir: Path,
    metric_names: tuple[str, ...],
) -> None:
    _validate_saved_config(source, run_dir / "config.yaml", "completed run")
    expected = set(metric_names)
    metrics = _read_metrics(run_dir / "metrics.json", expected, "completed run")
    if set(metrics) != expected:
        raise ConfigurationError(
            "completed run metric coverage mismatch; "
            f"missing={sorted(expected - set(metrics))}"
        )
    _validate_operations(run_dir / "operations.jsonl", "completed run")
    _cleanup_checkpoints(run_dir, metric_names)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label} must be a non-empty string")
    return value


def _text_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{label} must be a non-empty string array")
    items = tuple(_text(item, label) for item in value)
    if len(items) != len(set(items)):
        raise ConfigurationError(f"{label} contains duplicates")
    return items


def _metric_name(value: Any, label: str) -> str:
    name = canonical_metric(_text(value, label))
    if name not in SUPPORTED_METRICS:
        raise ConfigurationError(
            f"{label} must be one of {sorted(SUPPORTED_METRICS)}"
        )
    return name


def _metrics(value: Any, label: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if not isinstance(value, dict) or not value:
        raise ConfigurationError(f"{label} must be a non-empty object")
    result = []
    for metric, settings in value.items():
        name = _metric_name(metric, label)
        if not isinstance(settings, dict):
            raise ConfigurationError(
                f"{label}.{name} must contain mode and domains"
            )
        expected = {"mode", "domains"}
        missing = sorted(expected - set(settings))
        unknown = sorted(set(settings) - expected)
        if missing or unknown:
            raise ConfigurationError(
                f"{label}.{name} fields do not match; "
                f"missing={missing}, unknown={unknown}"
            )
        selected_mode = _text(settings["mode"], f"{label}.{name}.mode")
        if selected_mode not in {"base", "native"}:
            raise ConfigurationError(
                f"{label}.{name} must be 'base' or 'native'"
            )
        domains = settings["domains"]
        if not isinstance(domains, list) or not domains:
            raise ConfigurationError(
                f"{label}.{name}.domains must be a non-empty string array"
            )
        selected_domains = tuple(
            _text(domain, f"{label}.{name}.domains") for domain in domains
        )
        if len(selected_domains) != len(set(selected_domains)):
            raise ConfigurationError(
                f"{label}.{name}.domains contains duplicates"
            )
        invalid = sorted(set(selected_domains) - {"id", "env", "emb"})
        if invalid:
            raise ConfigurationError(
                f"{label}.{name}.domains contains invalid values: {invalid}"
            )
        if name == "csvc" and selected_domains != ("id",):
            raise ConfigurationError("CSVC supports only domains: [id]")
        result.append((name, selected_mode, selected_domains))
    names = [name for name, _, _ in result]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"{label} contains duplicate metric names")
    return tuple(result)


@dataclass(frozen=True)
class _ReferenceSettings:
    shot_mode: str | None = None
    data: Path | None = None
    view: str | None = None
    count: int | None = None


def _reference_settings(
    row: dict[str, Any], model: str, source: Path
) -> _ReferenceSettings:
    """Parse model-specific reference inputs without resolving their paths."""
    if model not in {"vlac", "robodopamine"}:
        if {"shot_mode", "reference_data", "reference_view", "ref_num"} & set(row):
            raise ConfigurationError(
                f"{source}.reference fields are only valid for VLAC or RoboDopamine"
            )
        return _ReferenceSettings()

    adapter = adapter_class(model)
    default_shot_mode = (
        adapter.resolve_shot_mode(row) if model == "vlac" else adapter.default_shot_mode
    )
    default_reference_view = adapter.default_reference_view

    shot_mode = _text(row.get("shot_mode", default_shot_mode), f"{source}.shot_mode")
    allowed_modes = {"zero_shot", "one_shot"} if model == "vlac" else {"one_shot"}
    if shot_mode not in allowed_modes:
        raise ConfigurationError(
            f"{source}.shot_mode must be one of {sorted(allowed_modes)}"
        )
    reference_data = None
    reference_view = None
    if shot_mode == "one_shot":
        reference_data = Path(
            _text(row.get("reference_data"), f"{source}.reference_data")
        )
        reference_view = _text(
            row.get(
                "reference_view",
                default_reference_view,
            ),
            f"{source}.reference_view",
        )
    elif "reference_data" in row or "reference_view" in row:
        raise ConfigurationError(
            f"{source}.reference_data/reference_view are only valid for one_shot"
        )
    ref_num = row.get("ref_num", adapter_class("vlac").default_ref_num)
    if isinstance(ref_num, bool) or not isinstance(ref_num, int) or ref_num < 2:
        raise ConfigurationError(f"{source}.ref_num must be an integer of at least 2")
    return _ReferenceSettings(shot_mode, reference_data, reference_view, ref_num)


def _gpu_selection(gpu: Any, source: Path) -> int | tuple[int, ...]:
    """Validate one GPU or an explicitly ordered nonempty GPU set."""
    if isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0:
        pass
    elif (
        isinstance(gpu, list)
        and gpu
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
            for item in gpu
        )
    ):
        gpu = tuple(gpu)
        if len(gpu) != len(set(gpu)):
            raise ConfigurationError(f"{source}.gpu contains duplicates")
    else:
        raise ConfigurationError(
            f"{source}.gpu must be a non-negative integer or a non-empty "
            "array of them"
        )
    return gpu


def _task_scope(
    value: Any, configured_names: set[str], source: Path, field: str,
) -> dict[str, tuple[str, ...]] | None:
    """Parse optional per-metric task scope separately from adapter settings."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigurationError(f"{source}.{field} must be an object")
    scopes = {}
    for metric, tasks in value.items():
        canonical = _metric_name(metric, f"{source}.{field}")
        if canonical not in configured_names:
            raise ConfigurationError(f"{source}.{field} contains unsupported metric: {metric}")
        if canonical in scopes:
            raise ConfigurationError(f"{source}.{field} contains duplicate metric: {metric}")
        scopes[canonical] = _text_list(tasks, f"{source}.{field}.{metric}")
    return scopes


@dataclass(frozen=True)
class RunConfig:
    """Validated run settings; metric tuples contain (name, mode, domains)."""

    model: str
    gpu: int | tuple[int, ...]
    batch_size: int
    python: Path
    checkpoint: Path
    data: Path
    output: Path
    metrics: tuple[tuple[str, str, tuple[str, ...]], ...]
    tasks: tuple[str, ...]
    metric_tasks: dict[str, tuple[str, ...]] | None = None
    candidate_tasks: dict[str, tuple[str, ...]] | None = None
    shot_mode: str | None = None
    reference_data: Path | None = None
    reference_view: str | None = None
    ref_num: int | None = None
    model_options: dict[str, Any] | None = None
    sia_stage: str = "all"
    sia_responses: Path | None = None

    def model_config(self) -> dict[str, Any]:
        """Return the adapter mapping shared by evaluation and visualization."""
        model = {
            "adapter": self.model,
            "python": str(self.python),
            "checkpoint": str(self.checkpoint),
            "batch_size": self.batch_size,
        }
        if self.shot_mode is not None:
            model["shot_mode"] = self.shot_mode
            model["ref_num"] = self.ref_num
        if self.reference_data is not None:
            model["reference_data"] = str(self.reference_data)
            model["reference_view"] = self.reference_view
        if self.model_options:
            model.update(self.model_options)
        return model

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        source = Path(path).resolve()
        try:
            row: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"{source}: invalid YAML: {exc}") from exc
        if not isinstance(row, dict):
            raise ConfigurationError(f"{source}: run config must be an object")
        required = {
            "model",
            "gpu",
            "batch_size",
            "python",
            "checkpoint",
            "data",
            "output",
            "metrics",
            "tasks",
        }
        optional = {
            "shot_mode",
            "reference_data",
            "reference_view",
            "ref_num",
            "model_options",
            "metric_tasks",
            "candidate_tasks",
            "sia_stage",
            "sia_responses",
        }
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - required - optional)
        if missing or unknown:
            raise ConfigurationError(
                f"{source}: fields do not match; missing={missing}, unknown={unknown}"
            )
        sia_stage = row.get("sia_stage", "all")
        if sia_stage not in {"all", "generate", "judge"}:
            raise ConfigurationError(f"{source}.sia_stage must be all, generate, or judge")
        sia_responses = row.get("sia_responses")
        if sia_responses is not None:
            sia_responses = Path(_text(sia_responses, f"{source}.sia_responses"))
            if not sia_responses.is_absolute():
                sia_responses = (source.parent / sia_responses).resolve()
        model = _text(row["model"], f"{source}.model")
        if model not in ADAPTERS:
            raise ConfigurationError(f"{source}.model must be one of {sorted(ADAPTERS)}")
        model_options = row.get("model_options")
        if model_options is not None and not isinstance(model_options, dict):
            raise ConfigurationError(f"{source}.model_options must be an object")
        reserved = sorted(
            set(model_options or {})
            & {
                "adapter",
                "python",
                "checkpoint",
                "batch_size",
                "sia_api_key",
                "sia_auth_token",
            }
        )
        if reserved:
            raise ConfigurationError(
                f"{source}.model_options contains reserved fields: {reserved}"
            )
        if "tga_candidate_tasks" in (model_options or {}):
            raise ConfigurationError(
                f"{source}.model_options.tga_candidate_tasks is retired; use top-level candidate_tasks"
            )
        reference = _reference_settings(row, model, source)
        parsed_metrics = _metrics(row["metrics"], f"{source}.metrics")
        configured_names = {name for name, _, _ in parsed_metrics}
        metric_tasks = _task_scope(row.get("metric_tasks"), configured_names, source, "metric_tasks")
        candidate_tasks = _task_scope(
            row.get("candidate_tasks"), configured_names & {"tga_easy", "tga_hard"},
            source, "candidate_tasks",
        )
        gpu = _gpu_selection(row["gpu"], source)
        batch_size = row["batch_size"]
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size == 0
            or batch_size < -1
            or (batch_size == -1 and model != "procvlm")
        ):
            raise ConfigurationError(
                f"{source}.batch_size must be a positive integer"
                + (" or -1 for procvlm" if model == "procvlm" else "")
            )
        python = Path(_text(row["python"], f"{source}.python"))
        checkpoint = Path(_text(row["checkpoint"], f"{source}.checkpoint"))
        data = Path(_text(row["data"], f"{source}.data"))
        output = Path(_text(row["output"], f"{source}.output"))
        python, checkpoint, data, output = (
            path if path.is_absolute() else (source.parent / path).resolve()
            for path in (python, checkpoint, data, output)
        )
        reference_data = reference.data
        if reference_data is not None:
            reference_data = (
                reference_data
                if reference_data.is_absolute()
                else (source.parent / reference_data).resolve()
            )
        return cls(
            model=model,
            gpu=gpu,
            batch_size=batch_size,
            python=python,
            checkpoint=checkpoint,
            data=data,
            output=output,
            metrics=parsed_metrics,
            tasks=_text_list(row["tasks"], f"{source}.tasks"),
            metric_tasks=metric_tasks,
            candidate_tasks=candidate_tasks,
            shot_mode=reference.shot_mode,
            reference_data=reference_data,
            reference_view=reference.view,
            ref_num=reference.count,
            model_options=dict(model_options) if model_options is not None else None,
            sia_stage=sia_stage,
            sia_responses=sia_responses,
        )


def _select_metric_tasks(config: RunConfig, dataset: Dataset) -> dict[str, tuple[str, ...]]:
    selections = {}
    policy_config = {
        "model": config.model, "checkpoint": str(config.checkpoint),
        "model_options": config.model_options or {},
    }
    policy = evaluation_policy(policy_config)
    for metric, _, domains in config.metrics:
        requested = (config.metric_tasks or {}).get(metric, config.tasks)
        unknown = sorted(set(requested) - set(config.tasks))
        if unknown:
            raise ConfigurationError(f"metric_tasks.{metric} contains tasks outside config.tasks: {unknown}")
        candidates = (config.candidate_tasks or {}).get(metric)
        if candidates is not None:
            unknown_candidates = sorted(set(candidates) - set(dataset.tasks))
            if unknown_candidates:
                raise ConfigurationError(f"candidate_tasks.{metric} contains unknown tasks: {unknown_candidates}")
        selected = selected_task_ids(metric, dataset, requested, domains)
        for task in selected:
            reason = policy.reason(policy_config, metric, task)
            if reason is not None:
                raise ConfigurationError(f"{metric}/{task}: {reason}")
        if not selected:
            raise ConfigurationError(f"metric {metric!r} has no compatible task in the run config")
        selections[metric] = selected
    return selections


def _prepare_run_directory(
    config: RunConfig, source: Path, run_dir: Path, temporary: Path, execution_mode: str,
) -> None:
    """Archive explicit reruns and validate the resumable staging directory."""
    if execution_mode == "rerun":
        previous = [path for path in (run_dir, temporary) if path.exists()]
        if previous:
            archive_root = config.output / "rerun_history"
            archive_root.mkdir(parents=True, exist_ok=True)
            history = Path(tempfile.mkdtemp(prefix=f"{source.stem}-", dir=archive_root))
            for path in previous:
                path.rename(history / path.name)

    config.output.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        saved_config = temporary / "config.yaml"
        if saved_config.exists():
            if _validate_saved_config(source, saved_config, "resumable run"):
                _write_text_atomic(
                    saved_config, source.read_text(encoding="utf-8")
                )
        elif any(temporary.iterdir()):
            raise ConfigurationError(
                f"resumable run has files but no config: {temporary}"
            )
        else:
            shutil.copyfile(source, saved_config)
    else:
        temporary.mkdir()
        shutil.copyfile(source, temporary / "config.yaml")


def _finish_run(
    temporary: Path, run_dir: Path, metric_names: tuple[str, ...], metrics: dict[str, Any],
) -> Path:
    """Publish a complete local run only after its operation files are present."""
    operation_output = temporary / "operations.jsonl"
    operation_staging = temporary / "operations.jsonl.tmp"
    with operation_staging.open("w", encoding="utf-8", newline="\n") as output:
        for metric in metric_names:
            metric_operations = temporary / f".{metric}.operations.jsonl"
            _validate_operations(metric_operations, metric)
            with metric_operations.open("r", encoding="utf-8") as source_handle:
                shutil.copyfileobj(source_handle, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(operation_staging, operation_output)
    _write_text_atomic(
        temporary / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    )
    os.replace(temporary, run_dir)
    _cleanup_checkpoints(run_dir, metric_names)
    return run_dir


def run_evaluation(
    config_path: str | Path, *, sia_stage: str | None = None,
    execution_mode: str = "normal",
) -> Path:
    source = Path(config_path).resolve()
    config = RunConfig.load(config_path)
    sia_stage = sia_stage if sia_stage is not None else config.sia_stage
    run_dir = config.output / source.stem
    temporary = config.output / f".{source.stem}.tmp"
    metric_names = tuple(metric for metric, _, _ in config.metrics)
    if config.model == "failsafe" and metric_names != ("sia",):
        raise ConfigurationError("FailSafe supports only SIA evaluation")
    if sia_stage not in {"all", "generate", "judge"}:
        raise ConfigurationError(f"invalid SIA stage: {sia_stage}")
    if sia_stage != "all" and metric_names != ("sia",):
        raise ConfigurationError(
            "--sia-stage requires a config containing only SIA or SIA-candidate"
        )
    if execution_mode not in {"normal", "rerun"}:
        raise ConfigurationError(f"invalid execution mode: {execution_mode}")
    # Scope dataset validation to the configured metrics.  A run should fail
    # only for malformed inputs that its selected metric implementations can
    # actually read; the standalone ``validate`` command remains strict.
    dataset = Dataset.load(config.data, metrics=metric_names)
    unknown_tasks = sorted(set(config.tasks) - set(dataset.tasks))
    if unknown_tasks:
        raise ConfigurationError(f"unknown tasks: {unknown_tasks}")
    selections = _select_metric_tasks(config, dataset)
    identity = run_identity(dataset)
    existing = run_dir if run_dir.exists() and execution_mode == "normal" else temporary
    if existing.exists() and execution_mode != "rerun":
        recorded = existing / "provenance.json"
        if not recorded.is_file() or json.loads(recorded.read_text(encoding="utf-8")) != identity:
            raise ConfigurationError("run inputs or implementation changed, or provenance is missing; use a new output directory or explicit rerun")
    if run_dir.exists() and execution_mode == "normal":
        _validate_completed_run(source, run_dir, metric_names)
        if metric_names != ("sia",):
            from vmbmk.tools.results.validation import inspect_result

            report = inspect_result(source, run_dir, dataset)
            if not report["valid"]:
                raise ConfigurationError(f"completed run is invalid: {report['reason']}")
        return run_dir
    settings = {metric: (mode, domains) for metric, mode, domains in config.metrics}
    if "vs" in settings and settings["vs"][0] != "base":
        raise ConfigurationError("VS requires mode=base")
    if {"vs", "cycle_voc"}.issubset(settings):
        if settings["cycle_voc"][0] != "base":
            raise ConfigurationError("VS cannot reuse metric-native Cycle-VOC predictions")
        if not set(selections["vs"]).issubset(selections["cycle_voc"]) or not set(
            settings["vs"][1]
        ).issubset(settings["cycle_voc"][1]):
            raise ConfigurationError("Cycle-VOC must cover all VS tasks and domains")
    shared_cycle_plan = (
        plan_cycle(dataset, selections["cycle_voc"], settings["cycle_voc"][1])
        if "cycle_voc" in settings and {"voc", "vs"}.intersection(settings) else None
    )

    _prepare_run_directory(config, source, run_dir, temporary, execution_mode)
    _write_text_atomic(temporary / "provenance.json", json.dumps(identity, indent=2) + "\n")

    model = config.model_config()

    metrics_checkpoint = temporary / ".metrics.partial.json"
    metrics = _read_metrics(metrics_checkpoint, set(metric_names), "resumable run")
    for metric, mode, domains in sorted(config.metrics, key=lambda item: {"cycle_voc": 0, "voc": 1, "vs": 2}.get(item[0], 1)):
        metric_operations = temporary / f".{metric}.operations.jsonl"
        if metric in metrics:
            _validate_operations(metric_operations, metric)
            continue
        metric_result = metric_runner(metric)(
            config.data,
            model,
            selections[metric],
            metric_operations,
            config.gpu,
            mode=mode,
            domains=domains,
            dataset=dataset,
            **(
                {"candidate_task_ids": (config.candidate_tasks or {}).get(metric)}
                if metric in {"tga_easy", "tga_hard"} else {}
            ),
            **(
                {"stage": sia_stage, "responses_path": config.sia_responses}
                if metric in {"sia"}
                else {}
            ),
            **(
                {
                    "cycle_operation_path": temporary / ".cycle_voc.operations.jsonl",
                    "cycle_task_ids": selections["cycle_voc"],
                    "cycle_domains": settings["cycle_voc"][1],
                    "cycle_plan": shared_cycle_plan,
                }
                if metric == "vs" and "cycle_voc" in settings else {}
            ),
            **(
                {"plan": shared_cycle_plan}
                if metric == "cycle_voc" and shared_cycle_plan is not None else {}
            ),
            **(
                {
                    "plan": shared_cycle_plan,
                    "cycle_operation_path": temporary / ".cycle_voc.operations.jsonl",
                }
                if metric == "voc" and shared_cycle_plan is not None
                and mode == settings["cycle_voc"][0] == "base"
                and set(selections["voc"]) == set(selections["cycle_voc"])
                and set(domains) == set(settings["cycle_voc"][1]) else {}
            ),
        )
        if metric in {"sia"} and sia_stage == "generate":
            _write_text_atomic(
                temporary / "sia_generation.json",
                json.dumps(metric_result, ensure_ascii=False, indent=2) + "\n",
            )
            return temporary
        _validate_operations(metric_operations, metric)
        metrics[metric] = metric_result
        _write_text_atomic(
            metrics_checkpoint,
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        )

    return _finish_run(temporary, run_dir, metric_names, metrics)
