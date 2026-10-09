"""CPU-only result publication, independent of model and GPU scheduling."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from vmbmk.metrics.registry import canonical_metric
from vmbmk.adapters.registry import canonical_baseline
from vmbmk.serialization import read_yaml_object

PUBLICATION_SCHEMA_VERSION = "robovalue-results-v1"


def csvc_domains(result: dict[str, Any]) -> dict[str, Any]:
    """Expose ID-only coverage without recomputing the CSVC estimand."""
    return {
        "id": {
            "mean": result["score"],
            "tasks": {task: row["score"] for task, row in result["task_results"].items()},
        }
    }


def _index_payload(final_root: Path, updates: dict[Path, bytes]) -> dict[str, Any]:
    """Build the canonical index from published summaries, using relative links."""
    entries = []
    paths = set(final_root.glob("*/*/summary.json"))
    paths.update(path for path in updates if path.name == "summary.json" and len(path.relative_to(final_root).parts) == 3)
    for path in sorted(paths):
        content = updates[path] if path in updates else path.read_bytes()
        summary = json.loads(content)
        entries.append({
            "baseline": summary["baseline"],
            "metric": summary["metric"],
            "tasks": len(summary["tasks"]),
            "summary": path.relative_to(final_root).as_posix(),
            "source_run": summary.get("source_run"),
            "protocol": summary["result"].get("protocol"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "updated_at": summary.get("updated_at"),
        })
    index = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "updated_at": now_iso(),
        "entries": entries,
    }
    return index


def refresh_index(final_root: Path) -> dict[str, Any]:
    """Refresh the index under the publication lock, recovering interrupted writes."""
    with PublicationBatch(final_root) as batch:
        pass
    return json.loads(batch.updates[batch.root / "index.json"])


@dataclass(frozen=True)
class PublicationRequest:
    """Validated result artifacts and publication metadata, independent of scheduling."""

    baseline: str
    config_path: Path
    run_name: str
    result: dict[str, Any]
    resume_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_mode: str = "normal"


def _validate_components(components: list[str], *, task_names: bool = False) -> None:
    """Reject nonportable path components and filesystem identity collisions."""
    seen: dict[str, str] = {}
    for component in components:
        if (
            not isinstance(component, str)
            or not component
            or component in {".", ".."}
            or component[-1] in {".", " "}
            or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in component)
        ):
            raise ValueError(f"Unsafe publication identity: {component!r}")
        normalized = unicodedata.normalize("NFC", component).casefold()
        stem = normalized.split(".")[0]
        if stem in {
            "con", "prn", "aux", "nul",
            *(f"com{number}" for number in range(1, 10)),
            *(f"lpt{number}" for number in range(1, 10)),
        }:
            raise ValueError(f"Unsafe publication identity: {component!r}")
        if task_names and normalized == "summary":
            raise ValueError("Task name conflicts with summary.json")
        if normalized in seen and seen[normalized] != component:
            raise ValueError(f"Colliding publication identities: {seen[normalized]!r}, {component!r}")
        seen[normalized] = component


@contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    """Hold a nonblocking OS lock; the kernel releases it if a writer exits."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".publication.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError(
                    "Publication root is busy; retry after the current writer finishes"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "Publication root is busy; retry after the current writer finishes"
                ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class PublicationBatch:
    """Stage related updates, commit the index last, and recover via an undo journal.

    Writers and consistent local readers must hold the publication lock. Unlocked
    readers can see intermediate files and must verify the index summary hashes.
    This is process-interruption recovery, not a filesystem power-loss guarantee.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.updates: dict[Path, bytes] = {}
        self.deletions: set[Path] = set()
        self._active = False

    def _target(self, path: Path) -> Path:
        target = path.resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError(f"Publication destination escapes root: {path}")
        return target

    def read_json(self, path: Path) -> Any:
        """Read staged content when present, otherwise read the published file."""
        target = self._target(path)
        return json.loads(self.updates[target] if target in self.updates else target.read_bytes())

    def exists(self, path: Path) -> bool:
        """Check file existence in the staged publication view."""
        target = self._target(path)
        return target in self.updates or (target not in self.deletions and target.is_file())

    def write_json(self, path: Path, value: Any) -> None:
        """Stage finite JSON without modifying published files."""
        target = self._target(path)
        self.updates[target] = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        self.deletions.discard(target)

    def delete(self, path: Path) -> None:
        """Stage removal of a caller-owned file inside the publication root."""
        target = self._target(path)
        self.updates.pop(target, None)
        self.deletions.add(target)

    def _recover(self) -> None:
        journal = self.root / ".publication-journal.json"
        if not journal.exists():
            return
        originals = json.loads(journal.read_bytes())
        targets = [(self._target(self.root / name), content) for name, content in originals.items()]
        for target, content in targets:
            if content is None:
                target.unlink(missing_ok=True)
            else:
                _atomic_bytes(target, base64.b64decode(content, validate=True))
        journal.unlink()

    def __enter__(self) -> PublicationBatch:
        self._lock = _publication_lock(self.root)
        self._lock.__enter__()
        try:
            self._recover()
        except BaseException:
            self._lock.__exit__(*sys.exc_info())
            raise
        self._active = True
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        try:
            if exception_type is None:
                self.write_json(self.root / "index.json", _index_payload(self.root, self.updates))
                paths = set(self.updates) | self.deletions
                originals = {
                    path.relative_to(self.root).as_posix(): (
                        base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else None
                    )
                    for path in sorted(paths)
                }
                atomic_json(self.root / ".publication-journal.json", originals)
                try:
                    for path in sorted(self.deletions):
                        path.unlink(missing_ok=True)
                    for path in sorted(
                        self.updates, key=lambda target: (target.name == "index.json", target)
                    ):
                        _atomic_bytes(path, self.updates[path])
                except BaseException:
                    self._recover()
                    raise
                (self.root / ".publication-journal.json").unlink()
        finally:
            self._active = False
            self._lock.__exit__(exception_type, exception, traceback)





def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    _atomic_bytes(path, (content + "\n").encode("utf-8"))


def _config_metadata(path: Path) -> dict[str, Any]:
    """Require valid model metadata instead of silently using the run directory."""
    value = read_yaml_object(path)
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError(f"Publication config requires an explicit model: {path}")
    return value


def _result_mean(values: list[Any]) -> float:
    """Reject malformed measured values rather than averaging a partial cohort."""
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) for value in values
    ):
        raise ValueError("task-level results must contain finite numeric scores")
    return sum(float(value) for value in values) / len(values)


def _merge_domain_tasks(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge the shared domain/task structure without borrowing a metric identity."""
    merged = copy.deepcopy(existing)
    merged_domains = merged.setdefault("domains", {})
    incoming_domains = incoming.get("domains", {})
    if not isinstance(merged_domains, dict) or not isinstance(incoming_domains, dict):
        raise ValueError("domain-task results must contain domains objects")
    for domain, incoming_value in incoming_domains.items():
        if not isinstance(incoming_value, dict):
            raise ValueError(f"result domain {domain} must be an object")
        target = merged_domains.setdefault(domain, {})
        if not isinstance(target, dict):
            raise ValueError(f"existing result domain {domain} must be an object")
        target_tasks = target.setdefault("tasks", {})
        incoming_tasks = incoming_value.get("tasks", {})
        if not isinstance(target_tasks, dict) or not isinstance(incoming_tasks, dict):
            raise ValueError(f"result domain {domain} must contain a tasks object")
        target_tasks.update(copy.deepcopy(incoming_tasks))

        # VOC task values are objects with mean/voc/vroc; the other
        # published task metrics use scalar values.  Preserve any additional
        # fields while refreshing their standard aggregates.
        values = list(target_tasks.values())
        if not values:
            continue
        if all(isinstance(value, dict) and "mean" in value for value in values):
            target["mean"] = _result_mean([value["mean"] for value in values])
            for key in ("voc", "vroc"):
                key_values = [value[key] for value in values if key in value]
                if key_values:
                    if len(key_values) != len(values):
                        raise ValueError(f"result domain {domain} has incomplete {key} task scores")
                    target[key] = _result_mean(key_values)
        else:
            target["mean"] = _result_mean(values)

    domain_values = [value for value in merged_domains.values() if isinstance(value, dict)]
    means = [
        value["mean"]
        for value in domain_values
        if isinstance(value.get("mean"), (int, float))
    ]
    if means:
        merged["mean"] = sum(float(value) for value in means) / len(means)
    for key in ("voc", "vroc"):
        values = [value[key] for value in domain_values if isinstance(value.get(key), (int, float))]
        if values:
            merged[key] = sum(float(value) for value in values) / len(values)
    return merged


def _merge_metric_result(
    metric: str,
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge task-level metric output from an incremental run.

    Direct single-config runs normally replace a metric atomically. Missing-task
    repairs can opt into this helper so a one-task run fills a published
    summary without deleting the already measured tasks.  Aggregate fields are
    recomputed from the merged per-domain task values for the metrics whose
    result schema is task based. CSVC uses its own covered-task aggregate;
    VS components and TRR group records retain their distinct diagnostics.
    """
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        raise ValueError(f"{metric} merge requires result objects")
    if existing.get("protocol") != incoming.get("protocol"):
        raise ValueError(f"Cannot merge {metric} results with different protocols")
    if metric == "csvc":
        from vmbmk.metrics.csvc import CSVC_PROTOCOL, aggregate_csvc

        if incoming.get("protocol") != CSVC_PROTOCOL:
            raise ValueError("Cannot merge CSVC results with an obsolete scoring protocol")

        task_results = copy.deepcopy(existing.get("task_results", {}))
        task_results.update(copy.deepcopy(incoming.get("task_results", {})))
        aggregate = aggregate_csvc(task_results)
        result = {
            **incoming,
            **aggregate,
            "mean": aggregate["score"],
            "task_results": task_results,
        }
        result["domains"] = csvc_domains(result)
        return result
    if metric not in {"voc", "cycle_voc", "voc_mem", "sa", "tga_easy", "tga_hard", "fpl", "trr", "vs"}:
        raise ValueError(f"Incremental merge is unsupported for {metric}; publish a complete run")
    merged = _merge_domain_tasks(existing, incoming)
    merged_domains = merged["domains"]
    incoming_domains = incoming["domains"]
    if metric == "vs":
        for key in ("er", "nonflat_time_fraction"):
            merged[key] = _merge_domain_tasks(existing[key], incoming[key])
    if metric == "trr":
        merged["task_scores"] = {
            f"{domain}::{task}": score
            for domain, value in merged_domains.items()
            for task, score in value["tasks"].items()
        }

    # TGA carries episode records in addition to domain task summaries.
    if isinstance(incoming.get("episodes"), list):
        old_episodes = merged.setdefault("episodes", [])
        if isinstance(old_episodes, list):
            replaced = {
                (domain, task)
                for domain, value in incoming_domains.items()
                for task in value.get("tasks", {})
            }
            old_episodes[:] = [
                item for item in old_episodes
                if (item.get("domain"), item.get("task_id")) not in replaced
            ]
            seen = {json.dumps(item, sort_keys=True) for item in old_episodes}
            for item in incoming["episodes"]:
                marker = json.dumps(item, sort_keys=True)
                if marker not in seen:
                    old_episodes.append(copy.deepcopy(item))
                    seen.add(marker)
    if metric in {"tga_easy", "tga_hard"}:
        old_matrix = merged.setdefault("confusion_matrix", {})
        new_matrix = incoming.get("confusion_matrix", {})
        if isinstance(old_matrix, dict) and isinstance(new_matrix, dict):
            for domain, task_rows in new_matrix.items():
                if not isinstance(task_rows, dict):
                    continue
                target_domain = old_matrix.setdefault(domain, {})
                if not isinstance(target_domain, dict):
                    target_domain = {}
                    old_matrix[domain] = target_domain
                for task_id, candidate_scores in task_rows.items():
                    if isinstance(candidate_scores, dict):
                        target_domain[task_id] = copy.deepcopy(candidate_scores)
        merged["protocol"] = incoming.get("protocol", merged.get("protocol"))
        merged["primary"] = incoming.get("primary", merged.get("primary"))
    if metric == "vs":
        merged["n_episodes"] = len(merged["episodes"])
        replaced = {(domain, task) for domain, value in incoming_domains.items() for task in value["tasks"]}
        merged["exclusions"] = [
            item for item in existing["exclusions"]
            if (item["domain"], item["task_id"]) not in replaced
        ] + copy.deepcopy(incoming["exclusions"])
    if metric == "trr":
        replaced = {(domain, task) for domain, value in incoming_domains.items() for task in value["tasks"]}
        merged["groups"] = [
            item for item in existing["groups"]
            if (item["domain"], item["task_id"]) not in replaced
        ] + copy.deepcopy(incoming["groups"])
        merged["n_groups"] = len(merged["groups"])
    return merged


def _published_tga_result(metric_result: dict[str, Any]) -> dict[str, Any]:
    """Retain primary scores and diagnostics needed for incremental TGA."""
    keys = ("protocol", "primary", "mean", "domains", "confusion_matrix", "episodes")
    result = {key: metric_result[key] for key in keys if key in metric_result}
    if "mean" in metric_result:
        result["tga"] = metric_result["mean"]
    return result


def _validate_publication(job: PublicationRequest, merge_existing: bool) -> dict[str, Any]:
    """Check completed results and safe destination components before any writes."""
    metadata = _config_metadata(job.config_path)
    for metric, result in job.result.items():
        if not isinstance(result, dict) or result.get("status") == "generated":
            raise ValueError(f"{metric} is not a completed scored result")
        json.dumps(result, allow_nan=False)
        if merge_existing and metric not in {
            "csvc", "voc", "cycle_voc", "voc_mem", "sa", "tga_easy", "tga_hard", "fpl", "trr", "vs"
        }:
            raise ValueError(
                f"Incremental merge is unsupported for {metric}; publish a complete run"
            )
        if metric == "csvc" and any(
            row.get("domains", ["id"]) != ["id"]
            for row in result["task_results"].values()
        ):
            raise ValueError("CSVC publication supports only ID results")
    _validate_components([job.baseline])
    _validate_components(list(job.result))
    for result in job.result.values():
        tasks = set(result.get("task_results", {}))
        for domain in result.get("domains", {}).values():
            if isinstance(domain, dict):
                tasks.update(domain.get("tasks", {}))
        _validate_components(list(tasks), task_names=True)
    return metadata


def _write_metric_artifacts(
    batch: PublicationBatch, job: PublicationRequest, metric_root: Path, published_baseline: str,
    adapter: str, checkpoint: Any, metric: str, metric_result: dict[str, Any],
    merged_result: dict[str, Any], contract: dict[str, Any] | None,
    previous_tasks: list[str], actual_tasks: tuple[str, ...],
    task_values: dict[str, dict[str, Any]],
) -> None:
    """Stage summary/task artifacts inside the existing publication transaction."""
    is_tga = metric in {"tga_easy", "tga_hard"}
    published_result = _published_tga_result(merged_result) if is_tga else merged_result
    identity = {
        "baseline": published_baseline,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "adapter": str(adapter),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "metric": metric,
        "source_run": job.run_name,
    }
    summary = {
        **identity,
        "tasks": list(actual_tasks),
        "result": published_result,
        "updated_at": now_iso(),
    }
    if contract is not None:
        summary["resume_contract"] = contract
    if is_tga and "mean" in metric_result:
        summary["tga"] = published_result.get("tga")
    batch.write_json(metric_root / "summary.json", summary)

    for task in set(previous_tasks) - set(actual_tasks):
        batch.delete(metric_root / f"{task}.json")
    for task, domains_for_task in task_values.items():
        payload = {
            **identity,
            "task": task,
            "domains": domains_for_task,
            "updated_at": now_iso(),
        }
        if is_tga:
            scores = [
                float(value)
                for value in domains_for_task.values()
                if isinstance(value, (int, float))
            ]
            if scores:
                payload["tga"] = sum(scores) / len(scores)
        batch.write_json(metric_root / f"{task}.json", payload)


def publish_result(
    job: PublicationRequest,
    final_root: Path,
    *,
    preserve_more_complete: bool = False,
    merge_existing: bool = False,
    batch: PublicationBatch | None = None,
) -> None:
    """Publish one validated job into the stable, overwrite-on-rerun tree."""
    if not isinstance(job.result, dict):
        raise ValueError(f"{job.run_name} has no validated result to publish")
    metadata = _validate_publication(job, merge_existing)
    if batch is None:
        with PublicationBatch(final_root) as transaction:
            publish_result(
                job, final_root, preserve_more_complete=preserve_more_complete,
                merge_existing=merge_existing, batch=transaction,
            )
        return
    if not batch._active or batch.root != final_root.resolve():
        raise ValueError("Publication requires an active batch for the same root")
    final_root = batch.root
    checkpoint = metadata.get("checkpoint")
    adapter = metadata["model"]
    # ``sync_result_to_final_results`` may supply an SIA judge-qualified
    # baseline name. Preserve that explicit identity when publishing instead
    # of recomputing the adapter-only name from the YAML metadata.
    published_baseline = job.baseline
    for metric, metric_result in job.result.items():
        hold_path = final_root / '.publication_holds.json'
        if hold_path.exists():
            held = json.loads(hold_path.read_text(encoding='utf-8'))
            if metric in held.get('metrics', []):
                print(
                    f'Publication held for {metric}: {held.get("reason", "campaign reset")}',
                    flush=True
                )
                continue
        metric_root = final_root / published_baseline / metric
        batch._target(metric_root)
        existing_summary = metric_root / "summary.json"
        old_summary = batch.read_json(existing_summary) if batch.exists(existing_summary) else {}
        contract = job.resume_contracts.get(metric)
        if job.execution_mode == "rerun" and old_summary:
            history = final_root / "rerun_history" / published_baseline / metric / job.run_name
            suffix = 1
            while batch.exists(history / "summary.json"):
                history = history.with_name(f"{job.run_name}-{suffix}")
                suffix += 1
            batch.write_json(history / "summary.json", old_summary)
            for previous_task in old_summary.get("tasks", []):
                old_task = metric_root / f"{previous_task}.json"
                if batch.exists(old_task):
                    batch.write_json(history / f"{previous_task}.json", batch.read_json(old_task))
        if old_summary and (
            old_summary.get("baseline") != published_baseline or old_summary.get("metric") != metric
        ):
            raise ValueError(
                f"Existing summary identity does not match destination: {existing_summary}"
            )
        previous_tasks = old_summary.get("tasks", [])
        _validate_components(previous_tasks, task_names=True)

        # TGA exposes several diagnostic aggregates in its raw result
        # (pairwise accuracy and counterfactual margins).  Those are useful in
        # ``results`` for debugging, but the stable final-results contract is
        # the actual TGA score: strict top-1 accuracy, overall and per task.
        # Keep only that primary score when publishing TGA.
        if metric == "csvc":
            metric_result = {**metric_result, "domains": csvc_domains(metric_result)}

        merged_result = metric_result
        merging = merge_existing and isinstance(old_summary.get("result"), dict)
        if merging and contract is not None:
            previous_contract = old_summary.get("resume_contract")
            if not isinstance(previous_contract, dict) or previous_contract.get("identity") != contract["identity"]:
                raise ValueError("Cannot merge results with incompatible resume contracts; rerun explicitly")
            for domain, tasks in contract["coverage"].items():
                for task, fingerprint in tasks.items():
                    previous = previous_contract.get("coverage", {}).get(domain, {}).get(task)
                    if previous is not None and previous != fingerprint:
                        raise ValueError(f"Cannot merge changed query/data inputs for {task}/{domain}")
            coverage = copy.deepcopy(previous_contract["coverage"])
            for domain, tasks in contract["coverage"].items():
                coverage.setdefault(domain, {}).update(tasks)
            contract = {**contract, "coverage": coverage}
        if merging:
            merged_result = _merge_metric_result(metric, old_summary["result"], metric_result)

        # Derive coverage from the result itself.  The scheduler task list is the
        # requested task set and can contain tasks that the metric runner
        # skipped (for example tasks without VOC frames).  Publishing that
        # requested set makes summaries claim coverage that was never measured.
        domains = merged_result.get("domains", {})
        task_values: dict[str, dict[str, Any]] = {}
        if isinstance(domains, dict):
            for domain, domain_result in domains.items():
                if not isinstance(domain_result, dict):
                    continue
                domain_tasks = domain_result.get("tasks", {})
                if not isinstance(domain_tasks, dict):
                    continue
                for task, value in domain_tasks.items():
                    task_values.setdefault(str(task), {})[str(domain)] = value
        actual_tasks = tuple(sorted(task_values))
        if merging:
            task_values = {task: task_values[task] for task in actual_tasks}
        if metric == "csvc" and isinstance(merged_result, dict):
            actual_tasks = tuple(sorted(merged_result.get("task_results", {})))
        if (
            preserve_more_complete
            and old_summary
            and isinstance(previous_tasks, list)
            and len(previous_tasks) > len(actual_tasks)
        ):
            continue
        _validate_components(list(actual_tasks), task_names=True)
        for task in task_values:
            destination = metric_root / f"{task}.json"
            if batch.exists(destination) and task not in previous_tasks:
                raise ValueError(f"Publication would overwrite an unowned file: {destination}")
        _write_metric_artifacts(
            batch, job, metric_root, published_baseline, adapter, checkpoint,
            metric, metric_result, merged_result, contract, previous_tasks, actual_tasks, task_values,
        )
