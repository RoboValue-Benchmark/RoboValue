from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthetic_data import make_dataset, set_task_metadata, write_json
from test_trr import make_trr_dataset
from test_voc_mem import make_voc_mem_dataset
from vmbmk.data.dataset import Dataset
from vmbmk.inference.queries import Result, query_state
from vmbmk.metrics.registry import METRICS, build_metric_queries, metric_runner
from vmbmk.runner.run import RunConfig
from vmbmk.tools.results.scoring_validation import validate_metric_scores


class ScoreValidationTest(unittest.TestCase):
    def dataset(self, root: Path, metric: str) -> Path:
        if metric == "trr":
            return make_trr_dataset(root)
        if metric == "voc_mem":
            return make_voc_mem_dataset(root)
        if metric.startswith("tga_"):
            data = make_dataset(root)
            task = set_task_metadata(data, ["TGA"])
            metadata = json.loads((task / "metadata.json").read_text())
            metadata["tga"] = {"counterfactual_instructions": [{"id": "001", "description": "wrong instruction"}]}
            write_json(task / "metadata.json", metadata)
            second = data / "task_002"
            shutil.copytree(task, second)
            metadata.update(task_id="task_002", instruction="another task instruction")
            write_json(second / "metadata.json", metadata)
            return data
        if metric == "fpl":
            data = make_dataset(root, [("failure", False)])
            task = set_task_metadata(data, ["FPL"])
            write_json(task / "episodes/failure/annotation.json", {"fpl": {"failure_start": 40, "failure_end": 50}})
            return data
        if metric == "csvc":
            data = make_dataset(root, [("normal", True), ("diverse", True)])
            task = set_task_metadata(data, ["CSVC"], [{"id": "001", "description": "Move"}])
            for episode in ("normal", "diverse"):
                episode_root = task / "episodes" / episode
                metadata = json.loads((episode_root / "metadata.json").read_text())
                metadata["cspc"] = {"solution_type": episode}
                write_json(episode_root / "metadata.json", metadata)
                write_json(episode_root / "annotation.json", {"subtask_segments": [
                    {"id": "001", "start_frame": 10, "end_frame_exclusive": 60},
                ]})
            return data
        data = make_dataset(root)
        set_task_metadata(data, ["SA"])
        return data

    def test_pure_validation_matches_original_runners_and_rejects_changed_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for metric in ("sa", "tga_easy", "tga_hard", "voc_mem", "fpl", "trr", "csvc"):
                with self.subTest(metric=metric):
                    data = self.dataset(root / metric, metric)
                    source = root / f"{metric}.json"
                    tasks = ["task_001", "task_002"] if metric.startswith("tga_") else ["task_001"]
                    write_json(source, {"model": "mock_service", "python": sys.executable,
                                        "batch_size": 2, "data": str(data), "output": str(root / "runs"),
                                        "tasks": tasks, "metrics": {metric: {"mode": "base", "domains": ["id"]}}})
                    config = RunConfig.load(source)
                    dataset = Dataset.load(data, metrics=[metric])
                    queries = build_metric_queries(metric, dataset, tasks, ["id"])
                    results = [Result(query.query_id, query.op, query_state(query).anchor_frame % 7 / 6) for query in queries]
                    target = f"vmbmk.metrics.{METRICS[metric].module}.run_queries"
                    with patch(target, return_value=results), patch("vmbmk.inference.dispatch.subprocess.run", side_effect=AssertionError("must not launch inference")):
                        recorded = metric_runner(metric)(data, config.model_config(), tasks, root / "unused.jsonl", None,
                                                         mode="base", domains=["id"], dataset=dataset)
                        recorded = json.loads(json.dumps(recorded))
                        validate_metric_scores(config, dataset, {metric: recorded}, results)
                        recorded["mean"] += 0.125
                        with self.assertRaisesRegex(ValueError, "saved predictions"):
                            validate_metric_scores(config, dataset, {metric: recorded}, results)


if __name__ == "__main__":
    unittest.main()
