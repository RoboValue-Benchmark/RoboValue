from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthetic_data import make_dataset, write_json
from vmbmk.data.dataset import Dataset
from vmbmk.errors import VMBMKError
from vmbmk.metrics.tga import build_tga_queries, score_tga
from vmbmk.inference.queries import Result


def set_tga_task(
    data: Path,
    task_id: str,
    instruction: str,
    counterfactuals: list[dict[str, str]],
) -> None:
    path = data / task_id / "metadata.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["instruction"] = instruction
    row["supports"]["metrics"] = ["TGA"]
    row["tga"] = {"counterfactual_instructions": counterfactuals}
    write_json(path, row)


class TGATest(unittest.TestCase):

    def test_excludes_tasks_that_support_voc_mem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(
                data,
                "task_001",
                "press the button twice",
                [{"id": "01", "description": "press the button once"}],
            )
            path = data / "task_001" / "metadata.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["supports"]["metrics"].append("VOC-MEM")
            write_json(path, row)
            with self.assertRaisesRegex(
                VMBMKError, "excluding tasks that support VOC-MEM"
            ):
                build_tga_queries(Dataset.load(data), variant="hard")

    def test_excludes_cspc_diverse_but_keeps_normal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(
                Path(directory),
                [("normal", True), ("diverse", True)],
            )
            set_tga_task(
                data,
                "task_001",
                "put the object in the box",
                [{"id": "001", "description": "put the object beside the box"}],
            )
            for episode_id, solution_type in (
                ("normal", "normal"),
                ("diverse", "diverse"),
            ):
                path = data / "task_001" / "episodes" / episode_id / "metadata.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                row["cspc"] = {"solution_type": solution_type}
                write_json(path, row)

            queries = build_tga_queries(Dataset.load(data), variant="hard")
            self.assertEqual(
                {query.state_a.episode_id for query in queries},
                {"normal"},
            )

    def test_hard_uses_edge_two_percent_and_counterfactuals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            episode_metadata = (
                data / "task_001" / "episodes" / "success" / "metadata.json"
            )
            episode_row = json.loads(episode_metadata.read_text(encoding="utf-8"))
            episode_row["num_frames"] = 101
            write_json(episode_metadata, episode_row)
            set_tga_task(
                data,
                "task_001",
                "put the object in the box",
                [
                    {"id": "001", "description": "put the object beside the box"},
                    {"id": "002", "description": "leave the object untouched"},
                ],
            )
            dataset = Dataset.load(data)
            queries = build_tga_queries(dataset, variant="hard")
            self.assertEqual(len(queries), 9)
            self.assertEqual(
                sorted({(q.state_a.anchor_frame, q.state_b.anchor_frame) for q in queries}),
                [(0, 98), (1, 99), (2, 100)],
            )
            self.assertEqual(len({q.instruction for q in queries}), 3)

    def test_hard_normalizes_counterfactual_prefixed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(
                data,
                "task_001",
                "put the object in the box",
                [
                    {"id": "counterfactual_1", "description": "put it beside the box"},
                    {"id": "counterfactual_2", "description": "leave it untouched"},
                ],
            )
            queries = build_tga_queries(Dataset.load(data), variant="hard")
            ids = {query.query_id.split(":")[3] for query in queries}
            self.assertIn("counterfactual-001", ids)
            self.assertIn("counterfactual-002", ids)

    def test_hard_reports_pairwise_credit_and_strict_top1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(
                data,
                "task_001",
                "correct instruction",
                [
                    {"id": "001", "description": "lower scoring instruction"},
                    {"id": "002", "description": "tied instruction"},
                ],
            )
            dataset = Dataset.load(data)
            results = []
            for query in build_tga_queries(dataset, variant="hard"):
                if ":correct:" in query.query_id:
                    value = 0.8
                elif ":counterfactual-001:" in query.query_id:
                    value = 0.2
                else:
                    value = 0.8
                results.append(Result(query.query_id, "compare", value))
            scored = score_tga(dataset, results, variant="hard")
            self.assertEqual(scored["mean"], 0.0)
            self.assertEqual(scored["pairwise_mean"], 0.75)
            self.assertEqual(scored["primary"], "strict_top1_accuracy")
            self.assertEqual(scored["tie_credit"], 0.5)
            self.assertIn("confusion_matrix", scored)
            self.assertIn("task_001", scored["confusion_matrix"]["id"])

    def test_hard_rejects_unmapped_counterfactual_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(
                data,
                "task_001",
                "correct instruction",
                [{"id": "objects", "description": "invalid ID"}],
            )
            with self.assertRaisesRegex(VMBMKError, "unsupported counterfactual IDs"):
                build_tga_queries(Dataset.load(data), variant="hard")

    def test_hard_exposes_raw_scores_and_type_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(
                data,
                "task_001",
                "correct instruction",
                [
                    {"id": "001", "description": "objects negative"},
                    {"id": "002", "description": "actions negative"},
                    {"id": "003", "description": "placement negative"},
                    {"id": "004", "description": "constraints negative"},
                ],
            )
            dataset = Dataset.load(data)
            results = []
            values = {"001": 0.9, "002": 0.7, "003": 0.8, "004": 0.2}
            for query in build_tga_queries(dataset, variant="hard"):
                if ":correct:" in query.query_id:
                    value = 0.8
                else:
                    candidate_id = query.query_id.split(":counterfactual-", 1)[1].split(":", 1)[0]
                    value = values[candidate_id]
                results.append(Result(query.query_id, "compare", value))

            scored = score_tga(dataset, results, variant="hard")
            record = scored["episodes"][0]
            self.assertEqual(record["correct_score"], 0.8)
            self.assertEqual(record["counterfactual_scores"], values)
            self.assertEqual(
                record["counterfactual_types"],
                {
                    "001": "objects",
                    "002": "actions",
                    "003": "placement",
                    "004": "constraints",
                },
            )
            by_type = scored["counterfactual_by_type"]
            self.assertEqual(set(by_type), {"objects", "actions", "placement", "constraints"})
            self.assertAlmostEqual(by_type["objects"]["pairwise_accuracy"]["mean"], 0.0)
            self.assertAlmostEqual(by_type["actions"]["pairwise_accuracy"]["mean"], 1.0)
            self.assertAlmostEqual(by_type["placement"]["pairwise_accuracy"]["mean"], 0.5)
            self.assertAlmostEqual(by_type["constraints"]["pairwise_accuracy"]["mean"], 1.0)
            self.assertAlmostEqual(by_type["objects"]["mean_margin"]["mean"], -0.1)
            self.assertAlmostEqual(by_type["constraints"]["mean_margin"]["mean"], 0.6)

    def test_easy_swaps_instructions_between_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            first = data / "task_001"
            second = data / "task_002"
            second.mkdir()
            (second / "episodes").mkdir()
            source_episode = first / "episodes" / "success"
            target_episode = second / "episodes" / "success"
            import shutil

            shutil.copytree(source_episode, target_episode)
            episode_metadata = target_episode / "metadata.json"
            row = json.loads(episode_metadata.read_text(encoding="utf-8"))
            row["episode_id"] = "success"
            write_json(episode_metadata, row)
            write_json(
                second / "metadata.json",
                {
                    "task_id": "task_002",
                    "instruction": "second task instruction",
                    "supports": {"metrics": ["TGA"]},
                    "subtasks": [],
                    "tga": {"counterfactual_instructions": []},
                },
            )
            set_tga_task(data, "task_001", "first task instruction", [])
            dataset = Dataset.load(data)
            queries = build_tga_queries(dataset, variant="easy")
            self.assertEqual(len(queries), 8)
            by_task = {task_id: set() for task_id in ("task_001", "task_002")}
            for query in queries:
                by_task[query.state_a.task_id].add(query.instruction)
            self.assertEqual(
                by_task["task_001"],
                {"first task instruction", "second task instruction"},
            )
            self.assertEqual(
                by_task["task_002"],
                {"first task instruction", "second task instruction"},
            )

    def test_easy_requires_two_distinct_task_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            set_tga_task(data, "task_001", "only instruction", [])
            with self.assertRaisesRegex(VMBMKError, "no usable negative"):
                build_tga_queries(Dataset.load(data), variant="easy")

    def test_easy_can_score_target_with_full_candidate_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = make_dataset(Path(directory), [("success", True)])
            first = data / "task_001"
            second = data / "task_002"
            second.mkdir()
            (second / "episodes").mkdir()
            import shutil

            shutil.copytree(first / "episodes" / "success", second / "episodes" / "success")
            write_json(
                second / "metadata.json",
                {
                    "task_id": "task_002",
                    "instruction": "second instruction",
                    "supports": {"metrics": ["TGA"]},
                    "subtasks": [],
                },
            )
            set_tga_task(data, "task_001", "first instruction", [])
            dataset = Dataset.load(data)
            queries = build_tga_queries(
                dataset,
                ["task_002"],
                variant="easy",
                candidate_task_ids=["task_001", "task_002"],
            )
            self.assertEqual({q.state_a.task_id for q in queries}, {"task_002"})
            self.assertEqual(
                {q.instruction for q in queries},
                {"first instruction", "second instruction"},
            )
