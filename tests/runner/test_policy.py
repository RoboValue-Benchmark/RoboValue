from __future__ import annotations

import unittest

from vmbmk.runner.policy import evaluation_policy


class EvaluationPolicyTest(unittest.TestCase):
    def test_roboreward_allows_voc_metrics_for_both_sizes(self) -> None:
        for size in (4, 8):
            config = {"model": "roboreward", "checkpoint": f"RoboReward-{size}B"}
            for metric in ("voc", "voc_mem", "cycle_voc"):
                with self.subTest(size=size, metric=metric):
                    self.assertIsNone(
                        evaluation_policy(config).reason(config, metric, "task_001")
                    )

    def test_roboreward_excludes_only_unreported_consistency_metrics(self) -> None:
        for size in (4, 8):
            config = {"model": "roboreward", "checkpoint": f"RoboReward-{size}B"}
            for metric in ("vs", "csvc"):
                with self.subTest(size=size, metric=metric):
                    self.assertIsNotNone(
                        evaluation_policy(config).reason(config, metric, "task_001")
                    )
            for metric in ("sa", "tga_easy", "tga_hard", "fpl", "trr"):
                with self.subTest(size=size, metric=metric):
                    self.assertIsNone(
                        evaluation_policy(config).reason(config, metric, "task_001")
                    )

    def test_procvlm_allows_memory_tasks_with_matching_checkpoints(self) -> None:
        config = {
            "model": "procvlm",
            "model_options": {
                "use_lora": True,
                "checkpoint_by_task": {
                    "press_by_number": "/checkpoints/press_by_number",
                    "swap_blocks": "/checkpoints/swap_blocks",
                },
            },
        }
        for task in ("press_by_number", "swap_blocks"):
            for metric in ("voc", "voc_mem"):
                with self.subTest(task=task, metric=metric):
                    self.assertIsNone(evaluation_policy(config).reason(config, metric, task))

    def test_procvlm_still_requires_matching_task_checkpoint(self) -> None:
        config = {"model": "procvlm", "model_options": {"use_lora": True}}
        for task in ("press_by_number", "swap_blocks", "task_001"):
            with self.subTest(task=task):
                self.assertEqual(
                    evaluation_policy(config).reason(config, "voc_mem", task),
                    f"no task checkpoint configured for {task}",
                )

    def test_other_recorded_constraints_are_unchanged(self) -> None:
        failsafe = {"model": "failsafe"}
        self.assertIsNone(evaluation_policy(failsafe).reason(failsafe, "sia", "task_001"))
        self.assertIsNotNone(evaluation_policy(failsafe).reason(failsafe, "voc", "task_001"))
        topreward = {"model": "topreward", "checkpoint": "Qwen3-VL-32B-Instruct"}
        self.assertEqual(evaluation_policy(topreward).batch_override(("tga_easy",)), 2)
        self.assertIsNone(evaluation_policy(topreward).batch_override(("fpl",)))
