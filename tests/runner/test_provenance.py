from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from vmbmk.runner.config_identity import normalized_result_config
from vmbmk.runner.provenance import model_identity


class ProvenanceTest(unittest.TestCase):
    def test_normalization_is_idempotent_and_does_not_mutate_inputs(self) -> None:
        config = {
            "gpu": [0, 1], "batch_size": 2,
            "model_options": {"sia_workers": 4, "sia_timeout": 60, "view": "front"},
            "metrics": {"sa": {"mode": "base", "domains": ["id"]}},
        }
        original = copy.deepcopy(config)
        normalized = normalized_result_config(config)
        self.assertEqual(config, original)
        self.assertEqual(normalized_result_config(normalized), normalized)
        self.assertEqual(normalized["model_options"], {"view": "front"})
        normalized["metrics"]["sa"]["domains"].append("env")
        self.assertEqual(config, original)

    def test_reference_assets_replacement_changes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "reference.mp4"
            video.write_bytes(b"reference")
            config = {"adapter": "mock_service", "reference_data": str(root)}
            previous = model_identity(config)
            video.write_bytes(b"changed reference frames")
            self.assertNotEqual(model_identity(config), previous)

    def test_implicit_one_shot_references_use_adapter_contract_and_zero_shot_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "reference/task_001/front.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"reference")
            configs = [
                {"adapter": "robodopamine"},
                {"adapter": "vlac", "shot_mode": "one_shot"},
                {"adapter": "vlac"},
            ]
            previous = [model_identity(config, data_root=root) for config in configs]
            video.write_bytes(b"changed reference frames")
            current = [model_identity(config, data_root=root) for config in configs]
            self.assertNotEqual(current[0], previous[0])
            self.assertNotEqual(current[1], previous[1])
            self.assertEqual(current[2], previous[2])

    def test_checkpoint_mapping_order_and_execution_controls_preserve_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.bin", root / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            config = {"adapter": "mock_service", "batch_size": 2, "gpu": 0,
                      "checkpoint_by_task": {"task_001": str(first), "task_002": str(second)}}
            reordered = {**config, "batch_size": 8, "gpu": [1, 2],
                         "checkpoint_by_task": {"task_002": str(second), "task_001": str(first)}}
            self.assertEqual(model_identity(config), model_identity(reordered))
            previous = model_identity(config)
            second.write_bytes(b"replaced checkpoint")
            self.assertNotEqual(model_identity(config), previous)


if __name__ == "__main__":
    unittest.main()
