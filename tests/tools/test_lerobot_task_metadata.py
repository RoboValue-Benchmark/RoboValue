from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare_lerobot_task_metadata",
                                           ROOT / "tools/prepare_lerobot_task_metadata.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class LeRobotTaskMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = json.loads((ROOT / "docs/data/real_world_tasks.json").read_text(encoding="utf-8"))["tasks"]
        self.definitions = {row["task_index"]: row["instruction"] for row in rows}
        self.indices = list(range(20))
        self.instructions = [TOOL.LEGACY_TASKS.get(index, f"unchanged {index}") for index in self.indices]

    def test_repetitions_and_sequence_preserve_indices_and_other_tasks(self) -> None:
        updated, replacements = TOOL.corrected_tasks(self.indices, self.instructions, self.definitions)
        self.assertEqual(len(updated), 20)
        self.assertIn("2-8-2-6-8", updated[1])
        for index in (5, 14, 15, 16):
            self.assertIn("5 times", updated[index])
        for index in set(self.indices) - set(self.definitions):
            self.assertEqual(updated[index], self.instructions[index])
        self.assertEqual(len(replacements), 20)

    def test_corrected_metadata_is_idempotent(self) -> None:
        updated, _ = TOOL.corrected_tasks(self.indices, self.instructions, self.definitions)
        repeated, _ = TOOL.corrected_tasks(self.indices, updated, self.definitions)
        self.assertEqual(updated, repeated)

    def test_corrected_task_table_still_updates_legacy_episode_labels(self) -> None:
        updated, _ = TOOL.corrected_tasks(self.indices, self.instructions, self.definitions)
        _, replacements = TOOL.corrected_tasks(self.indices, updated, self.definitions)
        for index, legacy in TOOL.LEGACY_TASKS.items():
            self.assertEqual(replacements[legacy], self.definitions[index])

    def test_unrecognized_or_missing_task_does_not_get_relabelled(self) -> None:
        self.instructions[1] = "Press different keys"
        with self.assertRaisesRegex(ValueError, "unexpected instruction"):
            TOOL.corrected_tasks(self.indices, self.instructions, self.definitions)
        with self.assertRaisesRegex(ValueError, "missing task indices"):
            TOOL.corrected_tasks([], [], self.definitions)

    def test_simulation_metadata_uses_current_task_contracts(self) -> None:
        rows = json.loads((ROOT / "docs/data/simulation_tasks.json").read_text(encoding="utf-8"))["tasks"]
        definitions = {row["task_index"]: row["instruction"] for row in rows}
        updated, _ = TOOL.corrected_tasks(list(range(15)),
            list(TOOL.SIM_LEGACY_TASKS.values()), definitions, legacy_tasks=TOOL.SIM_LEGACY_TASKS)
        self.assertNotIn("remaining miscellaneous", updated[6])
        self.assertIn("Move the box to the center", updated[7])
        self.assertNotIn("front sides facing left", updated[7])
        self.assertIn("dustpan upright", updated[14])

    def test_cosmetics_allows_upper_storage_before_or_after_drawer_closing(self) -> None:
        instruction = self.definitions[13]
        self.assertIn("before or after closing the drawer", instruction)
