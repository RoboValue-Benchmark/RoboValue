from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthetic_data import make_dataset
from vmbmk.adapters import create_adapter
from vmbmk.adapters.robodopamine import RoboDopamineAdapter
from vmbmk.data.dataset import Dataset
from vmbmk.inference.queries import CompareQuery, StateRef, SubtaskQuery, ValueQuery


class AdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.dataset = Dataset.load(make_dataset(Path(self.workspace.name)))

    def _adapter(self, name: str, **options):
        return create_adapter(
            {"adapter": name, "checkpoint": "checkpoint", "python": "python", **options},
            self.dataset,
        )

    def _assert_value_difference(self, adapter) -> None:
        before = StateRef("task_001", "ep_failure", 20)
        after = StateRef("task_001", "ep_failure", 70)
        query = CompareQuery("compare", before, after, "Move the cup")
        with patch.object(adapter, "value", return_value=[0.2, 0.9]) as value:
            self.assertAlmostEqual(adapter.compare([query])[0], 0.7)
        forwarded = value.call_args.args[0]
        self.assertEqual([item.state for item in forwarded], [before, after])
        self.assertEqual([item.instruction for item in forwarded], [query.instruction] * 2)

    def test_procvlm(self) -> None:
        self._assert_value_difference(self._adapter("procvlm"))

    def test_robometer(self) -> None:
        self._assert_value_difference(self._adapter("robometer"))

    def test_roboreward(self) -> None:
        self._assert_value_difference(self._adapter("roboreward"))

    def test_topreward(self) -> None:
        for backend in ("qwen", "molmo"):
            with self.subTest(backend=backend):
                adapter = self._adapter("topreward", backend=backend)
                self.assertEqual(adapter.backend, backend)
                self._assert_value_difference(adapter)
                if backend == "molmo":
                    with patch.object(adapter, "_load_molmo") as loading:
                        adapter._load()
                        loading.assert_called_once()
                    with patch.object(adapter, "_score_molmo_batch", return_value=[0.5]) as scoring:
                        self.assertEqual(adapter._score_batch([]), [0.5])
                        scoring.assert_called_once_with([])

    def test_robofac(self) -> None:
        self._assert_value_difference(self._adapter("robofac"))

    def test_liv(self) -> None:
        self._assert_value_difference(self._adapter("liv", source_root=self.workspace.name))

    def test_rynnvalue(self) -> None:
        adapter = self._adapter("rynnvalue")
        self._assert_value_difference(adapter)
        adapter.mode = "remaining_time"
        query = CompareQuery(
            "compare", StateRef("task_001", "ep_failure", 20),
            StateRef("task_001", "ep_failure", 70), "Move the cup",
        )
        with patch.object(adapter, "value", return_value=[9.0, 2.0]):
            self.assertEqual(adapter.compare([query]), [7.0])

    def test_vlac(self) -> None:
        adapter = self._adapter("vlac")
        query = CompareQuery(
            "compare", StateRef("task_001", "ep_failure", 20),
            StateRef("task_001", "ep_failure", 70), "Move the cup",
        )
        with patch.object(adapter, "_frame", side_effect=["before", "after"]), patch.object(
            adapter, "_native_trajectory", return_value=[-0.3]
        ) as trajectory:
            self.assertEqual(adapter.compare([query]), [-0.3])
        trajectory.assert_called_once_with("Move the cup", ["before", "after"], [], skip=1)

    def test_robodopamine(self) -> None:
        adapter = RoboDopamineAdapter(
            self.dataset, "checkpoint", incremental_hz=1.0, batch_size=4, seed=0,
        )
        reference = self.dataset.tasks["task_001"].episodes["ep_success"]
        query = ValueQuery("value", StateRef("task_001", "ep_failure", 20), "Move the cup")
        with patch.object(adapter, "_reference", return_value=(reference, [])), patch.object(
            adapter, "compare", return_value=[0.6, -0.2, 0.5, -0.2]
        ) as compare:
            self.assertAlmostEqual(adapter.value([query])[0], 0.6)
        comparisons = compare.call_args.args[0]
        self.assertEqual(
            [(item.state_a.episode_id, item.state_a.anchor_frame,
              item.state_b.episode_id, item.state_b.anchor_frame) for item in comparisons],
            [("ep_failure", 0, "ep_failure", 20),
             ("ep_success", 99, "ep_failure", 20),
             ("ep_failure", 0, "ep_failure", 10),
             ("ep_failure", 10, "ep_failure", 20)],
        )

    def test_failsafe(self) -> None:
        adapter = self._adapter("failsafe")
        query = SubtaskQuery("sia", StateRef("task_001", "ep_failure", 20), "Move the cup")
        with patch.object(adapter, "_responses", return_value=["Subtask: Pick up the cup. No."]):
            self.assertEqual(adapter.subtask([query]), ["Pick up the cup."])
