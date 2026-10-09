from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vmbmk.data.dataset import _asset_path
from vmbmk.errors import DataValidationError


class AssetPathTest(unittest.TestCase):
    def setUp(self) -> None:
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name)
        self.episode = self.workspace / "data" / "task" / "episodes" / "ep001"
        self.episode.mkdir(parents=True)

    def test_episode_local_asset(self) -> None:
        asset = self.episode / "front.mp4"
        asset.touch()
        self.assertEqual(_asset_path(self.episode, "front.mp4", "video"), asset.resolve())

    def test_shared_raw_data_video_and_robot_assets(self) -> None:
        for filename in ("front.mp4", "robot.parquet"):
            with self.subTest(filename=filename):
                asset = self.workspace / "raw_data" / "collection" / filename
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.touch()
                relative = f"../../../../raw_data/collection/{filename}"
                self.assertEqual(_asset_path(self.episode, relative, "asset"), asset.resolve())

    def test_absolute_asset_is_rejected(self) -> None:
        asset = self.workspace / "front.mp4"
        asset.touch()
        with self.assertRaisesRegex(DataValidationError, "must be a relative path"):
            _asset_path(self.episode, str(asset.resolve()), "video")

    def test_missing_external_asset_is_not_substituted(self) -> None:
        (self.episode / "front.mp4").touch()
        with self.assertRaisesRegex(DataValidationError, "does not exist"):
            _asset_path(self.episode, "../../../../raw_data/front.mp4", "video")

    def test_directory_is_not_an_asset(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "does not exist"):
            _asset_path(self.episode, ".", "video")

    def test_symbolic_link_to_shared_storage(self) -> None:
        asset = self.workspace / "external.mp4"
        asset.touch()
        link = self.episode / "front.mp4"
        try:
            link.symlink_to(asset)
        except OSError as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        self.assertEqual(_asset_path(self.episode, "front.mp4", "video"), asset.resolve())
