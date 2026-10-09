from __future__ import annotations

from types import ModuleType
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, positive_int, require_config
from .video_inputs import query_view, read_frame


def _load_module(name: str, path: Path, *, package: bool = False) -> ModuleType:
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=locations,
    )
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be an object")
    return value


class LIVAdapter(Adapter):
    """Official LIV-EPIC RN50 image/text cosine reward."""

    _ASSET_FILES = ("model.pt", "config.yaml", "RN50.pt")

    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str | Path,
        *,
        source_root: str | Path,
        batch_size: int = 32,
        view: str = "front",
        instruction_overrides: Mapping[str, str] | None = None,
        truncate_text: bool = False,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve()
        self.batch_size = batch_size
        self.view = view
        self.instruction_overrides = dict(instruction_overrides or {})
        self.truncate_text = truncate_text
        self._model = None
        self._clip = None
        self._frame_cache: dict[tuple[str, int], Any] = {}
        self._text_cache: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "LIVAdapter":
        require_config(
            config,
            "liv",
            {"batch_size", "source_root", "view", "instruction_overrides", "truncate_text"},
        )
        checkpoint = config["checkpoint"]
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ConfigurationError("liv.checkpoint must be a non-empty string")
        source_root = config.get("source_root")
        if not isinstance(source_root, str) or not source_root.strip():
            raise ConfigurationError("liv.source_root must be a non-empty string")
        view = config.get("view", "front")
        if not isinstance(view, str) or not view.strip():
            raise ConfigurationError("liv.view must be a non-empty string")
        raw_overrides = config.get("instruction_overrides", {})
        if not isinstance(raw_overrides, Mapping):
            raise ConfigurationError("liv.instruction_overrides must be an object")
        instruction_overrides: dict[str, str] = {}
        for source, target in raw_overrides.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ConfigurationError(
                    "liv.instruction_overrides keys and values must be strings"
                )
            instruction_overrides[source] = target
        truncate_text = config.get("truncate_text", False)
        if not isinstance(truncate_text, bool):
            raise ConfigurationError("liv.truncate_text must be a boolean")
        return cls(
            dataset,
            checkpoint,
            source_root=source_root,
            batch_size=positive_int(config.get("batch_size", 32), "liv.batch_size"),
            view=view.strip(),
            instruction_overrides=instruction_overrides,
            truncate_text=truncate_text,
        )

    def _validate_files(self) -> tuple[Path, Path, Path, Path]:
        if not self.checkpoint.is_dir():
            raise ConfigurationError(
                f"LIV checkpoint directory does not exist: {self.checkpoint}"
            )
        files = tuple(self.checkpoint / name for name in self._ASSET_FILES)
        for path in files:
            if not path.is_file():
                raise ConfigurationError(f"missing LIV asset: {path}")
        model_source = self.source_root / "liv" / "models" / "model_liv.py"
        clip_package = (
            self.source_root / "liv" / "models" / "clip" / "clip" / "__init__.py"
        )
        if not model_source.is_file():
            raise ConfigurationError(f"missing LIV source: {model_source}")
        if not clip_package.is_file():
            raise ConfigurationError(f"missing vendored CLIP source: {clip_package}")
        return files[0], files[1], files[2], clip_package

    @staticmethod
    def _model_options(path: Path) -> dict[str, Any]:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"invalid LIV config {path}: {exc}") from exc
        agent = _mapping(_mapping(raw, str(path)).get("agent"), f"{path}.agent")
        expected_target = agent.get("_target_")
        if expected_target not in {"liv.LIV", "liv.models.model_liv.LIV"}:
            raise ConfigurationError(
                f"{path}.agent._target_ must identify the official LIV class"
            )
        if agent.get("modelid") != "RN50":
            raise ConfigurationError(f"{path}.agent.modelid must be 'RN50'")
        if agent.get("metric", "cos") != "cos":
            raise ConfigurationError(f"{path}.agent.metric must be 'cos'")
        allowed = {
            "lr",
            "weight_decay",
            "visionweight",
            "langweight",
            "clipweight",
            "gamma",
            "metric",
            "num_negatives",
            "grad_text",
            "scratch",
        }
        return {name: value for name, value in agent.items() if name in allowed}

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        model_path, config_path, rn50_path, clip_package = self._validate_files()
        loaded_clip = sys.modules.get("clip")
        if loaded_clip is not None:
            loaded_path = Path(getattr(loaded_clip, "__file__", "")).resolve()
            if loaded_path != clip_package.resolve():
                raise ConfigurationError(
                    f"a different clip package is already loaded: {loaded_path}"
                )
            clip_module = loaded_clip
        else:
            clip_module = _load_module("clip", clip_package, package=True)
        model_module = _load_module(
            "_vmbmk_liv_model",
            self.source_root / "liv" / "models" / "model_liv.py",
        )
        options = self._model_options(config_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model_module.LIV(
            modelid=str(rn50_path),
            device=str(device),
            **options,
        )
        payload = torch.load(model_path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("liv"), dict):
            raise ConfigurationError("LIV model.pt must contain a 'liv' state dict")
        state = payload["liv"]
        if state and all(name.startswith("module.") for name in state):
            state = {name.removeprefix("module."): value for name, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        self._torch = torch
        self._device = device
        self._clip = clip_module
        self._model = model

    def _episode(self, state: StateRef) -> Episode:
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        episode.video(self.view)
        return episode

    def _ensure_frame_embeddings(
        self, items: Sequence[tuple[PlaybackView, int]]
    ) -> None:
        keys = list(
            dict.fromkeys(
                (str(view.path), view.source_index(frame)) for view, frame in items
            )
        )
        missing = [key for key in keys if key not in self._frame_cache]
        if not missing:
            return
        self._load()
        import numpy as np

        decoded: list[tuple[tuple[str, int], Any]] = []
        for path, frame in missing:
            image = read_frame(path, frame).convert("RGB")
            array = np.asarray(image, dtype=np.float32).copy()
            tensor = self._torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
            decoded.append(((path, frame), tensor))
        by_shape: dict[tuple[int, ...], list[tuple[tuple[str, int], Any]]] = {}
        for item in decoded:
            by_shape.setdefault(tuple(item[1].shape), []).append(item)
        for shape_items in by_shape.values():
            for start in range(0, len(shape_items), self.batch_size):
                batch = shape_items[start : start + self.batch_size]
                keys, tensors = zip(*batch)
                images = self._torch.stack(tensors).to(self._device)
                with self._torch.inference_mode():
                    embeddings = self._model(
                        input=images,
                        modality="vision",
                    ).detach()
                if embeddings.ndim != 2 or embeddings.shape[0] != len(keys):
                    raise VMBMKError(
                        "LIV returned an unexpected frame embedding shape: "
                        f"{tuple(embeddings.shape)}"
                    )
                if not self._torch.isfinite(embeddings).all():
                    raise VMBMKError("LIV returned non-finite frame embeddings")
                for key, embedding in zip(keys, embeddings.cpu()):
                    self._frame_cache[key] = embedding

    def _ensure_text_embeddings(self, instructions: Sequence[str]) -> None:
        missing = list(
            dict.fromkeys(
                instruction
                for instruction in instructions
                if instruction not in self._text_cache
            )
        )
        if not missing:
            return
        self._load()
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            tokens = self._clip.tokenize(
                chunk, truncate=self.truncate_text
            ).to(self._device)
            with self._torch.inference_mode():
                embeddings = self._model(
                    input=tokens,
                    modality="text",
                ).detach()
            if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk):
                raise VMBMKError(
                    "LIV returned an unexpected text embedding shape: "
                    f"{tuple(embeddings.shape)}"
                )
            if not self._torch.isfinite(embeddings).all():
                raise VMBMKError("LIV returned non-finite text embeddings")
            for instruction, embedding in zip(chunk, embeddings.cpu()):
                self._text_cache[instruction] = embedding

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        if not queries:
            return []
        episodes = [self._episode(query.state) for query in queries]
        views = [query_view(self.dataset, query, self.view) for query in queries]
        self._ensure_frame_embeddings(
            [
                (view, view.timeline_anchor(query.state.anchor_frame))
                for view, query in zip(views, queries)
            ]
        )
        instructions = [
            self.instruction_overrides.get(query.instruction, query.instruction)
            for query in queries
        ]
        self._ensure_text_embeddings(instructions)
        results: list[float] = []
        for start in range(0, len(queries), self.batch_size):
            chunk_queries = queries[start : start + self.batch_size]
            chunk_views = views[start : start + self.batch_size]
            chunk_instructions = instructions[start : start + self.batch_size]
            frames = self._torch.stack([
                self._frame_cache[
                    (
                        str(view.path),
                        view.source_index(
                            view.timeline_anchor(query.state.anchor_frame)
                        ),
                    )
                ]
                for view, query in zip(chunk_views, chunk_queries)
            ]).to(self._device)
            texts = self._torch.stack([
                self._text_cache[instruction]
                for instruction in chunk_instructions
            ]).to(self._device)
            with self._torch.inference_mode():
                scores = self._model.sim(frames, texts).detach().float()
            if scores.ndim != 1 or scores.shape[0] != len(chunk_queries):
                raise VMBMKError(
                    f"LIV returned {tuple(scores.shape)} scores for "
                    f"batch size {len(chunk_queries)}"
                )
            values = [float(score) for score in scores.cpu()]
            if any(not math.isfinite(value) for value in values):
                raise VMBMKError("LIV returned non-finite similarity values")
            results.extend(max(-1.0, min(1.0, value)) for value in values)
        return results

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        values: list[ValueQuery] = []
        for query in queries:
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: compare states must belong to the same task"
                )
            values.extend([
                ValueQuery(f"{query.query_id}:a", query.state_a, query.instruction),
                ValueQuery(f"{query.query_id}:b", query.state_b, query.instruction),
            ])
        scores = self.value(values)
        return [scores[index + 1] - scores[index] for index in range(0, len(scores), 2)]
