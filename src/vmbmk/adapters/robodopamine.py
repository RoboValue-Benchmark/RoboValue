from __future__ import annotations

import random
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vmbmk.data.dataset import Dataset, Episode
from vmbmk.errors import ConfigurationError, VMBMKError
from vmbmk.data.playback import PlaybackView
from vmbmk.inference.queries import CompareQuery, StateRef, ValueQuery
from .base import Adapter, positive_int, positive_number, require_config
from .video_inputs import query_view, read_frame, task_reference_clip


_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = “just beginning”; REFERENCE END = “fully completed.”
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


def _incremental_frames(
    anchor: int, fps: float, incremental_hz: float
) -> tuple[int, ...]:
    step = max(1, int(round(fps / incremental_hz)))
    frames = list(range(0, anchor + 1, step))
    if frames[-1] != anchor:
        frames.append(anchor)
    return tuple(frames)


def _incremental_progress(hops: Sequence[float]) -> float:
    if not hops:
        return 0.0
    progress = float(hops[0])
    for hop in hops[1:]:
        if hop >= 0:
            progress = progress + (1.0 - progress) * hop
        else:
            progress = progress + progress * hop
    return progress


@dataclass(frozen=True)
class _ProgressLayout:
    forward: int
    backward: int
    hop_start: int
    hop_end: int


class RoboDopamineAdapter(Adapter):
    default_shot_mode = "one_shot"
    default_reference_view = "front"

    _score = re.compile(r"<score>\s*([+-]?\d{1,3}(?:\.\d+)?)%\s*</score>")

    def __init__(
        self,
        dataset: Dataset,
        checkpoint: str,
        *,
        incremental_hz: float,
        batch_size: int,
        seed: int,
        enforce_eager: bool = False,
        reference_data: Path | None = None,
        reference_view: str = default_reference_view,
    ) -> None:
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.incremental_hz = incremental_hz
        self.batch_size = batch_size
        self.seed = seed
        self.enforce_eager = enforce_eager
        self.reference_data = (
            reference_data
            if reference_data is not None or dataset is None
            else dataset.root / "reference"
        )
        self.reference_view = reference_view
        self._model = None
        self._references: dict[str, tuple[Episode, list[Any]]] = {}
        self._cache: dict[tuple[Any, ...], float] = {}

    def close(self) -> None:
        """Release the vLLM engine and its worker processes."""
        model = self._model
        self._model = None
        if model is None:
            return
        engine = getattr(model, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del model

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], dataset: Dataset
    ) -> "RoboDopamineAdapter":
        require_config(
            config,
            "robodopamine",
            {
                "incremental_hz",
                "batch_size",
                "seed",
                "shot_mode",
                "reference_data",
                "reference_view",
                "ref_num",
                "enforce_eager",
            },
        )
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("robodopamine.seed must be an integer")
        shot_mode = str(config.get("shot_mode", cls.default_shot_mode))
        if shot_mode != "one_shot":
            raise ConfigurationError(
                "robodopamine.shot_mode must be 'one_shot'"
            )
        raw_reference_data = config.get("reference_data")
        if raw_reference_data is None:
            reference_data = (dataset.root / "reference").resolve()
        elif not isinstance(raw_reference_data, str) or not raw_reference_data:
            raise ConfigurationError(
                "robodopamine.reference_data must be a non-empty path"
            )
        else:
            reference_data = Path(raw_reference_data).resolve()
        if not reference_data.is_dir():
            raise ConfigurationError(
                f"robodopamine.reference_data does not exist: {reference_data}"
            )
        reference_view = str(config.get("reference_view", cls.default_reference_view))
        if reference_view != "front":
            raise ConfigurationError(
                "robodopamine.reference_view must be 'front'"
            )
        checkpoint = str(config["checkpoint"])
        if not checkpoint:
            raise ConfigurationError("robodopamine.checkpoint must be non-empty")
        enforce_eager = config.get("enforce_eager", False)
        if not isinstance(enforce_eager, bool):
            raise ConfigurationError("robodopamine.enforce_eager must be a boolean")
        return cls(
            dataset,
            checkpoint,
            incremental_hz=positive_number(
                config.get("incremental_hz", 3.0),
                "robodopamine.incremental_hz",
            ),
            batch_size=positive_int(
                config.get("batch_size", 4), "robodopamine.batch_size"
            ),
            seed=seed,
            enforce_eager=enforce_eager,
            reference_data=reference_data,
            reference_view=reference_view,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        # Robo-Dopamine is evaluated from an on-disk checkpoint. Keep model
        # loading deterministic and safe on machines without network access;
        # the runner also exports these variables so custom model code sees
        # the same offline policy.
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        import numpy as np
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._processor.image_processor.max_pixels = 76800
        self._processor.image_processor.min_pixels = 12544
        llm_options = {
            "model": self.checkpoint,
            "max_model_len": 8192,
            "limit_mm_per_prompt": {"image": 8},
            "enable_prefix_caching": True,
            # CUDA graphs trigger an illegal-memory-access failure for some
            # Robo-Dopamine checkpoints on H200.  This is a runtime-only
            # stability switch; frame sampling, prompts, and scoring remain
            # unchanged.
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }
        self._model = LLM(
            **llm_options,
        )
        self._sampling = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
            seed=self.seed,
        )

    def _episode(self, state: StateRef) -> Episode:
        if state.episode_id.startswith("external:"):
            episode, _ = self._reference(state.task_id)
            if episode.episode_id != state.episode_id:
                raise VMBMKError(
                    f"unknown external reference episode {state.task_id}/{state.episode_id}"
                )
            if not 0 <= state.anchor_frame < episode.num_frames:
                raise VMBMKError(
                    f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                    f"is outside [0, {episode.num_frames})"
                )
            return episode
        episode = self.dataset.episode(state.task_id, state.episode_id)
        if not 0 <= state.anchor_frame < episode.num_frames:
            raise VMBMKError(
                f"{state.task_id}/{state.episode_id} frame {state.anchor_frame} "
                f"is outside [0, {episode.num_frames})"
            )
        for view in ("front", "wrist_left", "wrist_right"):
            episode.video(view)
        return episode

    def _reference(self, task_id: str) -> tuple[Episode, list[Any]]:
        if task_id not in self._references:
            if self.reference_data is None:
                raise ConfigurationError(
                    "RoboDopamine one-shot reference_data is not configured"
                )
            video, frame_count = task_reference_clip(
                self.reference_data, task_id, self.reference_view
            )
            frames = [
                read_frame(video, 0),
                read_frame(video, frame_count - 1),
            ]
            episode = Episode(
                task_id=task_id,
                episode_id=f"external:{video}",
                root=video.parent,
                success=True,
                domain="id",
                fps=1.0,
                num_frames=frame_count,
                videos={"front": video},
                voc_frames=(),
                voc_mem_frames=(),
            )
            self._references[task_id] = (episode, frames)
        return self._references[task_id]

    @lru_cache(maxsize=64)
    def _state_frames(
        self, task_id: str, episode_id: str, anchor_frame: int
    ) -> tuple[Any, ...]:
        if episode_id.startswith("external:"):
            episode = self._episode(StateRef(task_id, episode_id, anchor_frame))
            frame = read_frame(episode.video("front"), anchor_frame)
            return (frame, frame, frame)
        episode = self.dataset.episode(task_id, episode_id)
        return tuple(
            read_frame(episode.video(view), anchor_frame)
            for view in ("front", "wrist_left", "wrist_right")
        )

    def _request(self, images: Sequence[Any], instruction: str) -> dict[str, Any]:
        self._load()
        parts = _PROMPT.format(task=instruction).split("<image>")
        content = []
        for index, text in enumerate(parts):
            if text:
                content.append({"type": "text", "text": text})
            if index < len(images):
                content.append({"type": "image"})
        prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "multi_modal_data": {"image": list(images)}}

    def _key(
        self, query: CompareQuery, reference_episode_id: str
    ) -> tuple[Any, ...]:
        return (
            query.state_a.task_id,
            query.state_a.episode_id,
            query.state_a.anchor_frame,
            query.state_b.task_id,
            query.state_b.episode_id,
            query.state_b.anchor_frame,
            query.instruction,
            reference_episode_id,
            self.seed,
        )

    def _parse(self, text: str) -> float:
        match = self._score.fullmatch(text.strip())
        if not match:
            raise VMBMKError(f"cannot parse Robo-Dopamine output {text!r}")
        value = float(match.group(1))
        if not -100 <= value <= 100:
            raise VMBMKError(f"Robo-Dopamine score is outside [-100, 100]: {value}")
        return value / 100.0

    def compare(self, queries: Sequence[CompareQuery]) -> list[float]:
        if not queries:
            return []
        pending: dict[
            tuple[Any, ...], tuple[CompareQuery, Episode, Episode, list[Any]]
        ] = {}
        owned: list[tuple[Any, ...]] = []
        for query in queries:
            if query.state_a.task_id != query.state_b.task_id:
                raise VMBMKError(
                    f"{query.query_id}: compare states must belong to the same task"
                )
            before = self._episode(query.state_a)
            after = self._episode(query.state_b)
            reference, reference_frames = self._reference(query.state_a.task_id)
            key = self._key(query, reference.episode_id)
            if query.state_a == query.state_b:
                self._cache[key] = 0.0
            elif key not in self._cache:
                pending[key] = (query, before, after, reference_frames)
            owned.append(key)
        items = list(pending.items())
        if items:
            self._load()
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        for batch_number, start in enumerate(
            range(0, len(items), self.batch_size), start=1
        ):
            chunk = items[start : start + self.batch_size]
            requests = []
            for _, (query, before, after, reference_frames) in chunk:
                before_frames = self._state_frames(
                    before.task_id, before.episode_id, query.state_a.anchor_frame
                )
                after_frames = self._state_frames(
                    after.task_id, after.episode_id, query.state_b.anchor_frame
                )
                requests.append(
                    self._request(
                        [*reference_frames, *before_frames, *after_frames],
                        query.instruction,
                    )
                )
            responses = self._model.generate(
                requests,
                sampling_params=self._sampling,
                use_tqdm=False,
            )
            if len(responses) != len(chunk):
                raise VMBMKError(
                    "Robo-Dopamine returned the wrong number of responses"
                )
            for (key, _), response in zip(chunk, responses):
                self._cache[key] = self._parse(response.outputs[0].text)
            if batch_number % 50 == 0 or batch_number == total_batches:
                print(
                    f"Robo-Dopamine batches: {batch_number}/{total_batches}",
                    flush=True,
                )
        return [self._cache[key] for key in owned]

    def value(self, queries: Sequence[ValueQuery]) -> list[float]:
        comparisons: list[CompareQuery] = []
        layouts: list[_ProgressLayout] = []
        for query in queries:
            episode = self._episode(query.state)
            view: PlaybackView = query_view(self.dataset, query, "front")
            logical_anchor = view.timeline_anchor(query.state.anchor_frame)
            start = StateRef(
                query.state.task_id,
                query.state.episode_id,
                view.source_index(0),
            )
            reference, _ = self._reference(query.state.task_id)
            goal = StateRef(
                query.state.task_id,
                reference.episode_id,
                reference.num_frames - 1,
            )
            forward_index = len(comparisons)
            comparisons.append(
                CompareQuery(
                    f"{query.query_id}:forward",
                    start,
                    query.state,
                    query.instruction,
                )
            )
            backward_index = len(comparisons)
            comparisons.append(
                CompareQuery(
                    f"{query.query_id}:backward",
                    goal,
                    query.state,
                    query.instruction,
                )
            )
            hop_start = len(comparisons)
            frames = _incremental_frames(
                logical_anchor,
                episode.fps,
                self.incremental_hz,
            )
            for left, right in zip(frames, frames[1:]):
                source_left = view.source_index(left)
                source_right = view.source_index(right)
                comparisons.append(
                    CompareQuery(
                        f"{query.query_id}:incremental:{source_left}:{source_right}",
                        StateRef(
                            query.state.task_id,
                            query.state.episode_id,
                            source_left,
                        ),
                        StateRef(
                            query.state.task_id,
                            query.state.episode_id,
                            source_right,
                        ),
                        query.instruction,
                    )
                )
            layouts.append(
                _ProgressLayout(forward_index, backward_index, hop_start, len(comparisons))
            )
        scores = self.compare(comparisons)
        values = []
        for layout in layouts:
            forward_progress = scores[layout.forward]
            backward_progress = 1.0 + scores[layout.backward]
            incremental_progress = _incremental_progress(scores[layout.hop_start:layout.hop_end])
            values.append(
                (forward_progress + backward_progress + incremental_progress) / 3.0
            )
        return values
