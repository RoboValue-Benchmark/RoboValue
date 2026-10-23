"""Prepare corrected LeRobot task and episode metadata without copying data assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


LEGACY_TASKS = {
    0: "Put colored blocks in the bowl and pour",
    1: "Press the numeric keypad",
    2: "Set the tableware",
    3: "Place seasoning jars",
    4: "Place plastic plates",
    5: "Press the blue button",
    6: "Place the small pot",
    7: "Organize the desk (workbench)",
    8: "Organize books",
    9: "Home tea setting",
    10: "Pull open and close drawers 2",
    11: "Sort colored blocks into the box",
    12: "Fill pen holder 4",
    13: "Organize colorful cosmetic storage box 2",
    14: "Beat the drum 1",
    15: "Apply dish soap 1",
    16: "Roast meat skewers 1",
    17: "Logistics packaging",
    18: "Floral arrangement and flameless aromatherapy placement",
    19: "Prepare breakfast",
}


SIM_LEGACY_TASKS = {
    0: "Arrange the numbers from left to right to form the largest possible number, and place them on the pad.",
    1: "Hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.",
    2: "Fold the clothes neatly.",
    3: "Hang all the mugs on the mug rack.",
    4: "Insert the three tubes into the rack one by one.",
    5: "Pick up two slices of bread, place them into the toaster, and press the lever down.",
    6: "Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside.",
    7: "Place all the objects into the box with their front sides facing left.",
    8: "Place all stacking toy pieces onto the correct pegs.",
    9: "Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm.",
    10: "Pick up the bottles and throw them into the dustbin, using handover when needed.",
    11: "Stack the three blocks with different textures.",
    12: "Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.",
    13: "Swap the two blocks using the empty mat, pressing the button after each move.",
    14: "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.",
}


def corrected_tasks(
    indices: Sequence[int], instructions: Sequence[str], definitions: dict[int, str],
    *, legacy_tasks: dict[int, str] = LEGACY_TASKS,
) -> tuple[list[str], dict[str, str]]:
    """Keep task indices fixed while replacing collection labels with instructions."""
    if len(indices) != len(instructions):
        raise ValueError("task indices and instructions have different lengths")
    updated = list(instructions)
    replacements = {}
    seen = set()
    for position, (index, instruction) in enumerate(zip(indices, instructions)):
        if index not in definitions:
            continue
        target = definitions[index]
        if instruction not in {legacy_tasks[index], target}:
            raise ValueError(f"task_index {index}: unexpected instruction {instruction!r}")
        if index in seen:
            raise ValueError(f"duplicate task_index {index}")
        seen.add(index)
        updated[position] = target
        replacements[legacy_tasks[index]] = target
    if seen != set(definitions):
        raise ValueError(f"missing task indices: {sorted(set(definitions) - seen)}")
    return updated, replacements


def prepare_metadata(
    source: Path, output: Path, definitions: dict[int, str],
    *, legacy_tasks: dict[int, str] = LEGACY_TASKS,
) -> None:
    """Write a metadata overlay; leave source parquet, videos and data untouched."""
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    if output.exists():
        raise FileExistsError(output)
    tasks = parquet.read_table(source / "meta/tasks.parquet")
    instructions, replacements = corrected_tasks(
        tasks["task_index"].to_pylist(), tasks["task"].to_pylist(), definitions,
        legacy_tasks=legacy_tasks,
    )
    tasks = tasks.set_column(tasks.schema.get_field_index("task"), tasks.schema.field("task"),
        arrow.array(instructions, type=tasks.schema.field("task").type))
    episode_paths = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise ValueError("no LeRobot episode metadata found")
    tables = [(Path("meta/tasks.parquet"), tasks)]
    for path in episode_paths:
        episodes = parquet.read_table(path)
        values = [[replacements.get(instruction, instruction) for instruction in row]
                  for row in episodes["tasks"].to_pylist()]
        episodes = episodes.set_column(
            episodes.schema.get_field_index("tasks"), episodes.schema.field("tasks"),
            arrow.array(values, type=episodes.schema.field("tasks").type),
        )
        tables.append((path.relative_to(source), episodes))
    output.mkdir(parents=True)
    for relative, table in tables:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, destination)
        if not parquet.read_table(destination).equals(table, check_metadata=True):
            raise ValueError(f"metadata verification failed: {relative}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeRobot full-shot or one-shot root")
    parser.add_argument("--output", type=Path, required=True, help="new metadata overlay directory")
    parser.add_argument("--definitions", type=Path, default=Path(__file__).resolve().parents[1]
                        / "docs/data/real_world_tasks.json")
    args = parser.parse_args(argv)
    definitions = json.loads(args.definitions.read_text(encoding="utf-8"))
    prepare_metadata(args.source, args.output,
                     {row["task_index"]: row["instruction"] for row in definitions["tasks"]},
                     legacy_tasks={"real-world": LEGACY_TASKS,
                                   "simulation": SIM_LEGACY_TASKS}[definitions["domain"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
