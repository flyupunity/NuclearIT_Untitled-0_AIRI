"""Create reproducible temporal indices for the ERA5 Weather Codec dataset."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


CHANNEL_NAMES = [
    "t2m",
    "mslp",
    "u10",
    "v10",
    "tp6h",
    "sst",
    "tcwv",
    "tcc",
    "T1000",
    "T925",
    "T850",
    "T700",
    "U1000",
    "U925",
    "U850",
    "U700",
    "V1000",
    "V925",
    "V850",
    "V700",
    "Z1000",
    "Z925",
    "Z850",
    "Z700",
    "Q1000",
    "Q925",
    "Q850",
    "Q700",
]

SPLIT_RANGES = {
    "train": (
        datetime(2014, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2019, 12, 31, 18, tzinfo=timezone.utc),
    ),
    "validation": (
        datetime(2020, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2020, 12, 31, 18, tzinfo=timezone.utc),
    ),
    "test": (
        datetime(2021, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2021, 12, 31, 18, tzinfo=timezone.utc),
    ),
}


def iso_timestamp(value: datetime) -> str:
    """Return an xarray/NumPy-friendly UTC timestamp without a timezone suffix."""
    return value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
        timespec="seconds"
    )


def six_hour_axis(start: datetime, end: datetime) -> list[datetime]:
    if end < start:
        raise ValueError("End timestamp precedes start timestamp.")

    step = timedelta(hours=6)
    values: list[datetime] = []
    current = start

    while current <= end:
        values.append(current)
        current += step

    return values


def stratified_random_indices(
    length: int,
    count: int,
    seed: int,
) -> list[int]:
    """
    Split a timeline into `count` consecutive bins and sample one index per bin.

    The result is sorted, unique and reproducible. This avoids clustering all
    selected frames in one season while retaining randomness inside each bin.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if count > length:
        raise ValueError(
            f"Cannot select {count} unique indices from a timeline of {length}."
        )

    rng = random.Random(seed)
    indices: list[int] = []

    for bin_index in range(count):
        left = (bin_index * length) // count
        right = ((bin_index + 1) * length) // count
        indices.append(rng.randrange(left, right))

    if indices != sorted(set(indices)):
        raise RuntimeError("Temporal selection is not strictly increasing.")

    return indices


def adjacent_local_pairs(source_indices: list[int]) -> list[list[int]]:
    """Return local array pairs whose source timestamps differ by exactly 6 h."""
    return [
        [local_index, local_index + 1]
        for local_index in range(len(source_indices) - 1)
        if source_indices[local_index + 1] - source_indices[local_index] == 1
    ]


def split_record(
    name: str,
    count: int,
    seed: int,
) -> dict:
    start, end = SPLIT_RANGES[name]
    full_axis = six_hour_axis(start, end)
    source_indices = stratified_random_indices(
        length=len(full_axis),
        count=count,
        seed=seed,
    )
    timestamps = [
        iso_timestamp(full_axis[index])
        for index in source_indices
    ]

    return {
        "period_start": iso_timestamp(start),
        "period_end": iso_timestamp(end),
        "cadence_hours": 6,
        "source_frame_count": len(full_axis),
        "selected_frame_count": len(source_indices),
        "source_indices": source_indices,
        "timestamps": timestamps,
        "adjacent_six_hour_pairs_local": adjacent_local_pairs(source_indices),
    }


def build_manifest(
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict:
    splits = {
        "train": split_record("train", train_count, seed + 0),
        "validation": split_record(
            "validation",
            validation_count,
            seed + 1,
        ),
        "test": split_record("test", test_count, seed + 2),
    }

    return {
        "schema_version": 1,
        "dataset": "ERA5 / WeatherBench2",
        "grid": {
            "name": "0.5_degree_cell_centred",
            "latitude_points": 360,
            "longitude_points": 720,
        },
        "selection": {
            "strategy": "stratified_random_one_per_contiguous_time_bin",
            "seed": seed,
            "cadence_hours": 6,
            "test_used_for_selection": False,
        },
        "channel_names": CHANNEL_NAMES,
        "splits": splits,
        # Compatibility aliases read directly by burmalda.ipynb.
        "train_times": splits["train"]["timestamps"],
        "validation_times": splits["validation"]["timestamps"],
        "test_times": splits["test"]["timestamps"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the temporal manifest used by burmalda.ipynb."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifest.json"),
    )
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest: {args.output.resolve()}")
    for split_name, record in manifest["splits"].items():
        print(
            f"{split_name}: "
            f"{record['selected_frame_count']} / "
            f"{record['source_frame_count']} frames, "
            f"{len(record['adjacent_six_hour_pairs_local'])} adjacent pairs"
        )


if __name__ == "__main__":
    main()
