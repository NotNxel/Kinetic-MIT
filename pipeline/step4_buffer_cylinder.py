#!/usr/bin/env python3
"""Step 4 — Cylinder Buffer Geometry.

The current FAA-standard protection shape: fixed horizontal radius, fixed
vertical height around each drone at each second — uniform in all
directions. Uses the exact same numbers as Step 3's ground-truth danger
threshold (200m horizontal radius, 50m vertical half-height) — it IS the FAA
standard ground truth was defined from, so its alerts should almost exactly
reproduce Step 3's DANGER labels. That equivalence is checked explicitly
below as a correctness test on the whole pipeline, before the same
evaluation framework (pipeline/buffer_common.py) is trusted for the sphere
(step 5) and ellipsoid (step 6) — same volume, different shape, which will
*not* trivially match.

Usage:
    python3 -m pipeline.step4_buffer_cylinder [--region Boston]
"""

import argparse

import pandas as pd

from pipeline.buffer_common import evaluate_buffer
from pipeline.config import (
    BUFFER_ALERTS_SCHEMA,
    CYLINDER_HORIZONTAL_RADIUS_M,
    CYLINDER_VERTICAL_HALF_HEIGHT_M,
    DRONE_DENSITY_LEVELS,
    REGIONS,
    region_slug,
)
from pipeline.parquet_io import (
    atomic_write_parquet,
    buffer_alerts_path,
    ground_truth_candidates_path,
    ground_truth_labels_path,
    simulated_partition_path,
)

SHAPE = "cylinder"


def contains(candidates_df):
    return (candidates_df["horizontal_m"] <= CYLINDER_HORIZONTAL_RADIUS_M) & (
        candidates_df["vertical_m"] <= CYLINDER_VERTICAL_HALF_HEIGHT_M
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    print(
        f"Cylinder buffer: R={CYLINDER_HORIZONTAL_RADIUS_M}m horizontal, "
        f"+/-{CYLINDER_VERTICAL_HALF_HEIGHT_M}m vertical, for '{args.region}'\n"
    )

    for density_name in DRONE_DENSITY_LEVELS:
        drone_df = pd.read_parquet(simulated_partition_path(slug, density_name))
        candidates_df = pd.read_parquet(ground_truth_candidates_path(slug, density_name))

        alerts_df = evaluate_buffer(drone_df, candidates_df, contains)
        atomic_write_parquet(alerts_df, buffer_alerts_path(slug, density_name, SHAPE), BUFFER_ALERTS_SCHEMA)

        labels_df = pd.read_parquet(ground_truth_labels_path(slug, density_name))
        merged = alerts_df[["drone_id", "timestamp_utc", "alert"]].merge(
            labels_df[["drone_id", "timestamp_utc", "danger"]], on=["drone_id", "timestamp_utc"], how="inner"
        )
        matches = (merged["alert"] == merged["danger"]).all() and len(merged) == len(alerts_df) == len(labels_df)

        n_alert = int(alerts_df["alert"].sum())
        print(
            f"  {density_name}: {len(alerts_df)} drone-seconds, {n_alert} alerts "
            f"({100 * n_alert / max(len(alerts_df), 1):.4f}%) -- "
            f"alert==danger check: {'PASS' if matches else 'FAIL'}"
        )
        if not matches:
            raise AssertionError(
                f"{density_name}: cylinder alerts do not exactly match ground-truth danger labels"
            )


if __name__ == "__main__":
    main()
