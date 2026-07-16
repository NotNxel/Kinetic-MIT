#!/usr/bin/env python3
"""Step 5 — Sphere Buffer Geometry.

A single-radius sphere around each drone at each second, sized to the exact
same volume as the Step 4 cylinder — uniform protection in every direction,
including vertically, unlike the cylinder's separate radius/height. Because
a sphere is symmetric in all directions, containment is true 3D Euclidean
distance (horizontal**2 + vertical**2 <= r**2), not the cylinder's
independent horizontal/vertical checks.

Usage:
    python3 -m pipeline.step5_buffer_sphere [--region Boston]
"""

import argparse

import pandas as pd

from pipeline.buffer_common import evaluate_buffer
from pipeline.config import (
    BUFFER_ALERTS_SCHEMA,
    DRONE_DENSITY_LEVELS,
    REGIONS,
    SPHERE_RADIUS_M,
    region_slug,
)
from pipeline.parquet_io import (
    atomic_write_parquet,
    buffer_alerts_path,
    ground_truth_candidates_path,
    simulated_partition_path,
)

SHAPE = "sphere"


def contains(candidates_df):
    return (candidates_df["horizontal_m"] ** 2 + candidates_df["vertical_m"] ** 2) <= SPHERE_RADIUS_M ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    print(f"Sphere buffer: r={SPHERE_RADIUS_M:.3f}m, for '{args.region}'\n")

    for density_name in DRONE_DENSITY_LEVELS:
        drone_df = pd.read_parquet(simulated_partition_path(slug, density_name))
        candidates_df = pd.read_parquet(ground_truth_candidates_path(slug, density_name))

        alerts_df = evaluate_buffer(drone_df, candidates_df, contains)
        atomic_write_parquet(alerts_df, buffer_alerts_path(slug, density_name, SHAPE), BUFFER_ALERTS_SCHEMA)

        n_alert = int(alerts_df["alert"].sum())
        print(
            f"  {density_name}: {len(alerts_df)} drone-seconds, {n_alert} alerts "
            f"({100 * n_alert / max(len(alerts_df), 1):.4f}%)"
        )


if __name__ == "__main__":
    main()
