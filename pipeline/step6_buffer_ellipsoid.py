#!/usr/bin/env python3
"""Step 6 — Ellipsoid Buffer Geometry (physics-motivated).

Same volume as the Step 4 cylinder, but stretched in the direction of
travel: longer in front of the drone (where danger accumulates fastest),
shorter to the sides. A prolate spheroid: semi-major axis `a` along the
drone's heading, semi-minor axis `b` equal in the two directions
perpendicular to it (lateral and vertical). Aspect ratio a:b = 3:1 is a
stated modeling choice (not derived from a cited standard) documented in the
Step 8 paper; solving V=(4/3)pi*a*b^2 with a=3b gives a=300m, b=100m.

Containment decomposes each candidate's horizontal offset (dx_m, dy_m) into
along-heading and cross-heading components using the drone's heading_deg
(compass bearing, 0=North, clockwise), then applies the standard ellipsoid
inequality.

Usage:
    python3 -m pipeline.step6_buffer_ellipsoid [--region Boston]
"""

import argparse

import numpy as np
import pandas as pd

from pipeline.buffer_common import evaluate_buffer
from pipeline.config import (
    BUFFER_ALERTS_SCHEMA,
    DRONE_DENSITY_LEVELS,
    ELLIPSOID_MAJOR_AXIS_M,
    ELLIPSOID_MINOR_AXIS_M,
    REGIONS,
    region_slug,
)
from pipeline.parquet_io import (
    atomic_write_parquet,
    buffer_alerts_path,
    ground_truth_candidates_path,
    simulated_partition_path,
)

SHAPE = "ellipsoid"


def contains(candidates_df):
    theta = np.radians(candidates_df["heading_deg"].to_numpy())
    dx = candidates_df["dx_m"].to_numpy()
    dy = candidates_df["dy_m"].to_numpy()
    vertical_m = candidates_df["vertical_m"].to_numpy()

    u = dx * np.sin(theta) + dy * np.cos(theta)  # along heading
    v = dx * np.cos(theta) - dy * np.sin(theta)  # cross heading

    return (u / ELLIPSOID_MAJOR_AXIS_M) ** 2 + (v / ELLIPSOID_MINOR_AXIS_M) ** 2 + (
        vertical_m / ELLIPSOID_MINOR_AXIS_M
    ) ** 2 <= 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    print(
        f"Ellipsoid buffer: a={ELLIPSOID_MAJOR_AXIS_M:.1f}m (heading axis), "
        f"b={ELLIPSOID_MINOR_AXIS_M:.1f}m (lateral/vertical), for '{args.region}'\n"
    )

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
