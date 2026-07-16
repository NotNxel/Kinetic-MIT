#!/usr/bin/env python3
"""Step 2 — Drone Simulation.

Generates synthetic drone flights with the same schema as the cleaned Step 1
ADS-B data (timestamp, lat, lon, altitude, heading, speed), over the exact
real-data time window Step 1 has already collected for a region, so Step 3's
ground-truth labeling and Step 5's buffer-geometry experiment have drone
tracks to test against real aircraft positions.

Each drone flies a straight-line point-to-point flight: a random start point
and compass bearing within the region's bounding box, constant speed,
constant altitude (0-400ft AGL), for a duration/distance capped to realistic
consumer-drone range (~1.5-8km, 5-25 minutes). Drones are spawned via a
homogeneous Poisson process at three traffic densities: sparse (10/hour),
medium (50/hour), dense (200/hour).

Usage:
    python3 -m pipeline.step2_simulate_drones [--region Boston] [--seed 42]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Geod

from pipeline.config import (
    CLEAN_DIR,
    DRONE_ALTITUDE_RANGE_M,
    DRONE_DENSITY_LEVELS,
    DRONE_DURATION_RANGE_S,
    DRONE_MAX_DISTANCE_M,
    DRONE_SCHEMA,
    DRONE_SIM_SEED,
    DRONE_SPEED_RANGE_MPS,
    REGIONS,
    region_slug,
)
from pipeline.parquet_io import atomic_write_parquet, simulated_partition_path

MAX_BEARING_RETRIES = 20
MAX_START_POINT_RETRIES = 20


def get_region_time_window(region_slug_):
    """Return (t_min, t_max) tz-aware timestamps spanning all clean data
    collected so far for this region."""
    pattern = os.path.join(CLEAN_DIR, f"region={region_slug_}", "date=*", "data.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(
            f"No clean data found for region={region_slug_} under {CLEAN_DIR} — "
            "Step 1 needs at least one compacted day before Step 2 can run."
        )
    tables = [pq.read_table(f, columns=["query_timestamp_utc"]) for f in files]
    ts = pd.concat([t.to_pandas() for t in tables], ignore_index=True)["query_timestamp_utc"]
    return ts.min(), ts.max()


def sample_spawn_times(t_min, t_max, rate_per_hour, rng):
    """Homogeneous Poisson process spawn times: Poisson-distributed count,
    sorted uniform offsets within the window — the vectorized equivalent of
    drawing sequential exponential inter-arrival gaps."""
    window_seconds = (t_max - t_min).total_seconds()
    window_hours = window_seconds / 3600.0
    n_drones = rng.poisson(rate_per_hour * window_hours)
    offsets = np.sort(rng.uniform(0, window_seconds, size=n_drones))
    # Round to microsecond precision: DRONE_SCHEMA pins timestamp[us], and the
    # float-second offsets otherwise carry sub-microsecond noise that pyarrow
    # refuses to silently truncate when casting to the pinned schema.
    return (t_min + pd.to_timedelta(offsets, unit="s")).round("us")


def sample_flight(rng, geod, bbox):
    """Sample one drone's flight parameters: bearing-first so distance/duration
    stay within realistic consumer-drone bounds by construction, retrying the
    bearing (and, if needed, the start point) until the endpoint lands inside
    the bbox."""
    lamin, lomin, lamax, lomax = bbox
    speed_mps = rng.uniform(*DRONE_SPEED_RANGE_MPS)
    duration_raw = rng.uniform(*DRONE_DURATION_RANGE_S)
    duration_s = min(duration_raw, DRONE_MAX_DISTANCE_M / speed_mps)
    altitude_m = rng.uniform(*DRONE_ALTITUDE_RANGE_M)
    distance_m = speed_mps * duration_s

    for _ in range(MAX_START_POINT_RETRIES):
        start_lat = rng.uniform(lamin, lamax)
        start_lon = rng.uniform(lomin, lomax)
        for _ in range(MAX_BEARING_RETRIES):
            bearing_deg = rng.uniform(0, 360)
            end_lon, end_lat, _ = geod.fwd(start_lon, start_lat, bearing_deg, distance_m)
            if lamin <= end_lat <= lamax and lomin <= end_lon <= lomax:
                return {
                    "speed_mps": speed_mps,
                    "duration_s": duration_s,
                    "altitude_m": altitude_m,
                    "start_lat": start_lat,
                    "start_lon": start_lon,
                    "bearing_deg": bearing_deg,
                }

    raise RuntimeError(
        "Failed to sample a valid in-bbox flight after repeated retries — "
        "bbox may be too small relative to DRONE_MAX_DISTANCE_M."
    )


def interpolate_track(geod, flight, spawn_ts):
    """One vectorized geod.fwd() call producing the drone's full per-second
    straight-line track, inclusive of both the spawn second and arrival
    second."""
    n = int(flight["duration_s"]) + 1
    t = np.arange(n)
    dists = flight["speed_mps"] * t
    lons, lats, _ = geod.fwd(
        np.full(n, flight["start_lon"]),
        np.full(n, flight["start_lat"]),
        np.full(n, flight["bearing_deg"]),
        dists,
    )
    return pd.DataFrame({
        "timestamp_utc": spawn_ts + pd.to_timedelta(t, unit="s"),
        "latitude": lats,
        "longitude": lons,
        "altitude_m": flight["altitude_m"],
        "heading_deg": flight["bearing_deg"],
        "speed_mps": flight["speed_mps"],
    })


def simulate_density_level(density_name, rate_per_hour, bbox, t_min, t_max, seed):
    rng = np.random.default_rng(seed)
    geod = Geod(ellps="WGS84")
    spawn_times = sample_spawn_times(t_min, t_max, rate_per_hour, rng)

    drone_tracks = []
    for i, spawn_ts in enumerate(spawn_times):
        flight = sample_flight(rng, geod, bbox)
        track = interpolate_track(geod, flight, spawn_ts)
        track.insert(0, "density_level", density_name)
        track.insert(0, "drone_id", f"{density_name}_{i:06d}")
        drone_tracks.append(track)

    if not drone_tracks:
        return pd.DataFrame({f.name: pd.Series(dtype=object) for f in DRONE_SCHEMA})

    return pd.concat(drone_tracks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    parser.add_argument("--seed", type=int, default=DRONE_SIM_SEED)
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    bbox = REGIONS[args.region]
    t_min, t_max = get_region_time_window(slug)
    window_hours = (t_max - t_min).total_seconds() / 3600.0
    print(f"Simulating drones for '{args.region}' over {t_min} -> {t_max} ({window_hours:.2f}h)\n")

    seed_seq = np.random.SeedSequence(args.seed)
    child_seeds = seed_seq.spawn(len(DRONE_DENSITY_LEVELS))

    for (density_name, rate_per_hour), child_seed in zip(DRONE_DENSITY_LEVELS.items(), child_seeds):
        df = simulate_density_level(density_name, rate_per_hour, bbox, t_min, t_max, child_seed)
        path = simulated_partition_path(slug, density_name)
        atomic_write_parquet(df, path, DRONE_SCHEMA)
        n_drones = df["drone_id"].nunique() if not df.empty else 0
        print(f"  {density_name} ({rate_per_hour}/hr): {n_drones} drones, {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
