#!/usr/bin/env python3
"""Step 3 — Ground Truth Labeling.

Defines what counts as a genuine drone-aircraft near-collision, independent
of any buffer shape: a real aircraft within 200m horizontally and 50m
vertically of a drone (FAA/ICAO separation minima). Every drone-second (from
Step 2's simulated tracks) is labeled DANGER or SAFE against real Step 1
aircraft positions — the answer key the three buffer geometries (steps 4-6)
are graded against in Step 5's experiment.

Real aircraft are polled ~every 60s (with gaps); drones are simulated at 1Hz.
To label at 1Hz, each aircraft's position is dead-reckoned (linearly
interpolated) between consecutive polls, but never across a gap longer than
AIRCRAFT_GAP_CUTOFF_S (the track is presumed lost/reacquired, not a straight
line). Drone altitude (AGL) is compared directly against aircraft
geo_altitude with no terrain offset — a documented simplifying assumption
(no elevation dataset in this pipeline), reasonable for a low-relief coastal
region but a limitation worth stating explicitly in Step 8's paper.

Matching is done via a whole-second bucket index (not scipy.spatial.cKDTree
— unnecessary at this dataset's concurrency, see the plan) so the O(n*m)
distance math only ever runs over the handful of drones/aircraft active in
the same second, not a full cross-join.

Usage:
    python3 -m pipeline.step3_ground_truth_labeling [--region Boston]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

from pipeline.config import (
    AIRCRAFT_GAP_CUTOFF_S,
    CANDIDATE_SEARCH_RADIUS_M,
    CLEAN_DIR,
    DANGER_HORIZONTAL_M,
    DANGER_VERTICAL_M,
    DRONE_DENSITY_LEVELS,
    GROUND_TRUTH_CANDIDATES_SCHEMA,
    GROUND_TRUTH_LABELS_SCHEMA,
    REGIONS,
    region_slug,
    utm_epsg_for_bbox,
)
from pipeline.parquet_io import (
    atomic_write_parquet,
    ground_truth_candidates_path,
    ground_truth_labels_path,
    simulated_partition_path,
)

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _epoch_seconds(series):
    """Float seconds since epoch, resolution-independent (works regardless of
    whether the underlying datetime64 is us/ns precision)."""
    return (series - _EPOCH) / pd.Timedelta(seconds=1)


def load_airborne_aircraft(region_slug_):
    pattern = os.path.join(CLEAN_DIR, f"region={region_slug_}", "date=*", "data.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(
            f"No clean data found for region={region_slug_} under {CLEAN_DIR} — "
            "Step 1 needs at least one compacted day before Step 3 can run."
        )
    cols = ["icao24", "query_timestamp_utc", "longitude", "latitude", "geo_altitude", "on_ground"]
    tables = [pq.read_table(f, columns=cols) for f in files]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    df = df[(~df["on_ground"]) & df["geo_altitude"].notna()].copy()
    return df.sort_values(["icao24", "query_timestamp_utc"]).reset_index(drop=True)


def build_aircraft_segments(air_df, gap_cutoff_s=AIRCRAFT_GAP_CUTOFF_S):
    """Interpolatable segments (consecutive same-aircraft poll pairs within
    gap_cutoff_s) plus a zero-width singleton segment for every valid
    observation, so first/last observations and either side of a dropped gap
    are still queryable."""
    singles = pd.DataFrame({
        "icao24": air_df["icao24"],
        "t0": air_df["query_timestamp_utc"], "t1": air_df["query_timestamp_utc"],
        "lon0": air_df["longitude"], "lat0": air_df["latitude"], "alt0": air_df["geo_altitude"],
        "lon1": air_df["longitude"], "lat1": air_df["latitude"], "alt1": air_df["geo_altitude"],
    })

    nxt = air_df.groupby("icao24", sort=False)[
        ["query_timestamp_utc", "longitude", "latitude", "geo_altitude"]
    ].shift(-1)
    gap_s = (nxt["query_timestamp_utc"] - air_df["query_timestamp_utc"]).dt.total_seconds()
    valid = nxt["query_timestamp_utc"].notna() & (gap_s > 0) & (gap_s <= gap_cutoff_s)

    pairs = pd.DataFrame({
        "icao24": air_df.loc[valid, "icao24"],
        "t0": air_df.loc[valid, "query_timestamp_utc"], "t1": nxt.loc[valid, "query_timestamp_utc"],
        "lon0": air_df.loc[valid, "longitude"], "lat0": air_df.loc[valid, "latitude"], "alt0": air_df.loc[valid, "geo_altitude"],
        "lon1": nxt.loc[valid, "longitude"], "lat1": nxt.loc[valid, "latitude"], "alt1": nxt.loc[valid, "geo_altitude"],
    })

    return pd.concat([singles, pairs], ignore_index=True)


def _bucket_by_second(seconds_float, span_end_float=None):
    """Range-expand each (start[, end]) into whole-second bucket keys, sorted
    and split by numpy boundary detection (not pandas.groupby — too much
    per-group overhead at tens of thousands of keys).

    If span_end_float is None, each entry occupies exactly its own floor
    second (drone rows). Otherwise each entry spans floor(start)..floor(end)
    inclusive (aircraft segments). Returns dict {second: np.array(row indices)}.
    """
    start_floor = np.floor(seconds_float).astype(np.int64)
    if span_end_float is None:
        bucket_seconds = start_floor
        row_idx = np.arange(len(seconds_float))
    else:
        end_floor = np.floor(span_end_float).astype(np.int64)
        lengths = (end_floor - start_floor + 1).astype(np.int64)
        n = len(seconds_float)
        total = int(lengths.sum())
        cum_lengths = np.cumsum(lengths)
        group_start_pos = cum_lengths - lengths
        offsets = np.arange(total) - np.repeat(group_start_pos, lengths)
        bucket_seconds = np.repeat(start_floor, lengths) + offsets
        row_idx = np.repeat(np.arange(n), lengths)

    order = np.argsort(bucket_seconds, kind="stable")
    bucket_seconds_sorted = bucket_seconds[order]
    row_idx_sorted = row_idx[order]
    unique_seconds, start_positions = np.unique(bucket_seconds_sorted, return_index=True)
    groups = np.split(row_idx_sorted, start_positions[1:])
    return dict(zip(unique_seconds.tolist(), groups))


def label_density_level(drone_df, segments_df, transformer):
    seg_t0 = _epoch_seconds(segments_df["t0"]).to_numpy()
    seg_t1 = _epoch_seconds(segments_df["t1"]).to_numpy()
    seg_x0, seg_y0 = transformer.transform(segments_df["lon0"].to_numpy(), segments_df["lat0"].to_numpy())
    seg_x1, seg_y1 = transformer.transform(segments_df["lon1"].to_numpy(), segments_df["lat1"].to_numpy())
    seg_alt0 = segments_df["alt0"].to_numpy()
    seg_alt1 = segments_df["alt1"].to_numpy()
    seg_icao = segments_df["icao24"].to_numpy()
    seg_denom = seg_t1 - seg_t0
    seg_denom[seg_denom == 0] = 1.0  # singleton segments: t0==t1, x0==x1 so frac is irrelevant

    bucket_index = _bucket_by_second(seg_t0, seg_t1)

    drone_sec = _epoch_seconds(drone_df["timestamp_utc"]).to_numpy()
    drone_x, drone_y = transformer.transform(drone_df["longitude"].to_numpy(), drone_df["latitude"].to_numpy())
    drone_alt = drone_df["altitude_m"].to_numpy()
    drone_buckets = _bucket_by_second(drone_sec)

    n = len(drone_df)
    danger = np.zeros(n, dtype=bool)
    n_candidates = np.zeros(n, dtype=np.int16)
    nearest_icao = np.full(n, None, dtype=object)
    nearest_h = np.full(n, np.nan)
    nearest_v = np.full(n, np.nan)

    cand_drone_row, cand_icao, cand_dx, cand_dy, cand_h, cand_v = [], [], [], [], [], []

    for sec, drone_rows in drone_buckets.items():
        seg_rows = bucket_index.get(sec)
        if seg_rows is None or len(seg_rows) == 0:
            continue

        d_ts = drone_sec[drone_rows]
        d_x = drone_x[drone_rows]
        d_y = drone_y[drone_rows]
        d_alt = drone_alt[drone_rows]

        t0 = seg_t0[seg_rows]
        denom = seg_denom[seg_rows]
        x0, y0, alt0 = seg_x0[seg_rows], seg_y0[seg_rows], seg_alt0[seg_rows]
        x1, y1, alt1 = seg_x1[seg_rows], seg_y1[seg_rows], seg_alt1[seg_rows]
        icao = seg_icao[seg_rows]

        frac = np.clip((d_ts[:, None] - t0[None, :]) / denom[None, :], 0.0, 1.0)
        ax = x0[None, :] + (x1 - x0)[None, :] * frac
        ay = y0[None, :] + (y1 - y0)[None, :] * frac
        aalt = alt0[None, :] + (alt1 - alt0)[None, :] * frac

        dx = ax - d_x[:, None]  # aircraft x - drone x (signed, for Step 6's ellipsoid)
        dy = ay - d_y[:, None]
        hdist = np.sqrt(dx ** 2 + dy ** 2)
        vdist = np.abs(d_alt[:, None] - aalt)
        within = hdist <= CANDIDATE_SEARCH_RADIUS_M

        n_candidates[drone_rows] = within.sum(axis=1)
        is_danger = within & (hdist <= DANGER_HORIZONTAL_M) & (vdist <= DANGER_VERTICAL_M)
        danger[drone_rows] = is_danger.any(axis=1)

        masked_h = np.where(within, hdist, np.inf)
        best_j = np.argmin(masked_h, axis=1)
        has_any = within.any(axis=1)
        local_i = np.arange(len(drone_rows))
        nearest_icao[np.array(drone_rows)[has_any]] = icao[best_j[has_any]]
        nearest_h[np.array(drone_rows)[has_any]] = hdist[local_i[has_any], best_j[has_any]]
        nearest_v[np.array(drone_rows)[has_any]] = vdist[local_i[has_any], best_j[has_any]]

        li, lj = np.nonzero(within)
        if len(li):
            cand_drone_row.append(np.array(drone_rows)[li])
            cand_icao.append(icao[lj])
            cand_dx.append(dx[li, lj])
            cand_dy.append(dy[li, lj])
            cand_h.append(hdist[li, lj])
            cand_v.append(vdist[li, lj])

    # Built from drone_df's own columns (not via .to_numpy()) so the
    # tz-aware timestamp_utc dtype survives intact — round-tripping a
    # tz-aware Series through .to_numpy() silently degrades it to an object
    # array of Timestamps, which pyarrow then fights with the pinned schema.
    drone_reset = drone_df.reset_index(drop=True)
    labels_df = drone_reset[["drone_id", "density_level", "timestamp_utc", "latitude", "longitude", "altitude_m"]].copy()
    labels_df["danger"] = danger
    labels_df["n_candidates"] = n_candidates
    labels_df["nearest_icao24"] = nearest_icao
    labels_df["nearest_horizontal_m"] = nearest_h
    labels_df["nearest_vertical_m"] = nearest_v

    if cand_drone_row:
        rows = np.concatenate(cand_drone_row)
        base = drone_reset.iloc[rows][["drone_id", "density_level", "timestamp_utc"]].reset_index(drop=True)
        candidates_df = base.assign(
            icao24=np.concatenate(cand_icao),
            dx_m=np.concatenate(cand_dx),
            dy_m=np.concatenate(cand_dy),
            horizontal_m=np.concatenate(cand_h),
            vertical_m=np.concatenate(cand_v),
        )
    else:
        candidates_df = pd.DataFrame({f.name: pd.Series(dtype=object) for f in GROUND_TRUTH_CANDIDATES_SCHEMA})

    return labels_df, candidates_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    bbox = REGIONS[args.region]
    epsg = utm_epsg_for_bbox(bbox)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    air_df = load_airborne_aircraft(slug)
    segments_df = build_aircraft_segments(air_df)
    print(
        f"Aircraft: {air_df['icao24'].nunique()} unique, {len(air_df)} airborne polls, "
        f"{len(segments_df)} segments (incl. singletons) for '{args.region}'\n"
    )

    for density_name in DRONE_DENSITY_LEVELS:
        drone_path = simulated_partition_path(slug, density_name)
        drone_df = pd.read_parquet(drone_path)
        labels_df, candidates_df = label_density_level(drone_df, segments_df, transformer)

        atomic_write_parquet(labels_df, ground_truth_labels_path(slug, density_name), GROUND_TRUTH_LABELS_SCHEMA)
        atomic_write_parquet(candidates_df, ground_truth_candidates_path(slug, density_name), GROUND_TRUTH_CANDIDATES_SCHEMA)

        n_danger = int(labels_df["danger"].sum())
        print(
            f"  {density_name}: {len(labels_df)} drone-seconds, {n_danger} DANGER "
            f"({100 * n_danger / max(len(labels_df), 1):.4f}%), {len(candidates_df)} candidate rows "
            f"-> {ground_truth_labels_path(slug, density_name)}"
        )


if __name__ == "__main__":
    main()
