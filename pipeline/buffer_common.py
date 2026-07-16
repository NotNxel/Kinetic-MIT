"""Shared buffer-shape evaluation, reused by steps 4-6 (cylinder, sphere,
ellipsoid). Each step supplies its own containment predicate over Step 3's
candidates.parquet (one row per drone-second/aircraft pair within the search
radius, columns: dx_m, dy_m, horizontal_m, vertical_m) plus the drone's
heading_deg, joined in here — only the predicate differs per shape; the
aggregation into per-drone-second alerts is identical across all three.
"""

import pandas as pd


def evaluate_buffer(drone_df, candidates_df, contains_fn):
    """
    drone_df: this density's full Step 2 drone table (one row per drone-second).
    candidates_df: Step 3's candidates.parquet for the same density.
    contains_fn: candidates DataFrame (with dx_m, dy_m, horizontal_m,
        vertical_m, heading_deg columns) -> bool array, True where the
        aircraft falls inside this shape's buffer. Cylinder/sphere only need
        horizontal_m/vertical_m; the ellipsoid also needs dx_m/dy_m/heading_deg.

    Returns one row per drone-second: `alert` plus shape-agnostic
    nearest-aircraft diagnostics (nearest by horizontal distance among all
    search-radius candidates, not just ones inside the shape — mirrors Step
    3's convention).
    """
    drone_reset = drone_df.reset_index(drop=True)
    base = drone_reset[["drone_id", "density_level", "timestamp_utc", "latitude", "longitude", "altitude_m"]].copy()

    if candidates_df.empty:
        base["alert"] = False
        base["n_candidates"] = pd.array([0] * len(base), dtype="int16")
        base["nearest_icao24"] = None
        base["nearest_horizontal_m"] = float("nan")
        base["nearest_vertical_m"] = float("nan")
        return base

    candidates_df = candidates_df.merge(
        drone_reset[["drone_id", "timestamp_utc", "heading_deg"]],
        on=["drone_id", "timestamp_utc"], how="left",
    )

    inside = contains_fn(candidates_df)
    alert_keys = (
        candidates_df.loc[inside, ["drone_id", "timestamp_utc"]]
        .drop_duplicates()
        .assign(alert=True)
    )

    n_candidates = (
        candidates_df.groupby(["drone_id", "timestamp_utc"])
        .size()
        .rename("n_candidates")
        .reset_index()
    )

    nearest = (
        candidates_df.sort_values("horizontal_m")
        .drop_duplicates(subset=["drone_id", "timestamp_utc"], keep="first")
        [["drone_id", "timestamp_utc", "icao24", "horizontal_m", "vertical_m"]]
        .rename(columns={
            "icao24": "nearest_icao24",
            "horizontal_m": "nearest_horizontal_m",
            "vertical_m": "nearest_vertical_m",
        })
    )

    base = base.merge(alert_keys, on=["drone_id", "timestamp_utc"], how="left")
    base["alert"] = base["alert"].fillna(False)
    base = base.merge(n_candidates, on=["drone_id", "timestamp_utc"], how="left")
    base["n_candidates"] = base["n_candidates"].fillna(0).astype("int16")
    base = base.merge(nearest, on=["drone_id", "timestamp_utc"], how="left")

    return base
