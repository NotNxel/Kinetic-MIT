#!/usr/bin/env python3
"""One-off export: compact JSON payload for the interactive flight-map
Artifact. Not part of the pipeline proper -- aggregates existing Step 1-7
outputs into a small, browser-friendly bundle (drone paths are exported as
reconstructable flight parameters, not per-second rows, to keep size sane).
"""

import argparse
import json
import math
import os

import pandas as pd

from pipeline.config import (
    CYLINDER_HORIZONTAL_RADIUS_M,
    CYLINDER_VERTICAL_HALF_HEIGHT_M,
    DRONE_DENSITY_LEVELS,
    ELLIPSOID_MAJOR_AXIS_M,
    ELLIPSOID_MINOR_AXIS_M,
    REGIONS,
    SPHERE_RADIUS_M,
    region_slug,
)
from pipeline.parquet_io import (
    buffer_alerts_path,
    ground_truth_candidates_path,
    ground_truth_labels_path,
    simulated_partition_path,
)

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

# --------------------------------------------------------------------------
# Uniform affine quantization (paper eq. 5-8) applied to lat/lon/altitude,
# the highest-volume numeric fields in the export -- compresses each value
# from a ~8-character 5-decimal float literal down to a short int16 literal
# in the JSON, shrinking the payload without touching precision meaningfully
# (int16 spread across each field's actual min/max gives ~1m resolution,
# comparable to the 5-decimal rounding this replaces). Dequantized back to
# floats once, client-side, immediately after load -- every other line of
# app code still sees plain floats, unchanged.
QMIN, QMAX = -32768, 32767


def quant_params(beta, alpha):
    """eq. 6-7: scale S and integer zero-point Z from the tensor's clipping bounds."""
    if alpha == beta:
        alpha = beta + 1e-9  # degenerate single-value field guard
    S = (alpha - beta) / (QMAX - QMIN)
    Z = round(-beta / S) + QMIN
    return S, Z


def quantize(x, S, Z):
    """eq. 5: q = round(x/S) + Z"""
    return round(x / S) + Z


def dequantize(q, S, Z):
    """eq. 8: x_hat = S*(q-Z) -- used here only to self-check round-trip error before writing output."""
    return S * (q - Z)


def epoch_s(series):
    return ((series - EPOCH) / pd.Timedelta(seconds=1)).round().astype("int64")


def export_aircraft(SLUG):
    df = pd.read_parquet(f"data/clean/region={SLUG}")
    air = df[~df["on_ground"]].copy()
    air = air[air["latitude"].notna() & air["longitude"].notna()]
    air["t"] = epoch_s(air["query_timestamp_utc"])
    air["alt"] = air["geo_altitude"].fillna(air["baro_altitude"])

    lat_S, lat_Z = quant_params(float(air["latitude"].min()), float(air["latitude"].max()))
    lon_S, lon_Z = quant_params(float(air["longitude"].min()), float(air["longitude"].max()))
    alt_vals = air["alt"].dropna()
    alt_S, alt_Z = quant_params(float(alt_vals.min()), float(alt_vals.max()))

    max_err_lat = max_err_lon = max_err_alt = 0.0
    out = []
    for icao, g in air.sort_values(["icao24", "t"]).groupby("icao24", sort=False):
        callsign = next((c for c in g["callsign"] if isinstance(c, str) and c.strip()), "")
        origin = g["origin_country"].iloc[0]
        pts = []
        for t, lat, lon, alt in zip(g["t"], g["latitude"], g["longitude"], g["alt"]):
            qlat, qlon = quantize(lat, lat_S, lat_Z), quantize(lon, lon_S, lon_Z)
            qalt = quantize(alt, alt_S, alt_Z) if pd.notna(alt) else None
            max_err_lat = max(max_err_lat, abs(dequantize(qlat, lat_S, lat_Z) - lat))
            max_err_lon = max(max_err_lon, abs(dequantize(qlon, lon_S, lon_Z) - lon))
            if qalt is not None:
                max_err_alt = max(max_err_alt, abs(dequantize(qalt, alt_S, alt_Z) - alt))
            pts.append([int(t), qlat, qlon, qalt])
        out.append({"icao": icao, "cs": callsign, "country": origin, "pts": pts})

    print(f"  aircraft quant round-trip max error: lat={max_err_lat:.7f}deg lon={max_err_lon:.7f}deg alt={max_err_alt:.3f}m")
    quant = {"lat": [lat_S, lat_Z], "lon": [lon_S, lon_Z], "alt": [alt_S, alt_Z]}
    return out, quant


def export_drones(SLUG):
    aggs = {}
    for density in DRONE_DENSITY_LEVELS:
        df = pd.read_parquet(simulated_partition_path(SLUG, density), columns=[
            "drone_id", "timestamp_utc", "latitude", "longitude", "altitude_m", "heading_deg", "speed_mps",
        ])
        df["t"] = epoch_s(df["timestamp_utc"])
        aggs[density] = df.sort_values(["drone_id", "t"]).groupby("drone_id", sort=False).agg(
            t0=("t", "min"), t1=("t", "max"),
            lat0=("latitude", "first"), lon0=("longitude", "first"),
            alt=("altitude_m", "first"), heading=("heading_deg", "first"), speed=("speed_mps", "first"),
        ).reset_index()

    all_agg = pd.concat(aggs.values(), ignore_index=True)
    lat_S, lat_Z = quant_params(float(all_agg["lat0"].min()), float(all_agg["lat0"].max()))
    lon_S, lon_Z = quant_params(float(all_agg["lon0"].min()), float(all_agg["lon0"].max()))
    alt_S, alt_Z = quant_params(float(all_agg["alt"].min()), float(all_agg["alt"].max()))

    max_err_lat = max_err_lon = max_err_alt = 0.0
    out = []
    for density, agg in aggs.items():
        for row in agg.itertuples(index=False):
            qlat, qlon, qalt = quantize(row.lat0, lat_S, lat_Z), quantize(row.lon0, lon_S, lon_Z), quantize(row.alt, alt_S, alt_Z)
            max_err_lat = max(max_err_lat, abs(dequantize(qlat, lat_S, lat_Z) - row.lat0))
            max_err_lon = max(max_err_lon, abs(dequantize(qlon, lon_S, lon_Z) - row.lon0))
            max_err_alt = max(max_err_alt, abs(dequantize(qalt, alt_S, alt_Z) - row.alt))
            out.append({
                "id": row.drone_id, "d": density,
                "t0": int(row.t0), "t1": int(row.t1),
                "lat0": qlat, "lon0": qlon,
                "alt": qalt, "hdg": round(float(row.heading), 2),
                "spd": round(float(row.speed), 2),
            })

    print(f"  drone quant round-trip max error: lat={max_err_lat:.7f}deg lon={max_err_lon:.7f}deg alt={max_err_alt:.3f}m")
    quant = {"lat": [lat_S, lat_Z], "lon": [lon_S, lon_Z], "alt": [alt_S, alt_Z]}
    return out, quant


def export_danger(SLUG):
    out = []
    for density in DRONE_DENSITY_LEVELS:
        labels = pd.read_parquet(ground_truth_labels_path(SLUG, density))
        d = labels[labels["danger"]].copy()
        if d.empty:
            continue
        d["t"] = epoch_s(d["timestamp_utc"])

        # Pull the true signed (dx_m, dy_m) offset to the real aircraft from
        # candidates.parquet -- labels.parquet only has the scalar nearest_*
        # distances, not direction, but the zoomed buffer-shape inset needs
        # to place the aircraft at its actual relative bearing, not a guess.
        cands = pd.read_parquet(ground_truth_candidates_path(SLUG, density))
        cands["t"] = epoch_s(cands["timestamp_utc"])
        merged = d.merge(
            cands[["drone_id", "t", "icao24", "dx_m", "dy_m"]],
            left_on=["drone_id", "t", "nearest_icao24"],
            right_on=["drone_id", "t", "icao24"],
            how="left",
        )

        for row in merged.itertuples(index=False):
            out.append({
                "drone_id": row.drone_id, "density": density, "t": int(row.t),
                "lat": round(float(row.latitude), 5), "lon": round(float(row.longitude), 5),
                "alt": round(float(row.altitude_m), 1),
                "icao": row.nearest_icao24,
                "h": round(float(row.nearest_horizontal_m), 1), "v": round(float(row.nearest_vertical_m), 1),
                "dx": round(float(row.dx_m), 1) if pd.notna(row.dx_m) else None,
                "dy": round(float(row.dy_m), 1) if pd.notna(row.dy_m) else None,
            })
    return out


def export_alert_counts(SLUG):
    counts = {}
    for shape in ["cylinder", "sphere", "ellipsoid"]:
        counts[shape] = {}
        for density in DRONE_DENSITY_LEVELS:
            df = pd.read_parquet(buffer_alerts_path(SLUG, density, shape), columns=["alert"])
            counts[shape][density] = int(df["alert"].sum())
    return counts


def export_region(region_name):
    slug = region_slug(region_name)
    bbox = REGIONS[region_name]

    aircraft, aircraft_quant = export_aircraft(slug)
    drones, drone_quant = export_drones(slug)
    danger = export_danger(slug)
    alerts = export_alert_counts(slug)

    all_t = [p[0] for a in aircraft for p in a["pts"]] + [d["t0"] for d in drones] + [d["t1"] for d in drones]
    t_min, t_max = min(all_t), max(all_t)

    payload = {
        "meta": {
            "region": region_name,
            "bbox": list(bbox),  # lamin, lomin, lamax, lomax
            "t_min": t_min, "t_max": t_max,
            "densityRates": DRONE_DENSITY_LEVELS,
            "shapes": {
                "cylinder": {"r": CYLINDER_HORIZONTAL_RADIUS_M, "h": CYLINDER_VERTICAL_HALF_HEIGHT_M},
                "sphere": {"r": SPHERE_RADIUS_M},
                "ellipsoid": {"a": ELLIPSOID_MAJOR_AXIS_M, "b": ELLIPSOID_MINOR_AXIS_M},
            },
            "alertCounts": alerts,
            "nAircraft": len(aircraft),
            "nDrones": {density: sum(1 for d in drones if d["d"] == density) for density in DRONE_DENSITY_LEVELS},
            "nDanger": len(danger),
            # Uniform affine quantization params (paper eq. 6-7), one [S,Z]
            # pair per field -- the client dequantizes (eq. 8) once at load.
            "quant": {"aircraft": aircraft_quant, "drones": drone_quant},
        },
        "aircraft": aircraft,
        "drones": drones,
        "danger": danger,
    }

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/viz_data_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    print(f"  aircraft: {len(aircraft)}, drones: {len(drones)}, danger events: {len(danger)}")
    print(f"  time window: {t_min} -> {t_max} ({(t_max - t_min) / 3600:.2f}h)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", action="append", default=None, help="Region name; repeat for multiple. Default: all regions with simulated data.")
    args = parser.parse_args()

    if args.region:
        regions = args.region
    else:
        regions = [r for r in REGIONS if REGIONS[r] and os.path.isdir(f"data/simulated/region={region_slug(r)}")]

    for region_name in regions:
        export_region(region_name)


if __name__ == "__main__":
    main()
