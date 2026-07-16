#!/usr/bin/env python3
"""One-off export: today's clean ADS-B data (deduped across raw polls, same
logic as pipeline/compact.py) per region, written to separate CSVs. Today's
day isn't over yet, so it hasn't been compacted into data/clean/ by the live
collector -- this reads directly from data/raw/ without touching any
collector checkpoint state, so it's safe to run mid-collection.

For regions with no raw data for today (no collector currently running,
e.g. Boston), falls back to that region's most recent compacted clean date
and labels the output filename accordingly.
"""

import glob
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import CLEAN_SCHEMA, REGIONS, region_slug
from pipeline.parquet_io import raw_partition_dir

OUT_DIR = "data/results/csv_exports"


def dedup_day(files, schema=CLEAN_SCHEMA):
    tables = [pq.read_table(f, schema=schema) for f in files]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    has_time_position = df["time_position"].notna()
    with_tp = df[has_time_position].drop_duplicates(subset=["icao24", "time_position"], keep="last")
    without_tp = df[~has_time_position]
    df = pd.concat([with_tp, without_tp], ignore_index=True)
    return df.sort_values(["icao24", "query_timestamp_utc"]).reset_index(drop=True)


def export_today_or_latest(region_name, today_str):
    slug = region_slug(region_name)
    raw_dir = raw_partition_dir(slug, today_str)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))

    if files:
        df = dedup_day(files)
        label = today_str
        source = f"raw/ ({len(files)} polls so far today, not yet compacted)"
    else:
        clean_dates = sorted(glob.glob(os.path.join("data", "clean", f"region={slug}", "date=*")))
        if not clean_dates:
            print(f"  {region_name}: no data at all (no raw today, no clean history) -- skipped")
            return None
        latest_dir = clean_dates[-1]
        label = os.path.basename(latest_dir).split("=", 1)[1]
        df = pd.read_parquet(latest_dir)
        source = f"clean/ (no live collector today -- fell back to latest available date: {label})"

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{slug}_{label}.csv")
    df.to_csv(out_path, index=False)
    print(f"  {region_name}: {len(df)} rows -> {out_path}  [{source}]")
    return out_path


def main():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    regions = ["Boston", "New York", "San Francisco", "Los Angeles", "Miami",
               "Seattle", "Atlanta", "Chicago", "Dallas"]
    print(f"Exporting per-region CSVs (today = {today_str} UTC)\n")
    for region_name in regions:
        assert region_name in REGIONS, region_name
        export_today_or_latest(region_name, today_str)


if __name__ == "__main__":
    main()
