#!/usr/bin/env python3
"""One-time, idempotent backfill: convert a legacy per-day CSV (written by the
old main.py / opensky_multiday_logger.py) into the raw/clean Parquet layout.

Each source CSV's rows are grouped by UTC date, cleaned through the same
logic pipeline/step1_collect.py uses for live polls, written as a single raw
Parquet file per date (`migrated_from_csv.parquet`), then compacted into
data/clean/ immediately for all dates except one still-open "skip date" (the
day the live collector may still be actively polling for, so it can merge
naturally with new raw poll files once that day actually rolls over).

Usage (run as a module from the project root, so `pipeline` resolves):
    python3 -m scripts.migrate_csv_to_parquet --csv <path> --region-slug <slug> [--skip-date YYYY-MM-DD]
"""

import argparse
import sys
import warnings

import pandas as pd

from pipeline.clean import raw_csv_rows_to_clean_df
from pipeline.compact import compact_day_to_clean
from pipeline.config import CLEAN_SCHEMA
from pipeline.parquet_io import atomic_write_parquet, raw_migrated_path

_CSV_COLUMNS = [
    "query_timestamp_utc", "icao24", "callsign", "origin_country",
    "time_position", "last_contact", "longitude", "latitude",
    "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate",
    "sensors", "geo_altitude", "squawk", "spi", "position_source",
]


def migrate(csv_path, region_slug, skip_date=None):
    # dtype=str for everything: avoids pandas mis-inferring numeric-looking
    # columns like squawk (which can have meaningful leading zeros).
    df = pd.read_csv(csv_path, dtype=str)
    missing = set(_CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")

    ts = pd.to_datetime(df["query_timestamp_utc"], utc=False)
    if ts.dt.tz is None:
        warnings.warn(
            f"{csv_path}: timestamps have no UTC offset — assuming they are "
            "already UTC (matches the rest of the codebase's convention)."
        )
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["query_timestamp_utc"] = ts

    dates_written = []
    for date_str, day_df in df.groupby(ts.dt.strftime("%Y-%m-%d")):
        clean_df = raw_csv_rows_to_clean_df(day_df.reset_index(drop=True))
        if clean_df.empty:
            print(f"  {date_str}: 0 usable rows, skipping")
            continue
        path = raw_migrated_path(region_slug, date_str)
        atomic_write_parquet(clean_df, path, CLEAN_SCHEMA)
        print(f"  {date_str}: {len(clean_df)} rows -> {path}")
        dates_written.append(date_str)

    compacted = []
    for date_str in dates_written:
        if date_str == skip_date:
            print(f"  {date_str}: left uncompacted (skip-date, likely still in progress)")
            continue
        n = compact_day_to_clean(region_slug, date_str)
        print(f"  {date_str}: compacted -> {n} rows in data/clean/region={region_slug}/date={date_str}/data.parquet")
        compacted.append(date_str)

    return dates_written, compacted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the legacy CSV file")
    parser.add_argument("--region-slug", required=True, help="e.g. new_york, boston")
    parser.add_argument("--skip-date", default=None, help="YYYY-MM-DD date to leave uncompacted (still in progress)")
    args = parser.parse_args()

    print(f"Migrating {args.csv} (region={args.region_slug})")
    dates_written, compacted = migrate(args.csv, args.region_slug, args.skip_date)
    print(f"\nDone. {len(dates_written)} date(s) written to raw/, {len(compacted)} compacted to clean/.")


if __name__ == "__main__":
    sys.exit(main())
