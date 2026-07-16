"""Compaction: consolidate a completed day's raw poll files into one clean,
deduped, sorted Parquet file per region-day. This is the only thing that
writes to the clean/ layer.
"""

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import CLEAN_SCHEMA
from pipeline.parquet_io import atomic_write_parquet, clean_partition_path, raw_partition_dir


def find_uncompacted_days(region_slug, compacted_dates, today_str):
    """Return sorted date strings under raw/ for this region that are strictly
    before today and not yet in compacted_dates."""
    pattern = os.path.join("data", "raw", f"region={region_slug}", "date=*")
    days = []
    for path in glob.glob(pattern):
        date_str = os.path.basename(path).split("=", 1)[1]
        if date_str < today_str and date_str not in compacted_dates:
            days.append(date_str)
    return sorted(days)


def compact_day_to_clean(region_slug, date_str, schema=CLEAN_SCHEMA):
    """Read every raw poll file for one region-day, cross-poll dedup, sort,
    and atomically write a single clean Parquet file. Returns the row count
    written (0 if there was no raw data for that day)."""
    raw_dir = raw_partition_dir(region_slug, date_str)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        return 0

    tables = [pq.read_table(f, schema=schema) for f in files]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)

    # Cross-poll dedup: a repeated time_position for the same aircraft means
    # the same underlying ADS-B report was re-observed across polls, not a
    # new position — collapse it. Rows with a null time_position (transponder
    # didn't report one) are left as-is beyond the per-poll dedup already
    # applied in pipeline/clean.py.
    has_time_position = df["time_position"].notna()
    with_tp = df[has_time_position].drop_duplicates(subset=["icao24", "time_position"], keep="last")
    without_tp = df[~has_time_position]
    df = pd.concat([with_tp, without_tp], ignore_index=True)
    df = df.sort_values(["icao24", "query_timestamp_utc"]).reset_index(drop=True)

    atomic_write_parquet(df, clean_partition_path(region_slug, date_str), schema)
    return len(df)
