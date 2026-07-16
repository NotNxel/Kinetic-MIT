"""Atomic Parquet writes and Hive-style path helpers for the raw/clean layers.

Raw layer: one immutable file per poll, written atomically (tmp + os.replace,
same pattern as the existing checkpoint's save_state()) so a crash mid-write
can never corrupt a previously-completed poll.

Clean layer: one file per region-day, produced only by compaction
(pipeline/compact.py) — never written incrementally.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import BUFFERS_DIR, CLEAN_DIR, GROUND_TRUTH_DIR, RAW_DIR, RESULTS_DIR, SIMULATED_DIR


def atomic_write_parquet(df, path, schema):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, path)


def raw_partition_dir(region_slug, date_str):
    return os.path.join(RAW_DIR, f"region={region_slug}", f"date={date_str}")


def raw_poll_path(region_slug, date_str, poll_ts_str):
    safe_ts = poll_ts_str.replace(":", "-")
    return os.path.join(raw_partition_dir(region_slug, date_str), f"{safe_ts}.parquet")


def raw_migrated_path(region_slug, date_str):
    return os.path.join(raw_partition_dir(region_slug, date_str), "migrated_from_csv.parquet")


def clean_partition_dir(region_slug, date_str):
    return os.path.join(CLEAN_DIR, f"region={region_slug}", f"date={date_str}")


def clean_partition_path(region_slug, date_str):
    return os.path.join(clean_partition_dir(region_slug, date_str), "data.parquet")


def simulated_partition_dir(region_slug, density):
    return os.path.join(SIMULATED_DIR, f"region={region_slug}", f"density={density}")


def simulated_partition_path(region_slug, density):
    return os.path.join(simulated_partition_dir(region_slug, density), "drones.parquet")


def ground_truth_partition_dir(region_slug, density):
    return os.path.join(GROUND_TRUTH_DIR, f"region={region_slug}", f"density={density}")


def ground_truth_labels_path(region_slug, density):
    return os.path.join(ground_truth_partition_dir(region_slug, density), "labels.parquet")


def ground_truth_candidates_path(region_slug, density):
    return os.path.join(ground_truth_partition_dir(region_slug, density), "candidates.parquet")


def buffer_partition_dir(region_slug, density, shape):
    return os.path.join(BUFFERS_DIR, f"region={region_slug}", f"density={density}", f"shape={shape}")


def buffer_alerts_path(region_slug, density, shape):
    return os.path.join(buffer_partition_dir(region_slug, density, shape), "alerts.parquet")


def results_dir(region_slug):
    return os.path.join(RESULTS_DIR, f"region={region_slug}")
