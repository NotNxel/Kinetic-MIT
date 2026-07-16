"""Cleaning logic shared by the live collector (pipeline/step1_collect.py) and
the one-time CSV backfill (scripts/migrate_csv_to_parquet.py), so cleaning
rules exist in exactly one place regardless of the data's source.
"""

import ast

import pandas as pd

from pipeline.config import CLEAN_SCHEMA

_RAW_STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


def states_payload_to_clean_df(payload, query_time, schema=CLEAN_SCHEMA):
    """Turn one raw OpenSky /states/all JSON payload into a cleaned DataFrame.

    `query_time` is a timezone-aware datetime for when the poll was made.
    """
    states = payload.get("states") or []
    rows = []
    for s in states:
        row = dict(zip(_RAW_STATE_FIELDS, s))
        row["query_timestamp_utc"] = query_time
        row["callsign"] = (row["callsign"] or "").strip() if row["callsign"] else None
        rows.append(row)
    df = pd.DataFrame(rows, columns=["query_timestamp_utc"] + _RAW_STATE_FIELDS)
    return _clean_dataframe(df, schema)


def raw_csv_rows_to_clean_df(df, schema=CLEAN_SCHEMA):
    """Clean a DataFrame read from one of the legacy per-day CSV files.

    Expects the same columns as `_RAW_STATE_FIELDS` plus `query_timestamp_utc`,
    all read as strings (see scripts/migrate_csv_to_parquet.py, which reads the
    CSV with dtype=str to avoid pandas mis-inferring numeric-looking columns
    like `squawk`).
    """
    df = df.copy()
    df["callsign"] = df["callsign"].where(df["callsign"].notna() & (df["callsign"].str.strip() != ""), None)
    df["callsign"] = df["callsign"].apply(lambda v: v.strip() if isinstance(v, str) else v)

    def _parse_sensors(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return None
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
                return list(parsed) if parsed else None
            except (ValueError, SyntaxError):
                return None
        return v

    df["sensors"] = df["sensors"].apply(_parse_sensors)
    return _clean_dataframe(df, schema)


def _clean_dataframe(df, schema):
    if df.empty:
        return pd.DataFrame({f.name: pd.Series(dtype=f.type.to_pandas_dtype() if not pa_is_nested(f.type) else object) for f in schema})

    df["query_timestamp_utc"] = pd.to_datetime(df["query_timestamp_utc"], utc=True)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["baro_altitude"] = pd.to_numeric(df["baro_altitude"], errors="coerce")
    df["geo_altitude"] = pd.to_numeric(df["geo_altitude"], errors="coerce")
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")
    df["true_track"] = pd.to_numeric(df["true_track"], errors="coerce")
    df["vertical_rate"] = pd.to_numeric(df["vertical_rate"], errors="coerce")
    df["time_position"] = pd.to_numeric(df["time_position"], errors="coerce").astype("Int64")
    df["last_contact"] = pd.to_numeric(df["last_contact"], errors="coerce").astype("Int64")
    df["position_source"] = pd.to_numeric(df["position_source"], errors="coerce").fillna(0).astype("int8")
    # NOT .astype(bool): when this column arrives as strings (the CSV
    # migration path), Python/pandas treat any non-empty string — including
    # the literal string "False" — as truthy, which would silently flip
    # every False row to True. Comparing against the string "True" instead
    # is correct for both a native-bool Series (live payload path, str(True)
    # == "True") and a string Series (CSV path).
    df["on_ground"] = df["on_ground"].astype(str) == "True"
    df["spi"] = df["spi"].astype(str) == "True"
    df["icao24"] = df["icao24"].astype(str)
    df["origin_country"] = df["origin_country"].astype(str)
    df["squawk"] = df["squawk"].where(df["squawk"].notna(), None)

    # No altitude filtering here — deferred to the analysis steps (4/5).
    # Only drop rows unusable for any geometry work: missing position.
    df = df.dropna(subset=["longitude", "latitude"])

    df = df.drop_duplicates(subset=["icao24", "query_timestamp_utc"], keep="last")
    df = df.sort_values(["icao24", "query_timestamp_utc"]).reset_index(drop=True)

    return df[[f.name for f in schema]]


def pa_is_nested(t):
    return str(t).startswith("list")
