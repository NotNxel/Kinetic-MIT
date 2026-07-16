#!/usr/bin/env python3
"""
OpenSky Network ADS-B Multi-Day Regional Collector — Step 1 of the drone
buffer-geometry research pipeline (30-60+ days)
================================================================

Polls the OpenSky Network REST API's `/states/all` endpoint for a chosen
region (bounding box) and writes cleaned, typed state vectors to a
Hive-partitioned Parquet data lake, continuously, over a long window
(e.g. 30-60 days).

WHY THIS IS DIFFERENT FROM A SIMPLE "run for N days" LOOP
-----------------------------------------------------------
A single Python process is NOT reliable for 30-60 days straight — laptops
sleep, processes get OOM-killed, networks blip, terminals close. So this
script is built to be:

  1. RESUMABLE — it writes a small JSON checkpoint file. If the process
     dies and you restart it, it picks up where it left off instead of
     restarting the whole 30/60-day window.
  2. RAW/CLEAN LAYERED — every poll writes one small immutable Parquet file
     to the raw/ layer (crash-safe: a mid-write crash can never corrupt a
     previously-completed poll). Once a day is over, its raw files are
     compacted into a single deduped, sorted Parquet file in the clean/
     layer — what steps 2-8 of the pipeline actually read.
  3. LOW-FREQUENCY BY DEFAULT — polling every 5 minutes instead of every
     minute, because your OpenSky credit budget has to last 30-60 days,
     not 24 hours.

HOW TO ACTUALLY KEEP IT RUNNING FOR 30-60 DAYS
------------------------------------------------
Run it on a machine that stays on (a cheap always-on Linux box, a
Raspberry Pi, or a small cloud VM), using one of:

  a) systemd (recommended on Linux):
       Create /etc/systemd/system/opensky-collector.service:

         [Unit]
         Description=OpenSky ADS-B collector
         After=network-online.target

         [Service]
         WorkingDirectory=/home/youruser/kinetic
         ExecStart=/usr/bin/python3 -m pipeline.step1_collect --auto
         Restart=always
         RestartSec=30
         Environment=OPENSKY_CLIENT_ID=xxx
         Environment=OPENSKY_CLIENT_SECRET=yyy

         [Install]
         WantedBy=multi-user.target

       Then: sudo systemctl enable --now opensky-collector

  b) A restart-loop wrapper (simplest, no systemd needed):

         #!/bin/bash
         while true; do
             python3 -m pipeline.step1_collect --auto
             echo "Collector exited, restarting in 30s..."
             sleep 30
         done

     Run that with: nohup ./run_collector.sh &

  On macOS, wrap either launch method in `caffeinate -i` to stop the OS from
  sleeping mid-run — a sleeping machine pauses the process, which does not
  lose data (raw files remain crash-safe) but does create large gaps in the
  polling cadence.

  Either way, because of the checkpoint file, restarts are safe — the
  script just resumes.

FOR REAL BULK HISTORICAL DATA (an alternative worth knowing about)
---------------------------------------------------------------------
If you don't strictly need to poll live for weeks, OpenSky also offers
an academic/research Trino database with full historical ADS-B history
you can query directly (no live polling required) — but it requires
applying for a research account. See:
https://opensky-network.org/data/apply
That's the "proper" way to get bulk historical months of data; this
script is the practical way to do it with just the public REST API.

Usage
-----
    python3 -m pipeline.step1_collect            # interactive setup
    python3 -m pipeline.step1_collect --auto      # resume/run using saved checkpoint,
                                                     # no prompts (for systemd/cron use)
"""

import argparse
import json
import os
import sys
import time
import signal
from datetime import datetime, timedelta, timezone

import requests

from pipeline.clean import states_payload_to_clean_df
from pipeline.compact import compact_day_to_clean, find_uncompacted_days
from pipeline.config import (
    CLEAN_SCHEMA,
    LEGACY_STATE_FILE,
    REGIONS,
    STATE_FILE,
    TOKEN_REFRESH_MARGIN,
    TOKEN_URL,
    STATES_URL,
    DEFAULT_POLL_INTERVAL_SECONDS,
    region_slug,
    state_file_for_region,
)
from pipeline.parquet_io import atomic_write_parquet, raw_poll_path

_stop = False


def _handle_sigterm(signum, frame):
    global _stop
    print("\nStop signal received — finishing current poll then exiting cleanly...")
    _stop = True


signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)


# --------------------------------------------------------------------------
# Setup / interactive prompts (skipped when resuming with --auto)
# --------------------------------------------------------------------------

def choose_region():
    print("\nWhich region do you want ADS-B data for?\n")
    names = list(REGIONS.keys())
    for i, name in enumerate(names, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input(f"\nEnter a number (1-{len(names)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            break
        print("Invalid choice, try again.")

    bbox = REGIONS[name]
    if bbox is None:
        print("\nEnter a custom bounding box (WGS84 decimal degrees).")
        lamin = float(input("  min latitude  (lamin): ").strip())
        lomin = float(input("  min longitude (lomin): ").strip())
        lamax = float(input("  max latitude  (lamax): ").strip())
        lomax = float(input("  max longitude (lomax): ").strip())
        bbox = (lamin, lomin, lamax, lomax)
        name = "Custom"

    return name, bbox


def choose_days():
    while True:
        raw = input("\nHow many days should this run for? (e.g. 30, 45, 60): ").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Enter a positive whole number.")


def choose_poll_interval():
    raw = input(
        f"\nPoll interval in seconds [default {DEFAULT_POLL_INTERVAL_SECONDS} = 5 min]: "
    ).strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_POLL_INTERVAL_SECONDS


def get_credentials():
    client_id = os.environ.get("OPENSKY_CLIENT_ID")
    client_secret = os.environ.get("OPENSKY_CLIENT_SECRET")

    if not client_id and sys.stdin.isatty():
        client_id = input("\nOpenSky client_id: ").strip()
    if not client_secret and sys.stdin.isatty():
        client_secret = input("OpenSky client_secret: ").strip()

    if not client_id or not client_secret:
        print(
            "\nNo credentials found — continuing WITHOUT authentication.\n"
            "You'll be limited to OpenSky's anonymous rate limits, which "
            "is NOT recommended for a 30-60 day run. Set OPENSKY_CLIENT_ID "
            "and OPENSKY_CLIENT_SECRET env vars for best results."
        )
        return None, None

    return client_id, client_secret


# --------------------------------------------------------------------------
# Checkpoint state
# --------------------------------------------------------------------------

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)

    if state_file == STATE_FILE and os.path.exists(LEGACY_STATE_FILE):
        print(f"Migrating legacy checkpoint {LEGACY_STATE_FILE} -> {STATE_FILE}")
        with open(LEGACY_STATE_FILE) as f:
            state = json.load(f)
        state.setdefault("compacted_dates", [])
        state["schema_version"] = 2
        save_state(state, state_file)
        return state

    return None


def save_state(state, state_file):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_file)


def build_new_state(region_name, bbox, days, poll_interval):
    now = datetime.now(timezone.utc)
    return {
        "region_name": region_name,
        "bbox": list(bbox),
        "start_ts": now.isoformat(),
        "end_ts": (now + timedelta(days=days)).isoformat(),
        "poll_interval_seconds": poll_interval,
        "total_rows_written": 0,
        "total_polls": 0,
        "last_poll_ts": None,
        "compacted_dates": [],
        "schema_version": 2,
    }


# --------------------------------------------------------------------------
# OAuth token
# --------------------------------------------------------------------------

class TokenManager:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = None

    def get_token(self):
        if not self.client_id:
            return None
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self):
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(
            seconds=expires_in - TOKEN_REFRESH_MARGIN
        )
        return self.token


# --------------------------------------------------------------------------
# Fetch / write
# --------------------------------------------------------------------------

def fetch_states(bbox, token):
    lamin, lomin, lamax, lomax = bbox
    params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(STATES_URL, params=params, headers=headers, timeout=20)
    if resp.status_code == 429:
        raise requests.exceptions.HTTPError("429 Too Many Requests (rate limited)")
    resp.raise_for_status()
    return resp.json()


def write_poll(region_slug_, payload, query_time):
    """Clean one poll's payload and write it as one immutable raw Parquet
    file. Returns the number of aircraft rows written."""
    df = states_payload_to_clean_df(payload, query_time)
    if df.empty:
        return 0
    date_str = query_time.strftime("%Y-%m-%d")
    path = raw_poll_path(region_slug_, date_str, query_time.isoformat())
    atomic_write_parquet(df, path, CLEAN_SCHEMA)
    return len(df)


def compact_completed_days(region_slug_, state, state_file):
    """Compact every past, not-yet-compacted day for this region. Covers
    both the normal end-of-day rollover and multi-day outages/crashes."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for date_str in find_uncompacted_days(region_slug_, state["compacted_dates"], today_str):
        n = compact_day_to_clean(region_slug_, date_str)
        state["compacted_dates"].append(date_str)
        print(f"  Compacted {date_str}: {n} rows -> data/clean/region={region_slug_}/date={date_str}/data.parquet")
    save_state(state, state_file)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Resume/run using saved checkpoint, no prompts")
    parser.add_argument("--region", default=None, help="Region name from pipeline.config.REGIONS (e.g. 'San Francisco'). Required with --auto for a region with no existing checkpoint.")
    parser.add_argument("--days", type=int, default=60, help="Days to run for, when auto-creating a new checkpoint non-interactively")
    parser.add_argument("--poll-interval", type=int, default=None, help="Poll interval in seconds, when auto-creating a new checkpoint non-interactively")
    args, _unknown = parser.parse_known_args()

    auto_mode = args.auto

    if args.region is not None:
        if args.region not in REGIONS or REGIONS[args.region] is None:
            print(f"Unknown region: {args.region!r}. Choices: {[r for r in REGIONS if REGIONS[r]]}")
            sys.exit(1)
        state_file = state_file_for_region(region_slug(args.region))
    else:
        state_file = STATE_FILE

    state = load_state(state_file)

    if state is None:
        if auto_mode:
            if args.region is None:
                print(
                    f"No checkpoint file ({state_file}) found and --auto was passed "
                    "without --region. Run once interactively first to configure "
                    "region/days, or pass --region to auto-create a checkpoint."
                )
                sys.exit(1)
            poll_interval = args.poll_interval or DEFAULT_POLL_INTERVAL_SECONDS
            state = build_new_state(args.region, REGIONS[args.region], args.days, poll_interval)
            save_state(state, state_file)
            print(f"Auto-created new checkpoint for '{args.region}' at {state_file} "
                  f"({args.days}d, poll every {poll_interval}s)")
        else:
            region_name, bbox = choose_region()
            days = choose_days()
            poll_interval = choose_poll_interval()
            state = build_new_state(region_name, bbox, days, poll_interval)
            state_file = state_file_for_region(region_slug(region_name))
            save_state(state, state_file)
    else:
        print(f"Resuming existing run from checkpoint ({state_file}):")
        print(f"  Region: {state['region_name']}  bbox={tuple(state['bbox'])}")
        print(f"  Window: {state['start_ts']}  ->  {state['end_ts']}")
        print(f"  Progress so far: {state['total_polls']} polls, "
              f"{state['total_rows_written']} rows written")

    client_id, client_secret = get_credentials()
    token_mgr = TokenManager(client_id, client_secret)

    region_name = state["region_name"]
    slug = region_slug(region_name)
    bbox = tuple(state["bbox"])
    end_time = datetime.fromisoformat(state["end_ts"])
    start_time = datetime.fromisoformat(state["start_ts"])
    poll_interval = state["poll_interval_seconds"]

    print(f"\nCollecting ADS-B data for '{region_name}' bbox={bbox}")
    print(f"Polling every {poll_interval}s until {end_time.isoformat()}")
    print(f"Writing raw Parquet to: data/raw/region={slug}/, clean to: data/clean/region={slug}/")
    print("Safe to stop (Ctrl+C) and restart any time — progress is checkpointed.\n")

    # Startup self-heal: compact any day left over from a crash/outage.
    compact_completed_days(slug, state, state_file)

    while not _stop:
        now = datetime.now(timezone.utc)
        if now >= end_time:
            print(f"\nReached end of {(end_time - start_time).days}-day window. Done.")
            break

        poll_start = time.monotonic()
        try:
            compact_completed_days(slug, state, state_file)
            token = token_mgr.get_token()
            payload = fetch_states(bbox, token)
            n = write_poll(slug, payload, now)

            state["total_rows_written"] += n
            state["total_polls"] += 1
            state["last_poll_ts"] = now.isoformat()
            save_state(state, state_file)

            remaining = end_time - now
            print(
                f"[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                f"poll #{state['total_polls']}: {n} aircraft "
                f"(remaining: {remaining.days}d {remaining.seconds // 3600}h)"
            )
        except requests.exceptions.HTTPError as e:
            print(f"  Request error: {e} — backing off 60s")
            time.sleep(60)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e} — retrying in 30s")
            time.sleep(30)
            continue

        elapsed_poll = time.monotonic() - poll_start
        sleep_for = max(0, poll_interval - elapsed_poll)
        # Sleep in short chunks so Ctrl+C / SIGTERM is responsive
        slept = 0
        while slept < sleep_for and not _stop:
            chunk = min(5, sleep_for - slept)
            time.sleep(chunk)
            slept += chunk

    print(
        f"\nTotal so far: {state['total_polls']} polls, "
        f"{state['total_rows_written']} rows across all days. "
        f"Re-run this script (or the --auto wrapper) any time to resume."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
