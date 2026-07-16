"""Shared configuration for the drone-buffer-geometry research pipeline.

Step 1 (collection/cleaning) uses this today; steps 2-8 (simulation, ground-truth
labeling, buffer geometries, ROC/statistics) will import REGIONS and the data
paths from here too, so region bounding boxes and directory layout stay defined
in exactly one place.
"""

import math
import os

import pyarrow as pa

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)
STATES_URL = "https://opensky-network.org/api/states/all"

DEFAULT_POLL_INTERVAL_SECONDS = 300  # 5 minutes — safer for long, multi-day runs
TOKEN_REFRESH_MARGIN = 30  # refresh token this many seconds early

REGIONS = {
    "Boston":        (41.9, -71.6, 42.7, -70.5),
    "New York":      (40.3, -74.6, 41.2, -73.3),
    "Los Angeles":   (33.5, -118.9, 34.5, -117.5),
    "Chicago":       (41.5, -88.3, 42.3, -87.3),
    "San Francisco": (37.2, -122.7, 38.1, -121.7),
    "Seattle":       (47.2, -122.8, 48.0, -121.8),
    "Miami":         (25.4, -80.6, 26.3, -80.0),
    "Dallas":        (32.4, -97.4, 33.2, -96.3),
    "Atlanta":       (33.3, -84.9, 34.1, -83.9),
    "Denver":        (39.4, -105.4, 40.2, -104.3),
    "London":        (51.1, -0.9, 51.9, 0.6),
    "Custom (enter your own bounding box)": None,
}

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
SIMULATED_DIR = os.path.join(DATA_DIR, "simulated")
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, "ground_truth")
BUFFERS_DIR = os.path.join(DATA_DIR, "buffers")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
STATE_DIR = os.path.join(DATA_DIR, "state")
STATE_FILE = os.path.join(DATA_DIR, "opensky_logger_state.json")  # pre-multi-region location, single collector only
LEGACY_STATE_FILE = "opensky_logger_state.json"  # pre-upgrade location, at repo root


def state_file_for_region(region_slug_):
    """Per-region checkpoint path, so multiple collector processes (one per
    city) can each track their own progress without clobbering each other's
    state -- STATE_FILE above only ever supported a single concurrent region."""
    return os.path.join(STATE_DIR, f"{region_slug_}.json")

# Pinned schema for every raw AND clean Parquet write. Pinning this once avoids
# pyarrow inferring a different dtype per file (e.g. an all-null column in a
# poll with zero aircraft), which would otherwise silently break merges across
# thousands of raw poll files at compaction time.
CLEAN_SCHEMA = pa.schema([
    pa.field("query_timestamp_utc", pa.timestamp("us", tz="UTC")),
    pa.field("icao24", pa.string()),
    pa.field("callsign", pa.string()),
    pa.field("origin_country", pa.string()),
    pa.field("time_position", pa.int64()),
    pa.field("last_contact", pa.int64()),
    pa.field("longitude", pa.float64()),
    pa.field("latitude", pa.float64()),
    pa.field("baro_altitude", pa.float64()),
    pa.field("on_ground", pa.bool_()),
    pa.field("velocity", pa.float64()),
    pa.field("true_track", pa.float64()),
    pa.field("vertical_rate", pa.float64()),
    pa.field("sensors", pa.list_(pa.int64())),
    pa.field("geo_altitude", pa.float64()),
    pa.field("squawk", pa.string()),
    pa.field("spi", pa.bool_()),
    pa.field("position_source", pa.int8()),
])


def region_slug(region_name):
    return region_name.lower().replace(" ", "_")


def utm_epsg_for_bbox(bbox):
    """UTM EPSG code for the bbox's centroid. bbox is (lamin, lomin, lamax, lomax)."""
    lamin, lomin, lamax, lomax = bbox
    lat_c = (lamin + lamax) / 2.0
    lon_c = (lomin + lomax) / 2.0
    zone = int((lon_c + 180) // 6) + 1
    return (32600 if lat_c >= 0 else 32700) + zone


# --------------------------------------------------------------------------
# Step 2: drone flight simulation
# --------------------------------------------------------------------------

DRONE_SCHEMA = pa.schema([
    pa.field("drone_id", pa.string()),
    pa.field("density_level", pa.string()),
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("altitude_m", pa.float64()),
    pa.field("heading_deg", pa.float64()),
    pa.field("speed_mps", pa.float64()),
])

DRONE_DENSITY_LEVELS = {"sparse": 10, "medium": 50, "dense": 200}  # drones/hour
DRONE_SPEED_RANGE_MPS = (5.0, 15.0)
DRONE_ALTITUDE_RANGE_M = (10.0, 121.92)  # 400 ft AGL ceiling (FAA Part 107)
DRONE_DURATION_RANGE_S = (300, 1500)  # 5-25 min, pre-distance-cap
DRONE_MAX_DISTANCE_M = 8000.0  # caps flight distance to ~1.5-8 km
DRONE_SIM_SEED = 42


# --------------------------------------------------------------------------
# Step 3: ground-truth danger labeling
# --------------------------------------------------------------------------

AIRCRAFT_GAP_CUTOFF_S = 180.0  # don't dead-reckon across gaps longer than this
DANGER_HORIZONTAL_M = 200.0  # FAA/ICAO separation minima
DANGER_VERTICAL_M = 50.0
CANDIDATE_SEARCH_RADIUS_M = 330.0  # covers Step 7's +/-20% sweep (240m) AND
# Step 6's 300m ellipsoid reach, with a 10% margin on the larger of the two

GROUND_TRUTH_LABELS_SCHEMA = pa.schema([
    pa.field("drone_id", pa.string()),
    pa.field("density_level", pa.string()),
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("altitude_m", pa.float64()),
    pa.field("danger", pa.bool_()),
    pa.field("n_candidates", pa.int16()),
    pa.field("nearest_icao24", pa.string()),
    pa.field("nearest_horizontal_m", pa.float64()),
    pa.field("nearest_vertical_m", pa.float64()),
])

GROUND_TRUTH_CANDIDATES_SCHEMA = pa.schema([
    pa.field("drone_id", pa.string()),
    pa.field("density_level", pa.string()),
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
    pa.field("icao24", pa.string()),
    pa.field("dx_m", pa.float64()),  # aircraft x - drone x, projected UTM meters
    pa.field("dy_m", pa.float64()),  # aircraft y - drone y, projected UTM meters
    pa.field("horizontal_m", pa.float64()),
    pa.field("vertical_m", pa.float64()),
])


# --------------------------------------------------------------------------
# Steps 4-6: buffer geometries (cylinder, sphere, ellipsoid)
# --------------------------------------------------------------------------

# The cylinder buffer uses the exact same numbers as the ground-truth danger
# threshold — it IS the FAA standard ground truth was defined from, so its
# alerts should almost exactly reproduce Step 3's danger labels. Aliased
# (not duplicated magic numbers) so that equivalence is explicit in code.
CYLINDER_HORIZONTAL_RADIUS_M = DANGER_HORIZONTAL_M
CYLINDER_VERTICAL_HALF_HEIGHT_M = DANGER_VERTICAL_M  # +/-50m => 100m total height

# Shared volume all three buffer shapes must match.
BUFFER_VOLUME_M3 = math.pi * CYLINDER_HORIZONTAL_RADIUS_M ** 2 * (2 * CYLINDER_VERTICAL_HALF_HEIGHT_M)

# Sphere: same volume, single radius. V = (4/3)pi r^3 => r = (3V/4pi)^(1/3).
SPHERE_RADIUS_M = (3 * BUFFER_VOLUME_M3 / (4 * math.pi)) ** (1 / 3)  # ~144.225m

# Ellipsoid: same volume, prolate spheroid stretched along heading.
# Aspect ratio a:b = 3:1 is a stated modeling choice (no cited standard gives
# a number) -- documented as an assumption in the Step 8 paper. Solving
# V = (4/3)pi*a*b^2 with a=3b gives clean values: b=100m, a=300m.
ELLIPSOID_ASPECT_RATIO = 3.0
# V = (4/3)pi*a*b^2 with a=ratio*b => b^3 = 3V/(4*pi*ratio)
ELLIPSOID_MINOR_AXIS_M = (3 * BUFFER_VOLUME_M3 / (4 * math.pi * ELLIPSOID_ASPECT_RATIO)) ** (1 / 3)  # ~100m
ELLIPSOID_MAJOR_AXIS_M = ELLIPSOID_ASPECT_RATIO * ELLIPSOID_MINOR_AXIS_M  # ~300m

BUFFER_ALERTS_SCHEMA = pa.schema([
    pa.field("drone_id", pa.string()),
    pa.field("density_level", pa.string()),
    pa.field("timestamp_utc", pa.timestamp("us", tz="UTC")),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
    pa.field("altitude_m", pa.float64()),
    pa.field("alert", pa.bool_()),
    pa.field("n_candidates", pa.int16()),
    pa.field("nearest_icao24", pa.string()),
    pa.field("nearest_horizontal_m", pa.float64()),
    pa.field("nearest_vertical_m", pa.float64()),
])
