#!/usr/bin/env python3
"""One-off: build per-region {bbox, rings} coastline JSON files for the web
viz from the cached Natural Earth 10m land polygons (data/geo_cache/
ne_10m_land.geojson), the same source/format already used for Boston and
New York. Pure-Python Sutherland-Hodgman clip, no shapely -- clips every
global land ring down to just the points inside (plus edge-intersection
points on) each region's bounding box.
"""

import json
import os

from pipeline.config import REGIONS, region_slug

SRC = "data/geo_cache/ne_10m_land.geojson"
OUT_DIR = "data/results"

# Rough continental-US envelope, used only to pre-filter the ~100k-point
# global dataset down to something clip_ring doesn't have to chew through
# entirely for every one of the 9 regions.
US_ENVELOPE = (18.0, -170.0, 72.0, -65.0)  # lamin, lomin, lamax, lomax


def clip_ring(ring, bbox):
    lamin, lomin, lamax, lomax = bbox

    def inside(p, edge):
        x, y = p
        if edge == "left":
            return x >= lomin
        if edge == "right":
            return x <= lomax
        if edge == "bottom":
            return y >= lamin
        return y <= lamax

    def intersect(a, b, edge):
        x1, y1 = a
        x2, y2 = b
        if edge in ("left", "right"):
            xe = lomin if edge == "left" else lomax
            t = (xe - x1) / (x2 - x1) if x2 != x1 else 0
            return [xe, y1 + t * (y2 - y1)]
        ye = lamin if edge == "bottom" else lamax
        t = (ye - y1) / (y2 - y1) if y2 != y1 else 0
        return [x1 + t * (x2 - x1), ye]

    poly = ring
    for edge in ["left", "right", "bottom", "top"]:
        if not poly:
            break
        out = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i - 1]
            cur_in, prev_in = inside(cur, edge), inside(prev, edge)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur, edge))
        poly = out
    return poly


def ring_bbox_overlaps(ring, bbox):
    lamin, lomin, lamax, lomax = bbox
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return not (max(lons) < lomin or min(lons) > lomax or max(lats) < lamin or min(lats) > lamax)


def main():
    print(f"Loading {SRC} ...")
    with open(SRC) as f:
        land = json.load(f)

    # Pre-filter to rings that touch the continental US envelope, across all
    # features/geometry types (Polygon and MultiPolygon).
    us_rings = []
    for feat in land["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                if ring_bbox_overlaps(ring, US_ENVELOPE):
                    us_rings.append(ring)
    print(f"  {len(us_rings)} land rings overlap the continental US envelope")

    regions = ["Boston", "New York", "San Francisco", "Los Angeles", "Miami",
               "Seattle", "Atlanta", "Chicago", "Dallas"]
    for region_name in regions:
        bbox = REGIONS[region_name]
        slug = region_slug(region_name)
        rings = []
        for ring in us_rings:
            clipped = clip_ring(ring, bbox)
            if len(clipped) >= 3:
                rings.append(clipped)

        out_path = os.path.join(OUT_DIR, f"coastline_{slug}.json")
        with open(out_path, "w") as f:
            json.dump({"bbox": list(bbox), "rings": rings}, f, separators=(",", ":"))
        npts = sum(len(r) for r in rings)
        size_kb = os.path.getsize(out_path) / 1e3
        print(f"  {region_name}: {len(rings)} rings, {npts} pts -> {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
