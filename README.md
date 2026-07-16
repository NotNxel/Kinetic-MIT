# Kinetic — Live US Airspace Reconstruction

A drone-vs-aircraft collision-buffer-geometry research project: live ADS-B
collection, synthetic drone simulation, FAA/ICAO ground-truth danger
labeling, and a comparison of three equal-volume buffer shapes (cylinder,
sphere, ellipsoid), visualized on an interactive map across nine U.S. cities.

## Host it (this is all you need)

**`index.html`** is the entire website — self-contained, no build step, no
external requests. Push this repo to GitHub and enable **Pages** (Settings →
Pages → set source to this branch/root). It'll serve `index.html`
automatically. That's it — nothing else in this repo is required for the
site to work.

To preview locally: just open `index.html` in a browser.

## What else is in here

- **`pipeline/`** — the Python research pipeline: OpenSky ADS-B live
  collector, drone-flight simulator, ground-truth danger labeling, and the
  three buffer-geometry evaluators (cylinder/sphere/ellipsoid).
- **`scripts/`** — one-off exporters: `export_viz_data.py` (packages
  pipeline output into the compact JSON embedded in `index.html`) and
  `build_coastlines.py` (clips Natural Earth coastline data per city).
- **`data/raw/`, `data/clean/`** — the real ADS-B data actually collected
  live from OpenSky for each city (small — this is the raw material, not
  the multi-hundred-MB simulated/labeled intermediate output the pipeline
  generates when you run it).
- **`data/results/`** — the compact per-city JSON (`viz_data_*.json`,
  `coastline_*.json`) that gets embedded into `index.html`; this is the
  same data already baked into the website, included here for reference.
- **`requirements.txt`** — Python dependencies for the pipeline.

## Reproducing the full pipeline

The heavier intermediate outputs (simulated drone flights, ground-truth
labels, buffer-shape alerts — roughly 1GB) aren't included here since
they're fully regeneratable and not needed to host the site. To rebuild
them from the raw data in `data/raw`/`data/clean`:

```
pip install -r requirements.txt
python3 -m pipeline.step2_simulate_drones --region "Boston"
python3 -m pipeline.step3_ground_truth_labeling --region "Boston"
python3 -m pipeline.step4_buffer_cylinder --region "Boston"
python3 -m pipeline.step5_buffer_sphere --region "Boston"
python3 -m pipeline.step6_buffer_ellipsoid --region "Boston"
python3 -m pipeline.step7_statistical_validation --region "Boston"
python3 -m scripts.export_viz_data --region "Boston"
```

(repeat `--region` for the other 8 cities: New York, San Francisco, Los
Angeles, Miami, Seattle, Atlanta, Chicago, Dallas)

## What's real vs. simulated

- Real aircraft positions (ICAO24, callsign, lat/lon/altitude) come from
  live OpenSky ADS-B data, real planes, real timestamps — but only while
  each plane is inside its city's small tracked area.
- Drone traffic is 100% synthetic, generated for the research.
- Once a real plane leaves its tracked area, the website continues its path
  as a simulated straight-line flight to another tracked city (clearly
  marked "(simulated)" in the UI) — there's no nationwide ADS-B feed behind
  that part.
