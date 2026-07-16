#!/usr/bin/env python3
"""Step 8 (part 1) — generate the ROC comparison figure from Step 7's results.

Reads data/results/region=<slug>/roc_summary.parquet and plots TPR vs FPR
for the three buffer shapes, one panel per drone density. Sparse/medium have
no ground-truth DANGER events (Step 3), so TPR is undefined there and only
FPR is plotted meaningfully; dense is the only panel with a real TPR axis.

Usage:
    python3 -m pipeline.step8_paper.generate_figures [--region Boston]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.config import DRONE_DENSITY_LEVELS, REGIONS, region_slug
from pipeline.parquet_io import results_dir

# Fixed categorical order (validated: node scripts/validate_palette.js
# "#2a78d6,#1baf7a,#eda100" --mode light -> ALL CHECKS PASS, contrast WARN
# mitigated by the direct labels + black marker edges used below).
SHAPE_COLORS = {"cylinder": "#2a78d6", "sphere": "#1baf7a", "ellipsoid": "#eda100"}
SHAPE_MARKERS = {"cylinder": "o", "sphere": "s", "ellipsoid": "^"}
SHAPE_ORDER = ["cylinder", "sphere", "ellipsoid"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#8a8a86"


def _empty_panel(ax, density, n_danger):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dddddb")
    ax.text(
        0.5, 0.55, "No ground-truth danger events",
        ha="center", va="center", fontsize=10.5, color=TEXT_PRIMARY,
    )
    ax.text(
        0.5, 0.42, "TPR undefined — FPR = 0 for all three shapes",
        ha="center", va="center", fontsize=9, color=TEXT_MUTED,
    )
    ax.set_title(f"{density} ({DRONE_DENSITY_LEVELS[density]}/hr)\n{n_danger} ground-truth danger events", fontsize=11)


def _dense_panel(ax, sub, density, n_danger):
    # FPR values here are ~1e-6 scale (rare-event regime) -- expressing as
    # false positives per million drone-seconds keeps the axis human-readable
    # instead of collapsing every point onto x=0 at a linear 0-1 scale.
    sub = sub.set_index("shape")
    ax.plot([0, 3], [0, 0.03], linestyle="--", color="#dddddb", linewidth=1, zorder=0)

    for shape in SHAPE_ORDER:
        row = sub.loc[shape]
        x = row["FPR"] * 1_000_000
        y = row["TPR"]
        ax.scatter(
            [x], [y], s=170, color=SHAPE_COLORS[shape], marker=SHAPE_MARKERS[shape],
            edgecolor="black", linewidth=0.8, zorder=3, label=shape,
        )
        ax.annotate(
            shape, (x, y), textcoords="offset points", xytext=(9, 7),
            fontsize=9.5, color=TEXT_PRIMARY,
        )

    ax.set_xlim(-0.15, max(2.8, sub["FPR"].max() * 1_000_000 * 1.25))
    ax.set_ylim(0.55, 1.05)
    ax.set_xlabel("False positives per million drone-seconds")
    ax.set_title(f"{density} ({DRONE_DENSITY_LEVELS[density]}/hr)\n{n_danger} ground-truth danger events", fontsize=11)


def plot_roc(roc_df, out_path):
    densities = list(DRONE_DENSITY_LEVELS.keys())
    fig, axes = plt.subplots(1, len(densities), figsize=(4.4 * len(densities), 4.6), sharey=False)

    for ax, density in zip(axes, densities):
        sub = roc_df[roc_df["density"] == density]
        n_danger = int(sub.iloc[0]["TP"] + sub.iloc[0]["FN"]) if len(sub) else 0
        if sub["TPR"].isna().all():
            _empty_panel(ax, density, n_danger)
        else:
            _dense_panel(ax, sub, density, n_danger)

    axes[0].set_ylabel("True Positive Rate")
    handles = [
        plt.Line2D([0], [0], marker=SHAPE_MARKERS[s], color="w", markerfacecolor=SHAPE_COLORS[s],
                   markeredgecolor="black", markersize=9, label=s)
        for s in SHAPE_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Buffer Shape Comparison: TPR vs FPR by Drone Traffic Density", y=1.03, fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    out_dir = results_dir(slug)
    roc_df = pd.read_parquet(os.path.join(out_dir, "roc_summary.parquet"))

    out_path = os.path.join(out_dir, "roc_plot.png")
    plot_roc(roc_df, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
