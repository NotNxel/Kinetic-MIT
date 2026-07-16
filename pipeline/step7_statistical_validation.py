#!/usr/bin/env python3
"""Step 7 — Experiment, ROC Analysis, and Statistical Validation.

For each buffer shape (steps 4-6) and each drone density (Step 2), merges
that shape's alerts against Step 3's ground-truth danger labels and computes
the confusion matrix (TP/FP/TN/FN), TPR, and FPR — one (FPR, TPR) point per
shape per density, since each shape has a single fixed size rather than a
swept threshold. The shape closest to the ROC top-left corner (0,1) wins.

Sparse and medium have zero ground-truth DANGER events (see Step 3), so
TP+FN=0 and TPR is undefined (NaN) there — only FPR is meaningful for those
two; dense (42 danger events) is the only density where TPR is computable.
This is reported honestly rather than papered over.

Statistical validation:
  - McNemar's exact test, pairwise between shapes per density, on whether
    each shape's per-drone-second correctness (alert == danger) disagrees
    from another's asymmetrically.
  - Bootstrap 95% CIs via a cluster bootstrap at the drone level (not the
    drone-second level, which would understate variance given how
    correlated seconds within one flight are), 1000 resamples.
  - Sensitivity analysis: ground-truth threshold shifted +/-20%, recomputed
    exactly from Step 3's candidates.parquet (no spatial-join rerun needed),
    checking whether the shape ranking holds.

Usage:
    python3 -m pipeline.step7_statistical_validation [--region Boston]
"""

import argparse
import itertools
import math
import os

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

from pipeline.config import (
    DANGER_HORIZONTAL_M,
    DANGER_VERTICAL_M,
    DRONE_DENSITY_LEVELS,
    REGIONS,
    region_slug,
)
from pipeline.parquet_io import (
    buffer_alerts_path,
    ground_truth_candidates_path,
    ground_truth_labels_path,
    results_dir,
    simulated_partition_path,
)

SHAPES = ["cylinder", "sphere", "ellipsoid"]


def _load_merged(slug, density, shape):
    labels_df = pd.read_parquet(ground_truth_labels_path(slug, density), columns=["drone_id", "timestamp_utc", "danger"])
    alerts_df = pd.read_parquet(buffer_alerts_path(slug, density, shape), columns=["drone_id", "timestamp_utc", "alert"])
    merged = alerts_df.merge(labels_df, on=["drone_id", "timestamp_utc"], how="inner")
    assert len(merged) == len(alerts_df) == len(labels_df), f"{shape}/{density}: grain mismatch"
    return merged


def _confusion(merged_df, alert_col="alert", danger_col="danger"):
    alert = merged_df[alert_col]
    danger = merged_df[danger_col]
    TP = int((alert & danger).sum())
    FP = int((alert & ~danger).sum())
    TN = int((~alert & ~danger).sum())
    FN = int((~alert & danger).sum())
    return TP, FP, TN, FN


def _tpr_fpr_dist(TP, FP, TN, FN):
    tpr = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    fpr = FP / (FP + TN) if (FP + TN) > 0 else float("nan")
    dist = math.sqrt((1 - tpr) ** 2 + fpr ** 2) if not math.isnan(tpr) else float("nan")
    return tpr, fpr, dist


def compute_roc_table(slug):
    rows = []
    for density in DRONE_DENSITY_LEVELS:
        for shape in SHAPES:
            merged = _load_merged(slug, density, shape)
            TP, FP, TN, FN = _confusion(merged)
            tpr, fpr, dist = _tpr_fpr_dist(TP, FP, TN, FN)
            rows.append(dict(
                shape=shape, density=density, TP=TP, FP=FP, TN=TN, FN=FN,
                TPR=tpr, FPR=fpr, dist_to_corner=dist,
            ))
    return pd.DataFrame(rows)


def mcnemar_pairwise(slug):
    rows = []
    for density in DRONE_DENSITY_LEVELS:
        correctness = {}
        for shape in SHAPES:
            merged = _load_merged(slug, density, shape)
            merged["correct"] = merged["alert"] == merged["danger"]
            correctness[shape] = merged.set_index(["drone_id", "timestamp_utc"])["correct"]

        for shape_a, shape_b in itertools.combinations(SHAPES, 2):
            both = pd.concat([correctness[shape_a].rename("a"), correctness[shape_b].rename("b")], axis=1)
            table = (
                pd.crosstab(both["a"], both["b"])
                .reindex(index=[False, True], columns=[False, True], fill_value=0)
                .to_numpy()
            )
            result = mcnemar_test(table, exact=True)
            rows.append(dict(
                density=density, shape_a=shape_a, shape_b=shape_b,
                both_correct=int(table[1, 1]), both_wrong=int(table[0, 0]),
                a_only_wrong=int(table[0, 1]), b_only_wrong=int(table[1, 0]),
                statistic=float(result.statistic), pvalue=float(result.pvalue),
            ))
    return pd.DataFrame(rows)


def _per_drone_confusion(merged_df):
    m = merged_df
    tp = (m["alert"] & m["danger"]).astype(np.int64)
    fp = (m["alert"] & ~m["danger"]).astype(np.int64)
    tn = (~m["alert"] & ~m["danger"]).astype(np.int64)
    fn = (~m["alert"] & m["danger"]).astype(np.int64)
    per_drone = pd.DataFrame({"drone_id": m["drone_id"], "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return per_drone.groupby("drone_id", sort=False)[["TP", "FP", "TN", "FN"]].sum()


def bootstrap_ci(slug, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for density in DRONE_DENSITY_LEVELS:
        for shape in SHAPES:
            merged = _load_merged(slug, density, shape)
            per_drone = _per_drone_confusion(merged)
            counts = per_drone.to_numpy()
            n_drones = len(counts)

            idx = rng.integers(0, n_drones, size=(n_boot, n_drones))
            resampled = counts[idx].sum(axis=1)  # (n_boot, 4): TP,FP,TN,FN
            TP, FP, TN, FN = resampled[:, 0], resampled[:, 1], resampled[:, 2], resampled[:, 3]
            with np.errstate(invalid="ignore", divide="ignore"):
                tpr = np.where(TP + FN > 0, TP / (TP + FN), np.nan)
                fpr = np.where(FP + TN > 0, FP / (FP + TN), np.nan)

            rows.append(dict(
                shape=shape, density=density, n_drones=n_drones,
                tpr_low=np.nanpercentile(tpr, 2.5) if np.any(~np.isnan(tpr)) else float("nan"),
                tpr_high=np.nanpercentile(tpr, 97.5) if np.any(~np.isnan(tpr)) else float("nan"),
                fpr_low=np.nanpercentile(fpr, 2.5), fpr_high=np.nanpercentile(fpr, 97.5),
            ))
    return pd.DataFrame(rows)


def sensitivity_analysis(slug, shift_fracs=(-0.2, 0.0, 0.2)):
    rows = []
    for density in DRONE_DENSITY_LEVELS:
        candidates_df = pd.read_parquet(ground_truth_candidates_path(slug, density), columns=["drone_id", "timestamp_utc", "horizontal_m", "vertical_m"])
        drone_keys = pd.read_parquet(simulated_partition_path(slug, density), columns=["drone_id", "timestamp_utc"])

        for shift in shift_fracs:
            h_thresh = DANGER_HORIZONTAL_M * (1 + shift)
            v_thresh = DANGER_VERTICAL_M * (1 + shift)
            danger_keys = (
                candidates_df.loc[(candidates_df["horizontal_m"] <= h_thresh) & (candidates_df["vertical_m"] <= v_thresh), ["drone_id", "timestamp_utc"]]
                .drop_duplicates()
                .assign(danger_shift=True)
            )
            shifted_labels = drone_keys.merge(danger_keys, on=["drone_id", "timestamp_utc"], how="left")
            shifted_labels["danger_shift"] = shifted_labels["danger_shift"].fillna(False)

            for shape in SHAPES:
                alerts_df = pd.read_parquet(buffer_alerts_path(slug, density, shape), columns=["drone_id", "timestamp_utc", "alert"])
                merged = alerts_df.merge(shifted_labels, on=["drone_id", "timestamp_utc"], how="inner")
                TP, FP, TN, FN = _confusion(merged, alert_col="alert", danger_col="danger_shift")
                tpr, fpr, dist = _tpr_fpr_dist(TP, FP, TN, FN)
                rows.append(dict(
                    shape=shape, density=density, shift_pct=int(round(shift * 100)),
                    horizontal_threshold_m=h_thresh, vertical_threshold_m=v_thresh,
                    TP=TP, FP=FP, TN=TN, FN=FN, TPR=tpr, FPR=fpr, dist_to_corner=dist,
                ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Boston", help="Region name from pipeline.config.REGIONS")
    args = parser.parse_args()

    if args.region not in REGIONS or REGIONS[args.region] is None:
        raise ValueError(f"Unknown region: {args.region!r}. Choices: {list(REGIONS)}")

    slug = region_slug(args.region)
    out_dir = results_dir(slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Step 7 statistical validation for '{args.region}'\n")

    roc_df = compute_roc_table(slug)
    roc_df.to_parquet(os.path.join(out_dir, "roc_summary.parquet"), index=False)
    print("ROC summary (one point per shape per density):")
    print(roc_df.to_string(index=False))
    print()

    mcnemar_df = mcnemar_pairwise(slug)
    mcnemar_df.to_parquet(os.path.join(out_dir, "mcnemar.parquet"), index=False)
    print("McNemar's exact test (pairwise, per density):")
    print(mcnemar_df.to_string(index=False))
    print()

    ci_df = bootstrap_ci(slug)
    ci_df.to_parquet(os.path.join(out_dir, "bootstrap_ci.parquet"), index=False)
    print("Bootstrap 95% CIs (drone-level cluster bootstrap, 1000 resamples):")
    print(ci_df.to_string(index=False))
    print()

    sens_df = sensitivity_analysis(slug)
    sens_df.to_parquet(os.path.join(out_dir, "sensitivity.parquet"), index=False)
    print("Sensitivity analysis (ground-truth threshold shifted +/-20%):")
    print(sens_df.to_string(index=False))
    print()

    print(f"All results written to {out_dir}/")


if __name__ == "__main__":
    main()
