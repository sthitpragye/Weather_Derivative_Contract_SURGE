"""
clustering.py
=============
Phase 1 — District Clustering for UP Weather Derivative Project

Implements Clustering_Plan.txt end-to-end:

  Phase 1  — Merge three Phase 0 feature files
  Phase 2  — Missing value handling (column drop → median imputation)
  Phase 3  — Feature selection (low-variance + high-correlation removal)
  Phase 4  — StandardScaler
  Phase 5  — PCA (90% cumulative variance)
  Phase 6  — Ward hierarchical clustering, K selection
  Phase 7  — Cluster profiling and narrative
  Phase 8  — Save all outputs, plots, models, metadata

Inputs
------
  phase0_output/district_weather_features.csv
  phase0_output/district_price_features.csv
  phase0_output/district_apy_features.csv

Outputs (all under output/)
-----------------------------------
  data/
    final_clustering_dataset.csv
    clustering_features_selected.csv
    district_cluster_assignments.csv
    cluster_profiles.csv
    cluster_feature_zscores.csv
  plots/
    dendrogram.png
    scree_plot.png
    pca_biplot.png
    silhouette_plot.png
    cluster_heatmap.png
  models/
    scaler.pkl
    pca.pkl
    ward_linkage.npy
  clustering_metadata.json

Usage
-----
  python clustering.py
  python clustering.py --base-dir /path/to/SURGE
  python clustering.py --base-dir /path/to/SURGE --k 5
"""

import argparse
import datetime
import json
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  — all tuneable parameters in one place
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR               = "/Users/sthitpragye/Desktop/Finance/SURGE"

VARIANCE_THRESHOLD     = 0.01   # CV below this → drop (near-zero variance)
CORRELATION_THRESHOLD  = 0.95   # |r| above this → drop one of the pair
PCA_VARIANCE_TARGET    = 0.90   # cumulative explained variance to retain
MAX_K                  = 10     # maximum clusters to evaluate
K_OVERRIDE             = None   # set to an integer to force K; None = auto
MISSING_COL_THRESHOLD  = 0.20   # drop column if fraction missing > this
MISSING_ROW_THRESHOLD  = 0.30   # flag district if fraction missing > this
RANDOM_STATE           = 42

# Distinguishing feature threshold for cluster profiling
Z_THRESHOLD            = 1.0


# ═══════════════════════════════════════════════════════════════════════
# PATHS  (derived from BASE_DIR, overrideable via CLI)
# ═══════════════════════════════════════════════════════════════════════

def build_paths(base_dir: str) -> dict:
    p0 = os.path.join(base_dir, "phase0", "output")
    if K_OVERRIDE is None:
        run_name = "auto"
    else:
        run_name = f"k{K_OVERRIDE}"

    p1 = os.path.join(
        base_dir,
        "phase1",
        "output",
        run_name,
    )   

    return {
        # inputs
        "weather_features": os.path.join(p0, "district_weather_features.csv"),
        "price_features":   os.path.join(p0, "district_price_features.csv"),
        "apy_features":     os.path.join(p0, "district_apy_features.csv"),
        # output directories
        "out_data":         os.path.join(p1, "data"),
        "out_plots":        os.path.join(p1, "plots"),
        "out_models":       os.path.join(p1, "models"),
        # output files
        "final_dataset":    os.path.join(p1, "data", "final_clustering_dataset.csv"),
        "selected_features":os.path.join(p1, "data", "clustering_features_selected.csv"),
        "assignments":      os.path.join(p1, "data", "district_cluster_assignments.csv"),
        "profiles":         os.path.join(p1, "data", "cluster_profiles.csv"),
        "zscores":          os.path.join(p1, "data", "cluster_feature_zscores.csv"),
        "dendrogram":       os.path.join(p1, "plots", "dendrogram.png"),
        "scree":            os.path.join(p1, "plots", "scree_plot.png"),
        "biplot":           os.path.join(p1, "plots", "pca_biplot.png"),
        "silhouette_plot":  os.path.join(p1, "plots", "silhouette_plot.png"),
        "heatmap":          os.path.join(p1, "plots", "cluster_heatmap.png"),
        "scaler_pkl":       os.path.join(p1, "models", "scaler.pkl"),
        "pca_pkl":          os.path.join(p1, "models", "pca.pkl"),
        "linkage_npy":      os.path.join(p1, "models", "ward_linkage.npy"),
        "metadata":         os.path.join(p1, "clustering_metadata.json"),
    }


def make_output_dirs(paths: dict) -> None:
    for key in ("out_data", "out_plots", "out_models"):
        os.makedirs(paths[key], exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def subsection(title: str) -> None:
    print(f"\n  ── {title}")


def check_input(path: str, label: str) -> None:
    if not os.path.isfile(path):
        print(f"\nERROR: {label} not found at:\n  {path}")
        print("Run the Phase 0 preprocessing and feature engineering scripts first.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — MERGE & INITIAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def phase1_merge(paths: dict) -> pd.DataFrame:
    section("PHASE 1 — MERGE & INITIAL VALIDATION")

    check_input(paths["weather_features"], "district_weather_features.csv")
    check_input(paths["price_features"],   "district_price_features.csv")
    check_input(paths["apy_features"],     "district_apy_features.csv")

    weather = pd.read_csv(paths["weather_features"])
    price   = pd.read_csv(paths["price_features"])
    apy     = pd.read_csv(paths["apy_features"])

    print(f"  Weather features : {weather.shape[0]} districts × {weather.shape[1]-1} features")
    print(f"  Price features   : {price.shape[0]} districts × {price.shape[1]-1} features")
    print(f"  APY features     : {apy.shape[0]} districts × {apy.shape[1]-1} features")

    # Normalise district names to consistent Title Case and strip whitespace
    for df in (weather, price, apy):
        df["district"] = df["district"].astype(str).str.strip()

    weather_districts = set(weather["district"])
    price_districts   = set(price["district"])
    apy_districts     = set(apy["district"])

    # District overlap report
    in_all_three  = weather_districts & price_districts & apy_districts
    missing_price = weather_districts - price_districts
    missing_apy   = weather_districts - apy_districts

    print(f"\n  District overlap:")
    print(f"    Present in all three  : {len(in_all_three)}")
    if missing_price:
        print(f"    Missing from price    : {sorted(missing_price)}")
    else:
        print(f"    Missing from price    : none")
    if missing_apy:
        print(f"    Missing from APY      : {sorted(missing_apy)}")
    else:
        print(f"    Missing from APY      : none")

    # Merge — weather is the left anchor
    merged = weather.merge(price, on="district", how="left", suffixes=("", "_price"))
    merged = merged.merge(apy,   on="district", how="left", suffixes=("", "_apy"))

    # Verify row count is preserved
    assert len(merged) == len(weather), (
        f"Row count changed after merge: {len(weather)} → {len(merged)}"
    )

    print(f"\n  Merged dataset   : {merged.shape[0]} districts × {merged.shape[1]-1} features")

    merged.to_csv(paths["final_dataset"], index=False)
    print(f"  Saved → {paths['final_dataset']}")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — MISSING VALUE HANDLING
# ═══════════════════════════════════════════════════════════════════════

def phase2_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    section("PHASE 2 — MISSING VALUE HANDLING")

    districts = df["district"].copy()
    feature_cols = [c for c in df.columns if c != "district"]
    X = df[feature_cols].copy()

    n_districts, n_features = X.shape
    subsection("Pass 1: Drop columns with >20% missing")

    col_missing_frac = X.isna().mean()
    cols_to_drop = col_missing_frac[col_missing_frac > MISSING_COL_THRESHOLD].index.tolist()
    if cols_to_drop:
        print(f"    Dropping {len(cols_to_drop)} column(s):")
        for c in cols_to_drop:
            print(f"      {c}  ({col_missing_frac[c]*100:.1f}% missing)")
        X = X.drop(columns=cols_to_drop)
    else:
        print(f"    No columns exceed the {MISSING_COL_THRESHOLD*100:.0f}% threshold.")

    subsection("Pass 2: Median imputation + row-level flag")

    # Flag districts with high residual missingness before imputing
    row_missing_frac = X.isna().mean(axis=1)
    flagged = districts[row_missing_frac > MISSING_ROW_THRESHOLD].tolist()
    if flagged:
        print(f"    Districts with >{MISSING_ROW_THRESHOLD*100:.0f}% features missing "
              f"(flagged, not dropped):")
        for d in flagged:
            frac = row_missing_frac[districts == d].values[0]
            print(f"      {d}  ({frac*100:.1f}% missing)")
    else:
        print(f"    No districts exceed the {MISSING_ROW_THRESHOLD*100:.0f}% row threshold.")

    # Median imputation column-wise
    n_imputed_cells = int(X.isna().sum().sum())
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    print(f"\n    Imputed {n_imputed_cells} cells via column median.")
    print(f"    Features remaining: {X.shape[1]} (from {n_features} pre-drop)")

    retained_features = list(X.columns)
    result = pd.concat([districts.reset_index(drop=True),
                        X.reset_index(drop=True)], axis=1)
    return result, retained_features


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════

def phase3_feature_selection(df: pd.DataFrame,
                              retained: list) -> tuple[pd.DataFrame, list]:
    section("PHASE 3 — FEATURE SELECTION")

    X = df[retained].copy()
    n_before = X.shape[1]

    # ── Step 1: Near-zero variance (CV < threshold) ───────────────────
    subsection(f"Step 1: Remove near-zero variance  (CV < {VARIANCE_THRESHOLD})")

    means = X.mean().abs()
    stds  = X.std(ddof=1)
    # Avoid division by zero: features with mean ≈ 0 get CV = inf if std > 0,
    # meaning they ARE variable; features with both mean=0 and std=0 are
    # truly constant and should be dropped.
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(means == 0,
                      np.where(stds == 0, 0.0, np.inf),
                      stds / means)

    cv_series = pd.Series(cv, index=X.columns)
    low_var_cols = cv_series[cv_series < VARIANCE_THRESHOLD].index.tolist()

    if low_var_cols:
        print(f"    Dropping {len(low_var_cols)} near-zero-variance feature(s):")
        for c in low_var_cols:
            print(f"      {c}  (CV={cv_series[c]:.5f})")
        X = X.drop(columns=low_var_cols)
    else:
        print(f"    No features dropped (all CV ≥ {VARIANCE_THRESHOLD}).")

    n_after_var = X.shape[1]

    # ── Step 2: High correlation (|r| > threshold, greedy) ───────────
    subsection(f"Step 2: Remove highly correlated features  (|r| > {CORRELATION_THRESHOLD})")

    corr  = X.corr().abs()
    # Process columns in a deterministic sorted order so results are
    # reproducible regardless of the order features appear in the input.
    cols_sorted  = sorted(X.columns.tolist())
    dropped_corr = []
    retained_set = set(cols_sorted)

    for i, col_a in enumerate(cols_sorted):
        if col_a not in retained_set:
            continue
        for col_b in cols_sorted[i + 1:]:
            if col_b not in retained_set:
                continue
            if corr.loc[col_a, col_b] > CORRELATION_THRESHOLD:
                # Keep col_a (appears first alphabetically within its group);
                # drop col_b.
                retained_set.discard(col_b)
                dropped_corr.append((col_a, col_b, float(corr.loc[col_a, col_b])))

    if dropped_corr:
        print(f"    Dropping {len(dropped_corr)} redundant feature(s):")
        for kept, dropped, r in dropped_corr:
            print(f"      DROP {dropped}  (|r|={r:.3f} with {kept})")
    else:
        print(f"    No features dropped (no pair exceeds |r|={CORRELATION_THRESHOLD}).")

    # Apply the drop
    final_cols = sorted(retained_set)
    X = X[final_cols]
    n_after_corr = X.shape[1]

    # ── Step 3: Report ────────────────────────────────────────────────
    subsection("Feature selection summary")
    print(f"    Features before selection  : {n_before}")
    print(f"    Dropped (low variance)     : {n_before - n_after_var}")
    print(f"    Dropped (high correlation) : {n_after_var - n_after_corr}")
    print(f"    Features retained          : {n_after_corr}")

    # Breakdown by thematic group
    groups = {}
    for col in final_cols:
        prefix = col.split("_")[0]
        groups.setdefault(prefix, []).append(col)
    print(f"\n    Retained features by group:")
    for grp, cols in sorted(groups.items()):
        print(f"      {grp:20s}: {len(cols)}")

    result = pd.concat([df[["district"]].reset_index(drop=True),
                        X.reset_index(drop=True)], axis=1)
    return result, final_cols


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — SCALING
# ═══════════════════════════════════════════════════════════════════════

def phase4_scale(df: pd.DataFrame,
                 feature_cols: list,
                 paths: dict) -> tuple[np.ndarray, StandardScaler]:
    section("PHASE 4 — SCALING (StandardScaler)")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(paths["scaler_pkl"], "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaled {X_scaled.shape[0]} districts × {X_scaled.shape[1]} features.")
    print(f"  Saved → {paths['scaler_pkl']}")

    return X_scaled, scaler


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — PCA
# ═══════════════════════════════════════════════════════════════════════

def phase5_pca(X_scaled: np.ndarray,
               districts: pd.Series,
               feature_cols: list,
               paths: dict) -> tuple[np.ndarray, PCA]:
    section(f"PHASE 5 — PCA  (target: {PCA_VARIANCE_TARGET*100:.0f}% cumulative variance)")

    # Fit full PCA first to inspect the variance curve
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
    n_components = min(n_components, X_scaled.shape[1])

    print(f"  Components to reach {PCA_VARIANCE_TARGET*100:.0f}% variance: {n_components}")
    print(f"  Variance explained by retained components: {cumvar[n_components-1]*100:.2f}%")

    # Refit with chosen component count
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    with open(paths["pca_pkl"], "wb") as f:
        pickle.dump(pca, f)
    print(f"  Saved → {paths['pca_pkl']}")

    # ── Scree plot ────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    evr = pca_full.explained_variance_ratio_
    n_show = min(30, len(evr))
    x_vals = np.arange(1, n_show + 1)

    ax1.bar(x_vals, evr[:n_show] * 100, color="steelblue", alpha=0.7,
            label="Individual")
    ax2.plot(x_vals, cumvar[:n_show] * 100, "o-", color="darkorange",
             linewidth=2, label="Cumulative")
    ax2.axhline(PCA_VARIANCE_TARGET * 100, ls="--", color="red", alpha=0.6,
                label=f"{PCA_VARIANCE_TARGET*100:.0f}% target")
    ax2.axvline(n_components, ls=":", color="green", alpha=0.8,
                label=f"Cut at PC{n_components}")

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Individual Explained Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_ylim(0, 105)
    ax1.set_title("PCA Scree Plot — UP District Features")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")

    plt.tight_layout()
    plt.savefig(paths["scree"], dpi=150)
    plt.close()
    print(f"  Saved → {paths['scree']}")

    # ── PC1 vs PC2 biplot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=60, color="steelblue",
               alpha=0.8, zorder=3)

    for i, name in enumerate(districts):
        ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Biplot — UP Districts (PC1 vs PC2)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(paths["biplot"], dpi=150)
    plt.close()
    print(f"  Saved → {paths['biplot']}")

    # ── Component loadings summary ────────────────────────────────────
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    # Print top 5 loading features for PC1 and PC2 as a quick sanity check
    for pc in ["PC1", "PC2"]:
        top5 = loadings[pc].abs().nlargest(5)
        print(f"\n  Top 5 loadings on {pc}:")
        for feat, val in top5.items():
            direction = "+" if loadings.loc[feat, pc] > 0 else "−"
            print(f"    {direction}{val:.3f}  {feat}")

    return X_pca, pca


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — HIERARCHICAL CLUSTERING (WARD)
# ═══════════════════════════════════════════════════════════════════════

def _dendrogram_k(Z: np.ndarray) -> int:
    """
    Estimate optimal K from the Ward linkage matrix by finding the
    largest gap between consecutive merge distances (the 'acceleration'
    method).  Returns the K whose last merge had the largest jump.
    """
    last   = Z[-MAX_K:, 2]           # distances of the last MAX_K merges
    accel  = np.diff(last, 2)        # second derivative of distances
    k_dend = int(accel[::-1].argmax()) + 2   # +2 because of double diff
    return max(2, min(k_dend, MAX_K))


def phase6_cluster(X_pca: np.ndarray,
                   districts: pd.Series,
                   paths: dict) -> tuple[np.ndarray, np.ndarray, int, dict]:
    section("PHASE 6 — WARD HIERARCHICAL CLUSTERING")

    # Ward linkage
    Z = linkage(X_pca, method="ward", metric="euclidean")
    np.save(paths["linkage_npy"], Z)
    print(f"  Ward linkage computed.  Saved → {paths['linkage_npy']}")

    # ── Dendrogram ────────────────────────────────────────────────────
    subsection("Dendrogram")
    fig, ax = plt.subplots(figsize=(16, 7))

    # optimal_leaf_ordering reorders leaves for visual clarity (minimises
    # the sum of adjacent distances in the dendrogram)
    try:
        Z_ordered = optimal_leaf_ordering(Z, X_pca)
    except Exception:
        Z_ordered = Z    # fallback if scipy version is old

    dend = dendrogram(
        Z_ordered,
        labels=districts.values,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=0,
        ax=ax,
    )
    ax.set_title("Ward Hierarchical Clustering — UP Districts", fontsize=13)
    ax.set_ylabel("Distance (Ward)")
    ax.set_xlabel("District")
    plt.tight_layout()
    plt.savefig(paths["dendrogram"], dpi=150)
    plt.close()
    print(f"  Saved → {paths['dendrogram']}")

    # ── Evaluate K = 2…MAX_K ─────────────────────────────────────────
    subsection(f"Evaluating K = 2 to {MAX_K}")

    silhouette_scores = {}
    for k in range(2, MAX_K + 1):
        labels_k = fcluster(Z, k, criterion="maxclust")
        sil = silhouette_score(X_pca, labels_k)
        silhouette_scores[k] = sil
        print(f"    K={k:2d}  silhouette={sil:.4f}")

    k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
    k_dendrogram = _dendrogram_k(Z)

    print(f"\n  K selected by silhouette : {k_silhouette} "
          f"(score={silhouette_scores[k_silhouette]:.4f})")
    print(f"  K suggested by dendrogram: {k_dendrogram}")

    # ── K decision rule ───────────────────────────────────────────────
    if K_OVERRIDE is not None:
        k_final = int(K_OVERRIDE)
        print(f"\n  K_OVERRIDE active → using K={k_final}")
    elif k_silhouette == k_dendrogram:
        k_final = k_silhouette
        print(f"\n  Both criteria agree → K={k_final}")
    else:
        k_final = k_silhouette
        print(f"\n  Criteria disagree — defaulting to silhouette K={k_final}.")
        print(f"  (Dendrogram suggested K={k_dendrogram}; override with --k or K_OVERRIDE)")

    # ── Silhouette plot ───────────────────────────────────────────────
    k_vals  = list(silhouette_scores.keys())
    sil_vals = list(silhouette_scores.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_vals, sil_vals, "o-", color="steelblue", linewidth=2)
    ax.axvline(k_final, ls="--", color="red", alpha=0.7,
               label=f"Chosen K={k_final}")
    if k_dendrogram != k_final:
        ax.axvline(k_dendrogram, ls=":", color="orange", alpha=0.7,
                   label=f"Dendrogram K={k_dendrogram}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Score vs K — Ward Clustering")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths["silhouette_plot"], dpi=150)
    plt.close()
    print(f"  Saved → {paths['silhouette_plot']}")

    # ── Final cluster assignment ───────────────────────────────────────
    final_labels = fcluster(Z, k_final, criterion="maxclust")

    k_diagnostics = {
        "k_final":          k_final,
        "k_silhouette":     k_silhouette,
        "k_dendrogram":     k_dendrogram,
        "silhouette_scores": {int(k): round(v, 6)
                               for k, v in silhouette_scores.items()},
        "silhouette_final": round(silhouette_scores[k_final], 6),
    }

    return Z, final_labels, k_final, k_diagnostics


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — CLUSTER PROFILING
# ═══════════════════════════════════════════════════════════════════════

def phase7_profile(df_selected: pd.DataFrame,
                   feature_cols: list,
                   final_labels: np.ndarray,
                   X_pca: np.ndarray,
                   k_final: int,
                   paths: dict) -> pd.DataFrame:
    section("PHASE 7 — CLUSTER PROFILING")

    districts = df_selected["district"].reset_index(drop=True)
    X_feat    = df_selected[feature_cols].reset_index(drop=True)

    # Individual silhouette scores
    sil_samples = silhouette_samples(X_pca, final_labels)

    # ── District-cluster assignments ──────────────────────────────────
    assignments = pd.DataFrame({
        "district":                   districts,
        "cluster_id":                 final_labels,
        "silhouette_score_individual": sil_samples.round(6),
    }).sort_values(["cluster_id", "district"]).reset_index(drop=True)

    assignments.to_csv(paths["assignments"], index=False)
    print(f"  Saved → {paths['assignments']}")
    cluster_counts = assignments["cluster_id"].value_counts().sort_index()

    small_clusters = cluster_counts[cluster_counts < 3]

    if not small_clusters.empty:
        print("\n  WARNING: Very small clusters detected.")
        print("  This may indicate over-clustering.")
        for cluster_id, size in small_clusters.items():
            print(f"    Cluster {cluster_id}: {size} district(s)")

    # ── Cluster size table ────────────────────────────────────────────
    subsection("Cluster sizes")
    for cid in sorted(assignments["cluster_id"].unique()):
        members = assignments.loc[assignments["cluster_id"] == cid, "district"].tolist()
        print(f"    Cluster {cid}  ({len(members)} districts):")
        for m in members:
            print(f"      {m}")

    # ── Cluster mean profiles ─────────────────────────────────────────
    X_feat_labelled = X_feat.copy()
    X_feat_labelled["cluster_id"] = final_labels

    cluster_means = X_feat_labelled.groupby("cluster_id")[feature_cols].mean()
    all_mean      = X_feat[feature_cols].mean().rename("All Districts")

    profiles = cluster_means.copy()
    profiles.index = [f"Cluster_{i}" for i in profiles.index]
    profiles.loc["All Districts"] = all_mean
    profiles.to_csv(paths["profiles"])
    print(f"  Saved → {paths['profiles']}")

    # ── Z-scores of cluster means ──────────────────────────────────────
    all_std = X_feat[feature_cols].std(ddof=1)
    z_scores = (cluster_means - all_mean) / all_std.replace(0, np.nan)
    z_scores.index = [f"Cluster_{i}" for i in z_scores.index]
    z_scores.to_csv(paths["zscores"])
    print(f"  Saved → {paths['zscores']}")

    # ── Cluster narrative ─────────────────────────────────────────────
    subsection("Cluster narrative (top 5 distinguishing features per cluster)")
    for cid in sorted(assignments["cluster_id"].unique()):
        label = f"Cluster_{cid}"
        z_row = z_scores.loc[label].dropna().abs().nlargest(5)
        print(f"\n  Cluster {cid}")
        for feat, z_abs in z_row.items():
            raw_z = z_scores.loc[label, feat]
            direction = "ABOVE" if raw_z > 0 else "BELOW"
            print(f"    {direction} mean  z={raw_z:+.2f}  {feat}")

    # ── Cluster heatmap ───────────────────────────────────────────────
    # Show only features that are distinguishing (|z| > Z_THRESHOLD)
    # for at least one cluster, to keep the heatmap readable.
    max_z_per_feature = z_scores.abs().max(axis=0)
    distinguishing    = max_z_per_feature[max_z_per_feature > Z_THRESHOLD].index.tolist()

    if len(distinguishing) > 0:
        heat_data = z_scores[distinguishing]
        n_feats   = min(len(distinguishing), 50)   # cap at 50 for legibility
        top_feats = max_z_per_feature[distinguishing].nlargest(n_feats).index.tolist()
        heat_data  = heat_data[top_feats]

        fig_h = max(8, n_feats * 0.28)
        fig, ax = plt.subplots(figsize=(max(8, k_final * 1.5), fig_h))

        vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()))
        im = ax.imshow(heat_data.T.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Z-score vs all-district mean")

        ax.set_xticks(range(k_final))
        ax.set_xticklabels([f"Cluster {i+1}" for i in range(k_final)])
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(top_feats, fontsize=7)
        ax.set_title(f"Cluster Feature Z-Scores  (|z|>{Z_THRESHOLD}, top {n_feats} features)")

        plt.tight_layout()
        plt.savefig(paths["heatmap"], dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {paths['heatmap']}")
    else:
        print(f"  NOTE: no features with |z|>{Z_THRESHOLD}; heatmap not generated.")

    return assignments


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 — SAVE METADATA & SELECTED FEATURE FILE
# ═══════════════════════════════════════════════════════════════════════

def phase8_metadata(df_selected: pd.DataFrame,
                    feature_cols: list,
                    pca: PCA,
                    k_diagnostics: dict,
                    paths: dict,
                    n_features_raw: int,
                    n_dropped_var: int,
                    n_dropped_corr: int) -> None:
    section("PHASE 8 — SAVING OUTPUTS & METADATA")

    # Save selected-feature CSV
    df_selected.to_csv(paths["selected_features"], index=False)
    print(f"  Saved → {paths['selected_features']}")

    # Metadata dict
    meta = {
        "generated_at":             datetime.datetime.now().isoformat(timespec="seconds"),
        "n_districts":              int(len(df_selected)),
        "n_features_before_selection": int(n_features_raw),
        "n_dropped_low_variance":   int(n_dropped_var),
        "n_dropped_high_correlation": int(n_dropped_corr),
        "n_features_after_selection": int(len(feature_cols)),
        "feature_names":            feature_cols,
        "n_pca_components":         int(pca.n_components_),
        "variance_explained_pct":   round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
        "pca_variance_target_pct":  PCA_VARIANCE_TARGET * 100,
        "linkage_method":           "ward",
        "distance_metric":          "euclidean",
        "max_k_evaluated":          MAX_K,
        "k_override":               K_OVERRIDE,
        **k_diagnostics,
        "config": {
            "VARIANCE_THRESHOLD":    VARIANCE_THRESHOLD,
            "CORRELATION_THRESHOLD": CORRELATION_THRESHOLD,
            "PCA_VARIANCE_TARGET":   PCA_VARIANCE_TARGET,
            "MISSING_COL_THRESHOLD": MISSING_COL_THRESHOLD,
            "MISSING_ROW_THRESHOLD": MISSING_ROW_THRESHOLD,
            "RANDOM_STATE":          RANDOM_STATE,
        },
    }

    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {paths['metadata']}")

    # ── Final summary ─────────────────────────────────────────────────
    section("CLUSTERING COMPLETE")
    print(f"""
  Districts clustered   : {meta['n_districts']}
  Features (raw)        : {meta['n_features_before_selection']}
  Features (selected)   : {meta['n_features_after_selection']}
  PCA components        : {meta['n_pca_components']}  ({meta['variance_explained_pct']:.1f}% variance)
  Final K               : {meta['k_final']}
  Silhouette score      : {meta['silhouette_final']:.4f}
  Linkage               : Ward / Euclidean

  Outputs → {os.path.dirname(paths['out_data'])}
  ├── data/
  │   ├── final_clustering_dataset.csv
  │   ├── clustering_features_selected.csv
  │   ├── district_cluster_assignments.csv
  │   ├── cluster_profiles.csv
  │   └── cluster_feature_zscores.csv
  ├── plots/
  │   ├── dendrogram.png
  │   ├── scree_plot.png
  │   ├── pca_biplot.png
  │   ├── silhouette_plot.png
  │   └── cluster_heatmap.png
  ├── models/
  │   ├── scaler.pkl
  │   ├── pca.pkl
  │   └── ward_linkage.npy
  └── clustering_metadata.json
""")


# ═══════════════════════════════════════════════════════════════════════
# CLI + MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="District clustering pipeline (Clustering_Plan.txt)"
    )
    parser.add_argument(
        "--base-dir", default=BASE_DIR,
        help="Path to SURGE project root (default: %(default)s)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override the automatically determined K (takes precedence over K_OVERRIDE)"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    base   = args.base_dir

    # CLI --k overrides the module-level constant
    global K_OVERRIDE
    if args.k is not None:
        K_OVERRIDE = args.k

    paths = build_paths(base)
    os.makedirs(paths["out_data"], exist_ok=True)
    os.makedirs(paths["out_plots"], exist_ok=True)
    os.makedirs(paths["out_models"], exist_ok=True)

    print("\n" + "=" * 70)
    print("DISTRICT CLUSTERING — UP Weather Derivative Project")
    print("=" * 70)
    if K_OVERRIDE is not None:
        print(f"  K_OVERRIDE = {K_OVERRIDE}  (will skip automatic K selection)")

    # ── Phase 1: Merge ────────────────────────────────────────────────
    merged = phase1_merge(paths)
    n_features_raw = merged.shape[1] - 1   # exclude 'district'

    # ── Phase 2: Missing value handling ───────────────────────────────
    df_clean, retained = phase2_missing(merged)

    # ── Phase 3: Feature selection ────────────────────────────────────
    df_selected, final_cols = phase3_feature_selection(df_clean, retained)

    n_dropped_var  = len(retained) - len(
        [c for c in retained
         if c in df_selected.columns or c in final_cols]
    )
    n_after_var  = len(retained) - (len(retained) - len(
        [c for c in retained
         if pd.Series(df_clean[retained][c].values).std(ddof=1) /
            max(abs(pd.Series(df_clean[retained][c].values).mean()), 1e-12)
            >= VARIANCE_THRESHOLD
        ]
    ))
    n_dropped_var  = len(retained) - n_after_var
    n_dropped_corr = n_after_var - len(final_cols)

    # ── Phase 4: Scale ────────────────────────────────────────────────
    X_scaled, scaler = phase4_scale(df_selected, final_cols, paths)

    # ── Phase 5: PCA ──────────────────────────────────────────────────
    districts = df_selected["district"].reset_index(drop=True)
    X_pca, pca = phase5_pca(X_scaled, districts, final_cols, paths)

    # ── Phase 6: Cluster ──────────────────────────────────────────────
    Z, final_labels, k_final, k_diagnostics = phase6_cluster(
        X_pca, districts, paths
    )

    # ── Phase 7: Profile ──────────────────────────────────────────────
    assignments = phase7_profile(
        df_selected, final_cols, final_labels, X_pca, k_final, paths
    )

    # ── Phase 8: Metadata ─────────────────────────────────────────────
    phase8_metadata(
        df_selected, final_cols, pca, k_diagnostics, paths,
        n_features_raw, n_dropped_var, n_dropped_corr
    )

    return assignments


if __name__ == "__main__":
    main()