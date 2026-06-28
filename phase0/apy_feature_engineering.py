"""
APY FEATURE ENGINEERING MODULE — Phase 0
=========================================
Reads district_apy_panel.csv (output of apy_preprocessing.py) and
produces district_apy_features.csv — one row per district, one column
per engineered feature.  This file is one of the three inputs to the
final merge that creates the clustering dataset.

FEATURE GROUPS:
  1. Yield Level Features       (mean, median, std, CV, max, min, range)
  2. Yield Trend Features       (linear slope, CAGR, Mann-Kendall tau)
  3. Area Features              (mean, CV, linear slope, CAGR)
  4. Production Features        (mean, CV, linear slope, CAGR)
  5. Data Quality Features      (years observed, fraction observed)

DESIGN NOTES:
  - No imputation.  If a district-year is absent from the panel, that
    year simply doesn't contribute to the district's statistics.
    Features that require a minimum number of data points (e.g. CAGR,
    Mann-Kendall) return NaN for data-sparse districts rather than
    unreliable estimates.
  - All features are numeric.  The 'district' column is the merge key.
  - Column names use the 'apy_' prefix throughout so there is zero
    ambiguity when merged with weather and price features.

INPUT:
  phase0_output/district_apy_panel.csv

OUTPUT:
  phase0_output/district_apy_features.csv
    One row per district.
    Approx. 20 engineered features.

Run:
  python apy_feature_engineering.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = "/Users/sthitpragye/Desktop/Finance/SURGE"
OUT_DIR  = os.path.join(BASE_DIR, "phase0", "output")

PANEL_FILE   = os.path.join(OUT_DIR, "district_apy_panel.csv")
OUTPUT_FILE  = os.path.join(OUT_DIR, "district_apy_features.csv")

YEAR_START = 2006
YEAR_END   = 2025

MIN_OBS_TREND   = 5   # minimum non-NaN observations to compute slope / CAGR / MK
MIN_OBS_CAGR    = 3   # minimum to attempt CAGR (needs valid first and last year)


# ═══════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def safe_cv(series: pd.Series) -> float:
    """CV = std / mean.  Returns NaN if mean is zero or series is empty."""
    s = series.dropna()
    if s.empty or s.mean() == 0:
        return np.nan
    return s.std() / s.mean()


def linear_slope(years: np.ndarray, values: np.ndarray) -> float:
    """
    Slope of OLS regression of values on years.
    Returns NaN if fewer than MIN_OBS_TREND valid pairs.
    """
    mask = ~np.isnan(values)
    if mask.sum() < MIN_OBS_TREND:
        return np.nan
    slope, _, _, _, _ = stats.linregress(years[mask], values[mask])
    return slope


def safe_cagr(series: pd.Series, years: pd.Series) -> float:
    """
    CAGR using the mean of the first two available years as 'first' and
    the mean of the last two available years as 'last', to reduce the
    influence of individual-year noise at the endpoints.

    Returns NaN if fewer than MIN_OBS_CAGR non-NaN values or if the
    effective first value is zero/negative.
    """
    valid = series.dropna()
    if len(valid) < MIN_OBS_CAGR:
        return np.nan

    # Sort by year
    sorted_idx = years[valid.index].sort_values().index
    valid_sorted = valid.loc[sorted_idx]

    n = len(valid_sorted)
    # Use first and last single observation when n < 4, else use pair averages
    if n >= 4:
        first_val = valid_sorted.iloc[:2].mean()
        last_val  = valid_sorted.iloc[-2:].mean()
        n_periods = (years.loc[sorted_idx].iloc[-2:].mean()
                     - years.loc[sorted_idx].iloc[:2].mean())
    else:
        first_val = valid_sorted.iloc[0]
        last_val  = valid_sorted.iloc[-1]
        n_periods = (years.loc[sorted_idx].iloc[-1]
                     - years.loc[sorted_idx].iloc[0])

    if first_val <= 0 or n_periods <= 0:
        return np.nan

    return (last_val / first_val) ** (1.0 / n_periods) - 1.0


def mann_kendall_tau(series: pd.Series) -> float:
    """
    Kendall's tau from the Mann-Kendall trend test.
    Returns NaN if fewer than MIN_OBS_TREND non-NaN values.
    """
    s = series.dropna()
    if len(s) < MIN_OBS_TREND:
        return np.nan
    tau, _ = stats.kendalltau(np.arange(len(s)), s.values)
    return tau


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & VALIDATE
# ═══════════════════════════════════════════════════════════════════════

def load_panel() -> pd.DataFrame:
    if not os.path.exists(PANEL_FILE):
        raise FileNotFoundError(
            f"Panel file not found: {PANEL_FILE}\n"
            f"Run apy_preprocessing.py first."
        )

    df = pd.read_csv(PANEL_FILE)
    print(f"  Loaded panel: {len(df):,} rows, columns: {list(df.columns)}")

    required = ["district", "year", "area", "production", "yield"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Panel is missing required columns: {missing_cols}")

    # Filter to analysis window
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)].copy()
    print(f"  After year filter ({YEAR_START}–{YEAR_END}): {len(df):,} rows, "
          f"{df['district'].nunique()} districts")

    for col in ["area", "production", "yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — YIELD LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_yield_level_features(grp: pd.DataFrame) -> dict:
    y = grp["yield"].dropna()
    if y.empty:
        return {
            "apy_mean_yield":   np.nan, "apy_median_yield": np.nan,
            "apy_std_yield":    np.nan, "apy_cv_yield":     np.nan,
            "apy_max_yield":    np.nan, "apy_min_yield":    np.nan,
            "apy_yield_range":  np.nan,
        }
    return {
        "apy_mean_yield":   y.mean(),
        "apy_median_yield": y.median(),
        "apy_std_yield":    y.std(),
        "apy_cv_yield":     safe_cv(y),
        "apy_max_yield":    y.max(),
        "apy_min_yield":    y.min(),
        "apy_yield_range":  y.max() - y.min(),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — YIELD TREND FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_yield_trend_features(grp: pd.DataFrame) -> dict:
    valid = grp[["year", "yield"]].dropna(subset=["yield"])
    y_arr = valid["yield"].values
    t_arr = valid["year"].values.astype(float)

    return {
        "apy_yield_trend_slope": linear_slope(t_arr, y_arr),
        "apy_yield_cagr":        safe_cagr(valid["yield"], valid["year"]),
        "apy_yield_mk_tau":      mann_kendall_tau(valid["yield"]),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — AREA FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_area_features(grp: pd.DataFrame) -> dict:
    valid = grp[["year", "area"]].dropna(subset=["area"])
    a = valid["area"]
    t = valid["year"].values.astype(float)
    a_arr = a.values

    return {
        "apy_mean_area":         a.mean()                    if not a.empty else np.nan,
        "apy_cv_area":           safe_cv(a),
        "apy_area_trend_slope":  linear_slope(t, a_arr),
        "apy_area_cagr":         safe_cagr(a, valid["year"]),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — PRODUCTION FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_production_features(grp: pd.DataFrame) -> dict:
    valid = grp[["year", "production"]].dropna(subset=["production"])
    p = valid["production"]
    t = valid["year"].values.astype(float)
    p_arr = p.values

    return {
        "apy_mean_production":        p.mean()               if not p.empty else np.nan,
        "apy_cv_production":          safe_cv(p),
        "apy_production_trend_slope": linear_slope(t, p_arr),
        "apy_production_cagr":        safe_cagr(p, valid["year"]),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — DATA QUALITY FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_data_quality_features(grp: pd.DataFrame) -> dict:
    total_possible = YEAR_END - YEAR_START + 1  # 20

    # A year counts as "observed" if at least yield is non-NaN
    years_observed = grp["yield"].notna().sum()
    # A year has all three metrics
    years_complete = grp[["area", "production", "yield"]].notna().all(axis=1).sum()

    return {
        "apy_years_yield_observed":     int(years_observed),
        "apy_fraction_yield_observed":  years_observed / total_possible,
        "apy_years_all_complete":       int(years_complete),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — ASSEMBLE FEATURES PER DISTRICT
# ═══════════════════════════════════════════════════════════════════════

def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    districts = sorted(panel["district"].unique())
    print(f"\n  Computing features for {len(districts)} districts...")

    for district in districts:
        grp = panel[panel["district"] == district].sort_values("year")

        feat = {"district": district}
        feat.update(compute_yield_level_features(grp))
        feat.update(compute_yield_trend_features(grp))
        feat.update(compute_area_features(grp))
        feat.update(compute_production_features(grp))
        feat.update(compute_data_quality_features(grp))

        rows.append(feat)

    features_df = pd.DataFrame(rows)
    return features_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 — VALIDATION & OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def validate_and_save(df: pd.DataFrame) -> pd.DataFrame:
    sep = "─" * 70
    print(f"\n{sep}")
    print("PHASE 8: Validation")
    print(sep)

    # Shape
    n_districts, n_cols = df.shape
    n_features = n_cols - 1  # exclude 'district'
    print(f"  Output shape: {n_districts} districts × {n_features} features")

    # Duplicate districts
    dup_districts = df[df.duplicated("district", keep=False)]["district"].unique()
    if len(dup_districts) > 0:
        print(f"  ERROR: duplicate district rows found: {dup_districts}")
    else:
        print(f"  ✓ No duplicate districts")

    # Non-numeric columns (other than 'district')
    non_numeric = [c for c in df.columns
                   if c != "district" and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"  WARNING: non-numeric feature columns: {non_numeric}")
    else:
        print(f"  ✓ All feature columns are numeric")

    # Missing value summary
    feature_cols = [c for c in df.columns if c != "district"]
    miss = df[feature_cols].isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print(f"  ✓ No missing values in feature matrix")
    else:
        print(f"  Missing values by feature:")
        for col, cnt in miss.items():
            print(f"    {col}: {cnt} ({cnt / n_districts * 100:.1f}%)")

    # Data quality summary
    if "apy_fraction_yield_observed" in df.columns:
        low_cov = df[df["apy_fraction_yield_observed"] < 0.5]
        if not low_cov.empty:
            print(f"  NOTE: {len(low_cov)} districts have <50% yield coverage:")
            print(f"    {sorted(low_cov['district'].tolist())}")

    # Column order: district first, then feature groups in order
    col_order = (
        ["district"]
        + [c for c in df.columns if c.startswith("apy_mean_yield")
           or c.startswith("apy_median_yield")
           or c.startswith("apy_std_yield")
           or c.startswith("apy_cv_yield")
           or c.startswith("apy_max_yield")
           or c.startswith("apy_min_yield")
           or c.startswith("apy_yield_range")]
        + [c for c in df.columns if "trend_slope" in c and "yield" in c]
        + [c for c in df.columns if c == "apy_yield_cagr"]
        + [c for c in df.columns if c == "apy_yield_mk_tau"]
        + [c for c in df.columns if "area" in c]
        + [c for c in df.columns if "production" in c]
        + [c for c in df.columns if "years_" in c or "fraction_" in c]
    )
    # Fill in any stragglers not caught by the above
    col_order += [c for c in df.columns if c not in col_order]
    df = df[col_order].sort_values("district").reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved → {OUTPUT_FILE}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(sep)
    print(f"APY FEATURE ENGINEERING — Phase 0  |  Years: {YEAR_START}–{YEAR_END}")
    print(sep)

    # PHASE 1: Load
    print(f"\n{'─' * 70}")
    print("PHASE 1: Loading panel")
    print(f"{'─' * 70}")
    panel = load_panel()

    # PHASES 2–7: Compute features
    print(f"\n{'─' * 70}")
    print("PHASES 2–7: Computing feature groups per district")
    print(f"{'─' * 70}")
    features = compute_features(panel)

    # PHASE 8: Validate and save
    features = validate_and_save(features)

    feature_cols = [c for c in features.columns if c != "district"]
    print(f"""
{sep}
APY FEATURE ENGINEERING COMPLETE.

Output : {OUTPUT_FILE}
Shape  : {features.shape[0]} districts × {len(feature_cols)} features

Feature groups:
  Yield level   : apy_mean/median/std/cv/max/min_yield, apy_yield_range
  Yield trend   : apy_yield_trend_slope, apy_yield_cagr, apy_yield_mk_tau
  Area          : apy_mean_area, apy_cv_area, apy_area_trend_slope, apy_area_cagr
  Production    : apy_mean_production, apy_cv_production,
                  apy_production_trend_slope, apy_production_cagr
  Data quality  : apy_years_yield_observed, apy_fraction_yield_observed,
                  apy_years_all_complete

NEXT STEP:
  Merge with phase0_output/district_weather_features.csv
  and     phase0_output/district_price_features.csv
  on the 'district' column to produce final_clustering_dataset.csv.
{sep}
""")
    return features


if __name__ == "__main__":
    main()