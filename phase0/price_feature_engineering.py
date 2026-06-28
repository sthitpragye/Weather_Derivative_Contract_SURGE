"""
price_feature_engineering.py

Implements price_feature_engineering_plan.txt, producing one row of
engineered features per district.

  Phase 1 - Load & validate district_monthly_prices.csv
  Phase 2 - Overall price features
  Phase 3 - Price variability features
  Phase 4 - Seasonal features
  Phase 5 - Trend features
  Phase 6 - SKIPPED ENTIRELY: market-activity / arrivals features. This
            project's price data has no arrivals column (confirmed - see
            price_preprocessing.py), so all 6 Phase 6 features are omitted.
  Phase 7 - Price dynamics features
  Phase 8 - Data quality features (the one arrivals-related item,
            "Fraction of Zero-Arrival Months", is omitted for the same
            reason as Phase 6)
  Phase 9 - Final output -> phase0_output/district_price_features.csv

Input:
  phase0_output/district_monthly_prices.csv
    columns: district, year, month, mean_modal_price, mean_min_price,
             mean_max_price, mean_price_spread, reporting_days, is_imputed

Optional dependency:
  ruptures, for Phase 5's structural-break count (the plan names it
  explicitly: "e.g. ruptures"). If it isn't installed, a simple variance-
  shift fallback is used instead and a warning is printed - install the
  real thing with:  pip install ruptures

Usage:
    python price_feature_engineering.py
    python price_feature_engineering.py --base-dir /path/to/SURGE --start-year 2006 --end-year 2025
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ruptures as rpt
    HAVE_RUPTURES = True
except ImportError:
    HAVE_RUPTURES = False

REQUIRED_COLUMNS = ["district", "year", "month", "mean_modal_price", "mean_min_price",
                    "mean_max_price", "mean_price_spread", "reporting_days", "is_imputed"]

MONTH_NAMES = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
               7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}


# ----------------------------------------------------------------------
# Phase 1 - Load & validate
# ----------------------------------------------------------------------
def load_and_validate(path, start_year, end_year):
    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input file missing required columns: {missing_cols}")

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    df = df.sort_values(["district", "year", "month"]).reset_index(drop=True)

    counts = df.groupby(["district", "year"])["month"].nunique()
    bad = counts[counts != 12]
    if len(bad) > 0:
        preview = bad.index.tolist()[:5]
        print(f"  \u26a0 {len(bad)} (district, year) groups do not have exactly 12 months "
              f"- continuity assumption may be violated, e.g. {preview}"
              f"{' ...' if len(bad) > 5 else ''}")
    else:
        print("  \u2713 Every (district, year) has exactly 12 months.")

    return df


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def longest_run(mask):
    """Longest run of consecutive True values in a boolean array."""
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def count_turning_points(diffs):
    """Count sign changes in a sequence of differences."""
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int((np.diff(signs) != 0).sum())


def max_drawdown_pct(series):
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(-drawdown.min() * 100)  # positive magnitude, in %


def structural_break_count(series):
    """Count detected structural breaks in a price series.

    Uses ruptures (PELT, rbf model) on a z-scored series when available;
    falls back to a simple rolling-mean-shift scan otherwise.
    """
    sd = series.std()
    x = ((series - series.mean()) / sd if sd and not np.isnan(sd) else series * 0).values

    if HAVE_RUPTURES:
        if len(x) < 4:
            return 0
        algo = rpt.Pelt(model="rbf").fit(x)
        breakpoints = algo.predict(pen=3)
        return len([b for b in breakpoints if b < len(x)])
    else:
        w = 6
        if len(x) < 2 * w:
            return 0
        n_breaks = 0
        for i in range(w, len(x) - w):
            before = x[i - w:i].mean()
            after = x[i:i + w].mean()
            if abs(after - before) > 1.5:
                n_breaks += 1
        return n_breaks


def mann_kendall(series):
    t = np.arange(len(series))
    tau, p_value = stats.kendalltau(t, series.values)
    return tau, p_value


# ----------------------------------------------------------------------
# Per-district feature computation
# ----------------------------------------------------------------------
def compute_district_features(g):
    g = g.sort_values(["year", "month"]).reset_index(drop=True)
    price = g["mean_modal_price"]
    min_p = g["mean_min_price"]
    max_p = g["mean_max_price"]
    spread = g["mean_price_spread"]
    is_imp = g["is_imputed"]
    n = len(g)
    t = np.arange(n)

    feats = {}

    # ---------------- Phase 2: Overall price features ----------------
    feats["price_mean_modal"] = price.mean()
    feats["price_median_modal"] = price.median()
    feats["price_mean_min"] = min_p.mean()
    feats["price_mean_max"] = max_p.mean()
    feats["price_mean_spread"] = spread.mean()

    diffs = price.diff().dropna()

    # ---------------- Phase 3: Price variability features ----------------
    std = price.std()
    mean_ = price.mean()
    feats["price_std"] = std
    feats["price_cv"] = (std / mean_ * 100) if mean_ else np.nan
    q1, q3 = price.quantile(0.25), price.quantile(0.75)
    feats["price_iqr"] = q3 - q1
    feats["price_max_monthly"] = price.max()
    feats["price_min_monthly"] = price.min()
    feats["price_rolling_vol_3m"] = price.rolling(3).std().mean()
    feats["price_rolling_vol_6m"] = price.rolling(6).std().mean()
    feats["price_max_monthly_change"] = diffs.abs().max() if len(diffs) else np.nan
    feats["price_mean_abs_monthly_change"] = diffs.abs().mean() if len(diffs) else np.nan

    # ---------------- Phase 4: Seasonal features ----------------
    month_avg = g.groupby("month")["mean_modal_price"].mean()
    month_std = g.groupby("month")["mean_modal_price"].std()
    for m in range(1, 13):
        name = MONTH_NAMES[m]
        feats[f"{name}_avg_price"] = month_avg.get(m, np.nan)
        feats[f"{name}_std_price"] = month_std.get(m, np.nan)

    ma_mean = month_avg.mean()
    feats["seasonal_amplitude"] = month_avg.max() - month_avg.min()
    feats["seasonal_cv"] = (month_avg.std() / ma_mean * 100) if ma_mean else np.nan

    full_var = price.var()
    feats["seasonal_strength"] = (month_avg.var() / full_var) if full_var else np.nan
    feats["peak_to_trough_ratio"] = (month_avg.max() / month_avg.min()) if month_avg.min() else np.nan

    peak_month = int(month_avg.idxmax())
    trough_month = int(month_avg.idxmin())
    feats["peak_price_month"] = peak_month
    feats["trough_price_month"] = trough_month

    yearly_peak_idx = g.groupby("year")["mean_modal_price"].idxmax()
    yearly_peak_months = g.loc[yearly_peak_idx, "month"]
    feats["peak_month_consistency"] = (yearly_peak_months == peak_month).mean() * 100

    # ---------------- Phase 5: Trend features ----------------
    lin_coefs = np.polyfit(t, price.values, 1)
    feats["price_trend_linear_slope"] = lin_coefs[0]

    if n > 2:
        quad_coefs = np.polyfit(t, price.values, 2)
        feats["price_trend_quadratic_coef"] = quad_coefs[0]
    else:
        feats["price_trend_quadratic_coef"] = np.nan

    yearly_avg = g.groupby("year")["mean_modal_price"].mean()
    n_years = len(yearly_avg)
    if n_years > 1 and yearly_avg.iloc[0]:
        feats["price_cagr"] = (yearly_avg.iloc[-1] / yearly_avg.iloc[0]) ** (1 / (n_years - 1)) - 1
    else:
        feats["price_cagr"] = np.nan

    tau, p_val = mann_kendall(price)
    feats["mann_kendall_tau"] = tau
    feats["mann_kendall_pvalue"] = p_val

    feats["structural_break_count"] = structural_break_count(price)

    # ---------------- Phase 7: Price dynamics features ----------------
    returns = price.pct_change().dropna()
    feats["monthly_return_mean"] = returns.mean() if len(returns) else np.nan
    feats["monthly_return_median"] = returns.median() if len(returns) else np.nan
    feats["monthly_return_volatility"] = returns.std() if len(returns) else np.nan
    feats["pct_positive_months"] = (returns > 0).mean() * 100 if len(returns) else np.nan

    feats["longest_increasing_streak"] = longest_run((diffs > 0).values)
    feats["longest_decreasing_streak"] = longest_run((diffs < 0).values)
    feats["num_turning_points"] = count_turning_points(diffs.values)
    feats["max_drawdown"] = max_drawdown_pct(price)

    # ---------------- Phase 8: Data quality features ----------------
    feats["pct_imputed"] = is_imp.mean() * 100

    year_imp = g.groupby("year")["is_imputed"]
    fully_missing_years = year_imp.apply(lambda s: bool(s.all()))
    any_imputed_years = year_imp.apply(lambda s: bool(s.any()))

    feats["observed_years"] = int((~fully_missing_years).sum())
    feats["imputed_years"] = int(any_imputed_years.sum())
    feats["missing_years_before_imputation"] = int(fully_missing_years.sum())

    observed_positions = np.where(is_imp.values == 0)[0]
    imputed_positions = np.where(is_imp.values == 1)[0]
    if len(observed_positions) > 0:
        first_obs, last_obs = observed_positions[0], observed_positions[-1]
        extrapolated = int(((imputed_positions < first_obs) | (imputed_positions > last_obs)).sum())
    else:
        extrapolated = n
    feats["fraction_extrapolated"] = extrapolated / n * 100 if n else np.nan

    return pd.Series(feats)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Price feature engineering (price_feature_engineering_plan.txt)")
    parser.add_argument("--base-dir", default="/Users/sthitpragye/Desktop/Finance/SURGE")
    parser.add_argument("--input-file", default=None,
                         help="Override path to district_monthly_prices.csv (default: <base-dir>/output/district_monthly_prices.csv)")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    in_file = args.input_file or os.path.join(args.base_dir, "phase0", "output", "district_monthly_prices.csv")
    out_file = os.path.join(args.base_dir, "phase0", "output", "district_price_features.csv")

    print("=" * 70)
    print("PRICE FEATURE ENGINEERING PIPELINE  (price_feature_engineering_plan.txt)")
    print("=" * 70)
    print("Note: Phase 6 (arrivals/market-activity features) and the arrivals-")
    print("related item in Phase 8 are skipped entirely - no arrivals data exists")
    print("in this project's price source (confirmed in price_preprocessing.py).")

    if not HAVE_RUPTURES:
        print("\n\u26a0 'ruptures' not installed - structural_break_count will use a simple")
        print("  fallback heuristic instead of PELT changepoint detection.")
        print("  For the plan's intended method, run: pip install ruptures")

    if not os.path.isfile(in_file):
        print(f"\nERROR: input file not found at {in_file}")
        print("Run price_preprocessing.py first to generate it.")
        sys.exit(1)

    print(f"\n[Phase 1] Loading & validating {in_file}")
    df = load_and_validate(in_file, args.start_year, args.end_year)
    print(f"  Loaded {len(df)} rows across {df['district'].nunique()} districts "
          f"({args.start_year}-{args.end_year}).")

    print("\n[Phases 2-5,7-8] Computing per-district features...")
    feature_rows = []
    for district, g in df.groupby("district"):
        feats = compute_district_features(g)
        feats["district"] = district
        feature_rows.append(feats)

    result = pd.DataFrame(feature_rows)
    cols = ["district"] + [c for c in result.columns if c != "district"]
    result = result[cols]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    result.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Districts processed : {len(result)}")
    print(f"Features generated  : {len(result.columns) - 1}")
    print(f"Written to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()