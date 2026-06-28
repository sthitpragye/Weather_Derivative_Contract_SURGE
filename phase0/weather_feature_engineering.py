"""
weather_feature_engineering.py
================================
PHASE 0 — Weather Feature Engineering for UP Weather Derivative Project

Reads the four preprocessed district-level weather files and produces
a single district-level feature matrix for clustering.

Inputs  (from phase0_output/)
------------------------------
  district_daily_weather.csv
  district_monthly_weather.csv
  district_yearly_weather.csv
  district_monthly_spi.csv

Output
------
  phase0_output/district_weather_features.csv   (~75 rows × ~100 cols)
  phase0_output/weather_feature_metadata.json

Phases
------
  1. Temperature Features
  2. Rainfall Features
  3. HDD Features   (base 18 as primary; 10 & 15 also computed)
  4. SPI Features
  5. Climate Stability Features
  6. Merge & Validate

Usage
-----
  python weather_feature_engineering.py
"""

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR   = "/Users/sthitpragye/Desktop/Finance/SURGE"
INPUT_DIR  = os.path.join(BASE_DIR, "phase0_output")
OUTPUT_DIR = INPUT_DIR                       # same folder, per plan

HDD_BASES  = [10, 15, 18]                   # must match preprocessing
PRIMARY_HDD = 18                            # named features use this base

# Season month definitions
WINTER_MONTHS     = [12, 1, 2]
SUMMER_MONTHS     = [3, 4, 5]
MONSOON_MONTHS    = [6, 7, 8, 9]
NON_MONSOON_MONTHS= [10, 11, 12, 1, 2, 3, 4, 5]
RABI_MONTHS       = [10, 11, 12, 1, 2, 3]

RAINY_DAY_MM      = 2.5
HEAVY_RAIN_MM     = 64.5
VERY_HEAVY_MM     = 115.6


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _slope(series: pd.Series) -> float:
    """OLS slope of series values vs integer index (year or month index)."""
    y = series.dropna().values
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _cagr(first: float, last: float, n: int) -> float:
    """Compound annual growth rate; returns NaN for non-positive values."""
    if n < 2 or first <= 0 or last <= 0:
        return np.nan
    return float((last / first) ** (1.0 / (n - 1)) - 1)


def _longest_run(mask: pd.Series) -> int:
    """Longest consecutive True streak in a boolean Series."""
    if mask.empty or not mask.any():
        return 0
    runs, cur = 0, 0
    for v in mask:
        cur = cur + 1 if v else 0
        runs = max(runs, cur)
    return int(runs)


def _spell_lengths(mask: pd.Series) -> list:
    """Returns list of lengths of all True-streaks in a boolean Series."""
    lengths, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur:
                lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


# ═══════════════════════════════════════════════════════════════════════════
# LOAD INPUTS
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs() -> tuple:
    section("LOADING INPUTS")
    paths = {
        "daily":   os.path.join(INPUT_DIR, "district_daily_weather.csv"),
        "monthly": os.path.join(INPUT_DIR, "district_monthly_weather.csv"),
        "yearly":  os.path.join(INPUT_DIR, "district_yearly_weather.csv"),
        "spi":     os.path.join(INPUT_DIR, "district_monthly_spi.csv"),
    }
    dfs = {}
    for key, path in paths.items():
        df = pd.read_csv(path)
        print(f"  {key:8s}: {len(df):>8,} rows  |  columns: {list(df.columns)}")
        dfs[key] = df

    # Parse date in daily
    dfs["daily"]["date"] = pd.to_datetime(dfs["daily"]["date"])

    return dfs["daily"], dfs["monthly"], dfs["yearly"], dfs["spi"]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — TEMPERATURE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_temperature_features(yearly: pd.DataFrame,
                                  monthly: pd.DataFrame) -> pd.DataFrame:
    section("PHASE 1 — TEMPERATURE FEATURES")

    records = []
    for district, yr in yearly.groupby("district"):
        yr = yr.sort_values("year")

        # ── Basic stats on yearly averages ───────────────────────────────
        mean_tmax  = yr["avg_tmax"].mean()
        mean_tmin  = yr["avg_tmin"].mean()
        mean_tmean = yr["avg_tmean"].mean()

        std_tmax   = yr["avg_tmax"].std(ddof=1)
        std_tmin   = yr["avg_tmin"].std(ddof=1)
        std_tmean  = yr["avg_tmean"].std(ddof=1)

        # ── Seasonal temperatures from monthly ───────────────────────────
        mo = monthly[monthly["district"] == district]

        def _season_mean(months):
            return mo.loc[mo["month"].isin(months), "avg_tmean"].mean()

        def _season_cv(months):
            # CV across all monthly observations in those months
            vals = mo.loc[mo["month"].isin(months), "avg_tmean"]
            return vals.std(ddof=1) / vals.mean() if vals.mean() != 0 else np.nan

        winter_mean = _season_mean(WINTER_MONTHS)
        summer_mean = _season_mean(SUMMER_MONTHS)

        # Diurnal range: mean daily (tmax - tmin) using monthly avgs
        diurnal = (mo["avg_tmax"] - mo["avg_tmin"]).mean()

        # Trend: slope of yearly avg_tmean vs year
        temp_trend = _slope(yr.set_index("year")["avg_tmean"])

        # CAGR of yearly avg_tmean
        first_val = yr["avg_tmean"].iloc[0]
        last_val  = yr["avg_tmean"].iloc[-1]
        temp_cagr = _cagr(first_val, last_val, len(yr))

        records.append({
            "district":                      district,
            "temperature_mean_tmax":         mean_tmax,
            "temperature_mean_tmin":         mean_tmin,
            "temperature_mean_tmean":        mean_tmean,
            "temperature_median_tmax":       yr["avg_tmax"].median(),
            "temperature_median_tmin":       yr["avg_tmin"].median(),
            "temperature_median_tmean":      yr["avg_tmean"].median(),
            "temperature_std_tmax":          std_tmax,
            "temperature_std_tmin":          std_tmin,
            "temperature_std_tmean":         std_tmean,
            "temperature_cv_tmax":           std_tmax  / mean_tmax  if mean_tmax  else np.nan,
            "temperature_cv_tmin":           std_tmin  / mean_tmin  if mean_tmin  else np.nan,
            "temperature_cv_tmean":          std_tmean / mean_tmean if mean_tmean else np.nan,
            "temperature_max_tmax":          yr["max_tmax"].max(),
            "temperature_min_tmin":          yr["min_tmin"].min(),
            "temperature_max_tmean":         yr["avg_tmean"].max(),
            "temperature_min_tmean":         yr["avg_tmean"].min(),
            "temperature_avg_diurnal_range": diurnal,
            "temperature_mean_winter":       winter_mean,
            "temperature_mean_summer":       summer_mean,
            "temperature_cv_winter":         _season_cv(WINTER_MONTHS),
            "temperature_cv_summer":         _season_cv(SUMMER_MONTHS),
            "temperature_summer_winter_diff":summer_mean - winter_mean,
            "temperature_trend":             temp_trend,
            "temperature_cagr":              temp_cagr,
        })

    df_temp = pd.DataFrame(records)
    print(f"  Temperature features: {len(df_temp)} districts × "
          f"{df_temp.shape[1]-1} features")
    return df_temp


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — RAINFALL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_rainfall_features(yearly: pd.DataFrame,
                               monthly: pd.DataFrame,
                               daily: pd.DataFrame) -> pd.DataFrame:
    section("PHASE 2 — RAINFALL FEATURES")

    records = []
    for district, yr in yearly.groupby("district"):
        yr  = yr.sort_values("year")
        mo  = monthly[monthly["district"] == district]
        da  = daily[daily["district"] == district].sort_values("date")

        annual_rain = yr["annual_rainfall"]
        mean_annual = annual_rain.mean()
        std_annual  = annual_rain.std(ddof=1)

        # ── Monthly rainfall stats ────────────────────────────────────────
        monthly_rain = mo["monthly_rainfall"]

        # ── Seasonal ─────────────────────────────────────────────────────
        monsoon_by_yr = (mo[mo["month"].isin(MONSOON_MONTHS)]
                         .groupby("year")["monthly_rainfall"].sum())
        non_monsoon_by_yr = (mo[mo["month"].isin(NON_MONSOON_MONTHS)]
                              .groupby("year")["monthly_rainfall"].sum())

        mean_monsoon     = monsoon_by_yr.mean()
        mean_non_monsoon = non_monsoon_by_yr.mean()
        monsoon_share    = mean_monsoon / mean_annual if mean_annual > 0 else np.nan
        monsoon_cv       = (monsoon_by_yr.std(ddof=1) / monsoon_by_yr.mean()
                            if monsoon_by_yr.mean() > 0 else np.nan)

        # ── Peak rainfall month (long-term mean by calendar month) ───────
        month_means       = mo.groupby("month")["monthly_rainfall"].mean()
        peak_rain_month   = int(month_means.idxmax())

        # ── Rainfall Concentration Index: 12 × Σ(Pi²) / (ΣPi)² ─────────
        pi  = month_means.values
        sp  = pi.sum()
        rci = (12 * (pi**2).sum() / (sp**2)) if sp > 0 else np.nan

        # ── Rainy-day & extreme-day frequencies (from yearly agg) ────────
        total_days         = yr["annual_rainfall"].count() * 365  # approx
        avg_rainy_days     = yr["rainy_days"].mean()
        total_da_days      = len(da)
        wet_day_freq       = (da["rainfall"] >= RAINY_DAY_MM).sum() / total_da_days
        heavy_rain_freq    = (da["rainfall"] >= HEAVY_RAIN_MM).sum() / total_da_days
        very_heavy_freq    = (da["rainfall"] >= VERY_HEAVY_MM).sum() / total_da_days

        # ── Longest dry spell (daily) ─────────────────────────────────────
        dry_mask         = da["rainfall"] < RAINY_DAY_MM
        longest_dry_spell = _longest_run(dry_mask)

        # ── Trend & CAGR of annual rainfall ─────────────────────────────
        rain_trend = _slope(yr.set_index("year")["annual_rainfall"])
        rain_cagr  = _cagr(annual_rain.iloc[0], annual_rain.iloc[-1], len(yr))

        # ── Rainfall Anomaly Index ────────────────────────────────────────
        rain_anomaly_index = ((annual_rain - mean_annual) / std_annual).mean() \
                              if std_annual > 0 else 0.0

        records.append({
            "district":                      district,
            "rainfall_mean_annual":          mean_annual,
            "rainfall_median_annual":        annual_rain.median(),
            "rainfall_std_annual":           std_annual,
            "rainfall_cv_annual":            std_annual / mean_annual if mean_annual > 0 else np.nan,
            "rainfall_max_annual":           annual_rain.max(),
            "rainfall_mean_monthly":         monthly_rain.mean(),
            "rainfall_std_monthly":          monthly_rain.std(ddof=1),
            "rainfall_cv_monthly":           (monthly_rain.std(ddof=1) / monthly_rain.mean()
                                              if monthly_rain.mean() > 0 else np.nan),
            "rainfall_max_monthly":          monthly_rain.max(),
            "rainfall_min_monthly":          monthly_rain.min(),
            "rainfall_mean_monsoon":         mean_monsoon,
            "rainfall_mean_non_monsoon":     mean_non_monsoon,
            "rainfall_monsoon_share":        monsoon_share,
            "rainfall_monsoon_cv":           monsoon_cv,
            "rainfall_peak_month":           peak_rain_month,
            "rainfall_concentration_index":  rci,
            "rainfall_avg_rainy_days":       avg_rainy_days,
            "rainfall_wet_day_frequency":    wet_day_freq,
            "rainfall_heavy_rain_frequency": heavy_rain_freq,
            "rainfall_very_heavy_frequency": very_heavy_freq,
            "rainfall_longest_dry_spell":    longest_dry_spell,
            "rainfall_trend":                rain_trend,
            "rainfall_cagr":                 rain_cagr,
            "rainfall_anomaly_index":        rain_anomaly_index,
        })

    df_rain = pd.DataFrame(records)
    print(f"  Rainfall features: {len(df_rain)} districts × "
          f"{df_rain.shape[1]-1} features")
    return df_rain


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — HDD FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_hdd_features(yearly: pd.DataFrame,
                          monthly: pd.DataFrame) -> pd.DataFrame:
    section("PHASE 3 — HDD FEATURES")

    all_records = []
    for base in HDD_BASES:
        col = f"HDD_{base}"
        records = []

        for district, yr in yearly.groupby("district"):
            yr   = yr.sort_values("year")
            mo   = monthly[monthly["district"] == district]
            hdd  = yr[col]
            mean_hdd = hdd.mean()

            # Winter & Summer HDD shares (from monthly)
            winter_hdd = mo.loc[mo["month"].isin(WINTER_MONTHS), col].sum()
            summer_hdd = mo.loc[mo["month"].isin(SUMMER_MONTHS), col].sum()
            annual_hdd_total = mo[col].sum()

            # High-HDD-year frequency: years above long-term mean
            high_hdd_freq = (hdd > mean_hdd).sum() / len(hdd) if len(hdd) > 0 else np.nan

            # HDD Persistence: longest run of years above long-term mean
            above_mean_mask = pd.Series(hdd > mean_hdd).reset_index(drop=True)
            hdd_persistence = _longest_run(above_mean_mask)

            std_hdd = hdd.std(ddof=1)

            records.append({
                "district":                      district,
                f"hdd{base}_mean":               mean_hdd,
                f"hdd{base}_median":             hdd.median(),
                f"hdd{base}_std":                std_hdd,
                f"hdd{base}_cv":                 std_hdd / mean_hdd if mean_hdd > 0 else np.nan,
                f"hdd{base}_max":                hdd.max(),
                f"hdd{base}_min":                hdd.min(),
                f"hdd{base}_winter_share":       (winter_hdd / annual_hdd_total
                                                  if annual_hdd_total > 0 else np.nan),
                f"hdd{base}_summer_share":       (summer_hdd / annual_hdd_total
                                                  if annual_hdd_total > 0 else np.nan),
                f"hdd{base}_nonzero_freq":       (hdd > 0).sum() / len(hdd),
                f"hdd{base}_high_freq":          high_hdd_freq,
                f"hdd{base}_trend":              _slope(yr.set_index("year")[col]),
                f"hdd{base}_variability_index":  (std_hdd / mean_hdd
                                                  if mean_hdd > 0 else np.nan),
                f"hdd{base}_persistence":        hdd_persistence,
                f"hdd{base}_anomaly":            (hdd - mean_hdd).mean(),
            })

        all_records.append(pd.DataFrame(records))

    # Merge all bases on district
    df_hdd = all_records[0]
    for extra in all_records[1:]:
        df_hdd = df_hdd.merge(extra, on="district", how="left")

    print(f"  HDD features: {len(df_hdd)} districts × "
          f"{df_hdd.shape[1]-1} features  "
          f"(bases: {HDD_BASES})")
    return df_hdd


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — SPI FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_spi_features(spi_df: pd.DataFrame) -> pd.DataFrame:
    section("PHASE 4 — SPI FEATURES")

    # Use SPI1 as the primary SPI series (monthly; most complete)
    records = []
    for district, grp in spi_df.groupby("district"):
        grp  = grp.sort_values(["year", "month"]).reset_index(drop=True)
        s1   = grp["SPI1"].dropna()

        # ── Basic stats ───────────────────────────────────────────────────
        mean_spi   = s1.mean()
        median_spi = s1.median()
        std_spi    = s1.std(ddof=1)

        # ── Drought / wet frequencies ─────────────────────────────────────
        total = len(s1)
        mild_drought_freq     = ((s1 > -1.0)   & (s1 <= -0.5)).sum() / total
        moderate_drought_freq = ((s1 > -1.5)   & (s1 <= -1.0)).sum() / total
        severe_drought_freq   = (s1 <= -1.5).sum() / total
        wet_freq              = (s1 >  1.0).sum() / total

        # ── Spell lengths ─────────────────────────────────────────────────
        drought_mask = (grp["SPI1"] < -1.0).reset_index(drop=True)
        wet_mask     = (grp["SPI1"] >  1.0).reset_index(drop=True)

        longest_drought = _longest_run(drought_mask)
        longest_wet     = _longest_run(wet_mask)

        drought_spells  = _spell_lengths(drought_mask)
        avg_drought_dur = float(np.mean(drought_spells)) if drought_spells else 0.0

        # ── Seasonal SPI ──────────────────────────────────────────────────
        mean_monsoon_spi = grp.loc[grp["month"].isin(MONSOON_MONTHS), "SPI1"].mean()
        mean_rabi_spi    = grp.loc[grp["month"].isin(RABI_MONTHS),    "SPI1"].mean()

        records.append({
            "district":                   district,
            "spi1_mean":                  mean_spi,
            "spi1_median":                median_spi,
            "spi1_std":                   std_spi,
            "spi1_mild_drought_freq":     mild_drought_freq,
            "spi1_moderate_drought_freq": moderate_drought_freq,
            "spi1_severe_drought_freq":   severe_drought_freq,
            "spi1_wet_freq":              wet_freq,
            "spi1_longest_drought_spell": longest_drought,
            "spi1_longest_wet_spell":     longest_wet,
            "spi1_avg_drought_duration":  avg_drought_dur,
            "spi1_min":                   s1.min(),
            "spi1_max":                   s1.max(),
            "spi1_mean_monsoon":          mean_monsoon_spi,
            "spi1_mean_rabi":             mean_rabi_spi,
        })

    df_spi = pd.DataFrame(records)
    print(f"  SPI features: {len(df_spi)} districts × "
          f"{df_spi.shape[1]-1} features")
    return df_spi


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — CLIMATE STABILITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_climate_stability_features(df_temp: pd.DataFrame,
                                        df_rain: pd.DataFrame,
                                        df_hdd:  pd.DataFrame,
                                        df_spi:  pd.DataFrame,
                                        yearly:  pd.DataFrame) -> pd.DataFrame:
    section("PHASE 5 — CLIMATE STABILITY FEATURES")

    # Pull the CV/Std columns already computed in earlier phases
    base  = df_temp[["district", "temperature_cv_tmean"]].copy()
    base  = base.merge(df_rain[["district", "rainfall_cv_annual"]], on="district")
    base  = base.merge(df_hdd[[f"district", f"hdd{PRIMARY_HDD}_cv"]], on="district")
    base  = base.merge(df_spi[["district", "spi1_std"]], on="district")

    records = []
    for district, yr in yearly.groupby("district"):
        row_base = base[base["district"] == district].iloc[0]

        temp_cv  = row_base["temperature_cv_tmean"]
        rain_cv  = row_base["rainfall_cv_annual"]
        hdd_cv   = row_base[f"hdd{PRIMARY_HDD}_cv"]
        spi_std  = row_base["spi1_std"]

        temp_stability = 1 / (1 + temp_cv) if not np.isnan(temp_cv) else np.nan
        rain_stability = 1 / (1 + rain_cv) if not np.isnan(rain_cv) else np.nan
        hdd_stability  = 1 / (1 + hdd_cv)  if not np.isnan(hdd_cv)  else np.nan
        spi_stability  = 1 / (1 + spi_std)  if not np.isnan(spi_std)  else np.nan

        # Variances
        temp_var = yr["avg_tmean"].var(ddof=1)
        rain_var = yr["annual_rainfall"].var(ddof=1)
        hdd_var  = yr[f"HDD_{PRIMARY_HDD}"].var(ddof=1)

        # Extreme years
        warmest_year = int(yr.loc[yr["avg_tmean"].idxmax(),  "year"])
        coldest_year = int(yr.loc[yr["avg_tmean"].idxmin(),  "year"])
        wettest_year = int(yr.loc[yr["annual_rainfall"].idxmax(), "year"])
        driest_year  = int(yr.loc[yr["annual_rainfall"].idxmin(), "year"])

        # Climate Variability Score:
        # Normalised (0–1) combination of temp, rain, HDD, SPI variability.
        # Each component normalised by its theoretical max CV in the dataset.
        components = [temp_cv, rain_cv, hdd_cv, spi_std]
        valid      = [v for v in components if not np.isnan(v)]
        clim_var_score = float(np.mean(valid)) if valid else np.nan

        records.append({
            "district":                   district,
            "climate_temp_stability":     temp_stability,
            "climate_rain_stability":     rain_stability,
            "climate_hdd_stability":      hdd_stability,
            "climate_spi_stability":      spi_stability,
            "climate_temp_variance":      temp_var,
            "climate_rain_variance":      rain_var,
            "climate_hdd_variance":       hdd_var,
            "climate_warmest_year":       warmest_year,
            "climate_coldest_year":       coldest_year,
            "climate_wettest_year":       wettest_year,
            "climate_driest_year":        driest_year,
            "climate_variability_score":  clim_var_score,
        })

    df_clim = pd.DataFrame(records)
    print(f"  Climate stability features: {len(df_clim)} districts × "
          f"{df_clim.shape[1]-1} features")
    return df_clim


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — MERGE & VALIDATE
# ═══════════════════════════════════════════════════════════════════════════

def merge_and_validate(df_temp:  pd.DataFrame,
                        df_rain:  pd.DataFrame,
                        df_hdd:   pd.DataFrame,
                        df_spi:   pd.DataFrame,
                        df_clim:  pd.DataFrame,
                        analysis_period: dict) -> pd.DataFrame:
    section("PHASE 6 — FEATURE MERGING & VALIDATION")

    # ── Merge in plan-specified order ─────────────────────────────────────
    feat = df_temp.copy()
    for df, label in [(df_rain, "Rainfall"),
                      (df_hdd,  "HDD"),
                      (df_spi,  "SPI"),
                      (df_clim, "Climate Stability")]:
        before = feat.shape[1]
        feat   = feat.merge(df, on="district", how="left")
        print(f"  + {label:22s}: {feat.shape[1] - before:3d} columns  "
              f"→ total {feat.shape[1]-1} features")

    # ── Validation checks ─────────────────────────────────────────────────
    print("\n  Validation:")

    # 1. One row per district
    assert feat["district"].nunique() == len(feat), \
        "FAIL: duplicate districts after merge"
    print(f"    ✓ Every district appears exactly once ({len(feat)} districts)")

    # 2. No duplicate columns
    dupe_cols = feat.columns[feat.columns.duplicated()].tolist()
    assert not dupe_cols, f"FAIL: duplicate columns: {dupe_cols}"
    print(f"    ✓ No duplicate column names")

    # 3. All feature columns are numeric
    non_num = [c for c in feat.columns
               if c != "district" and not pd.api.types.is_numeric_dtype(feat[c])]
    assert not non_num, f"FAIL: non-numeric columns: {non_num}"
    print(f"    ✓ All feature columns are numeric")

    # 4. Missing values
    miss = feat.drop(columns=["district"]).isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print(f"    ✓ No missing values")
    else:
        print(f"    ! Missing values detected (filling with column median):")
        for col, cnt in miss.items():
            print(f"        {col}: {cnt}")
            feat[col] = feat[col].fillna(feat[col].median())

    # 5. Final shape
    n_features = feat.shape[1] - 1
    print(f"\n  Final shape: {len(feat)} districts × {n_features} features")
    if not (90 <= n_features <= 110):
        print(f"  NOTE: expected 90–100 features; got {n_features}. "
              f"Review plan if this is unexpected.")

    # ── Enforce stable column order ───────────────────────────────────────
    non_district = sorted([c for c in feat.columns if c != "district"])
    feat = feat[["district"] + non_district]

    # Restore logical group order (temp → rain → hdd → spi → climate)
    def _group_order(col: str) -> int:
        if col.startswith("temperature"):  return 0
        if col.startswith("rainfall"):     return 1
        if col.startswith("hdd"):          return 2
        if col.startswith("spi"):          return 3
        if col.startswith("climate"):      return 4
        return 5

    ordered_features = sorted(non_district, key=lambda c: (_group_order(c), c))
    feat = feat[["district"] + ordered_features]

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════════════════════════════

def build_metadata(feat: pd.DataFrame, analysis_period: dict) -> dict:
    miss = feat.drop(columns=["district"]).isna().sum()
    miss_dict = {col: int(v) for col, v in miss.items() if v > 0}

    return {
        "generated_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        "analysis_period": analysis_period,
        "n_districts":     int(len(feat)),
        "n_features":      int(feat.shape[1] - 1),
        "districts":       sorted(feat["district"].tolist()),
        "feature_names":   [c for c in feat.columns if c != "district"],
        "missing_values":  miss_dict,
        "hdd_bases_used":  HDD_BASES,
        "primary_hdd_base": PRIMARY_HDD,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("WEATHER FEATURE ENGINEERING — UP Weather Derivative Project")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    daily, monthly, yearly, spi = load_inputs()

    analysis_period = {
        "start": int(yearly["year"].min()),
        "end":   int(yearly["year"].max()),
    }

    # ── Feature phases ────────────────────────────────────────────────────
    df_temp = compute_temperature_features(yearly, monthly)
    df_rain = compute_rainfall_features(yearly, monthly, daily)
    df_hdd  = compute_hdd_features(yearly, monthly)
    df_spi  = compute_spi_features(spi)
    df_clim = compute_climate_stability_features(
                  df_temp, df_rain, df_hdd, df_spi, yearly)

    # ── Merge & validate ──────────────────────────────────────────────────
    feat = merge_and_validate(df_temp, df_rain, df_hdd, df_spi, df_clim,
                               analysis_period)

    # ── Save outputs ──────────────────────────────────────────────────────
    section("SAVING OUTPUTS")

    feat_path = os.path.join(OUTPUT_DIR, "district_weather_features.csv")
    feat.to_csv(feat_path, index=False)
    print(f"  Saved → {feat_path}")

    meta      = build_metadata(feat, analysis_period)
    meta_path = os.path.join(OUTPUT_DIR, "weather_feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    section("FEATURE ENGINEERING COMPLETE")
    print(f"""
  Output : {feat_path}
  Shape  : {feat.shape[0]} districts × {feat.shape[1]-1} features

  Feature groups
  ├── Temperature      : {sum(1 for c in feat.columns if c.startswith('temperature'))}
  ├── Rainfall         : {sum(1 for c in feat.columns if c.startswith('rainfall'))}
  ├── HDD ({HDD_BASES})  : {sum(1 for c in feat.columns if c.startswith('hdd'))}
  ├── SPI              : {sum(1 for c in feat.columns if c.startswith('spi'))}
  └── Climate Stability: {sum(1 for c in feat.columns if c.startswith('climate'))}

  Period : {analysis_period['start']}–{analysis_period['end']}

  Next step → merge with district_price_features.csv and
              district_apy_features.csv using the 'district' column
              to produce final_clustering_dataset.csv
""")
    return feat


if __name__ == "__main__":
    feat = main()