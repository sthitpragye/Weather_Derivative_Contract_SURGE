"""
weather_preprocessing.py
========================
PHASE 0 — Weather Preprocessing for UP Weather Derivative Project

Transforms raw IMD gridded weather data into clean district-level
datasets ready for feature engineering.

Phases
------
1. Load & Validate Data
2. Grid-to-District Mapping
3. Daily District Weather Panel  → district_daily_weather.csv
4. Monthly Aggregation           → district_monthly_weather.csv
5. Yearly Aggregation            → district_yearly_weather.csv
6. SPI Construction              → district_monthly_spi.csv
7. Metadata                      → weather_metadata.json

Inputs
------
  Max_temp_data/   : IMD 1° GRD files (Maxtemp_MaxT_<year>.GRD)
  Min_temp_data/   : IMD 1° GRD files (Mintemp_MinT_<year>.GRD)
  Rainfall_data/   : IMD 0.25° NetCDF files (RF25_ind<year>_rfp25.nc)
  gadm41_IND_shp/  : GADM admin-level-2 shapefile

Usage
-----
  python weather_preprocessing.py

Adjust the CONFIG block below for your local paths.
"""

import os
import re
import glob
import json
import array
import warnings
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy import stats
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths; everything else is automatic
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR       = "/Users/sthitpragye/Desktop/Finance/SURGE"
MAXTEMP_DIR    = os.path.join(BASE_DIR, "Max_temp_data")
MINTEMP_DIR    = os.path.join(BASE_DIR, "Min_temp_data")
RAINFALL_DIR   = os.path.join(BASE_DIR, "Rainfall_data")
SHAPEFILE_PATH = os.path.join(BASE_DIR, "gadm41_IND_shp", "gadm41_IND_2.shp")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")

STATE_NAME     = "Uttar Pradesh"
HDD_BASES      = [10, 15, 18]      # °C
SPI_TIMESCALES = [1, 3]            # months
RAINY_DAY_MM   = 2.5               # mm — threshold for a rainy day

# IMD temperature grid constants (1° resolution, confirmed via imdlib)
TEMP_NLAT, TEMP_NLON   = 31, 31
TEMP_LAT = np.linspace(7.5,  37.5, TEMP_NLAT)
TEMP_LON = np.linspace(67.5, 97.5, TEMP_NLON)
TEMP_MISSING_THRESHOLD = 90.0      # sentinel ≥ 99.9 treated as missing

# IMD rainfall grid constants (0.25° resolution, confirmed from .nc files)
RAIN_NLAT, RAIN_NLON = 129, 135
RAIN_LAT = np.linspace(6.5,  38.5, RAIN_NLAT)
RAIN_LON = np.linspace(66.5, 100.0, RAIN_NLON)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def parse_year_from_filename(fname: str) -> int:
    match = re.search(r"(19|20)\d{2}", os.path.basename(fname))
    if not match:
        raise ValueError(f"Cannot parse year from: {fname}")
    return int(match.group(0))


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def subsection(title: str) -> None:
    print(f"\n  — {title}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════════════════

def read_grd_year(filepath: str, year: int) -> np.ndarray:
    """
    Reads one year of IMD temperature GRD data.

    Returns np.ndarray of shape (n_days, TEMP_NLAT, TEMP_NLON) with
    NaN where missing.

    Validations:
      - File size must match expected bytes for the year's day count.
      - Values are checked against physical range for India (-10 to 55°C).
      - Byte order is auto-corrected if the range test fails.
    """
    n_days = 366 if is_leap(year) else 365
    expected_floats = n_days * TEMP_NLAT * TEMP_NLON
    expected_bytes  = expected_floats * 4

    actual_bytes = os.path.getsize(filepath)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"[{os.path.basename(filepath)}] Size mismatch: "
            f"{actual_bytes} bytes found, {expected_bytes} expected "
            f"({n_days} days × {TEMP_NLAT}×{TEMP_NLON} grid)."
        )

    raw = array.array("f")
    with open(filepath, "rb") as f:
        raw.fromfile(f, expected_floats)
    data = np.frombuffer(raw, dtype=np.float32).copy()
    data = data.reshape((n_days, TEMP_NLAT, TEMP_NLON))

    # Physical range sanity check; auto-retry with byteswap
    sample = data[data < TEMP_MISSING_THRESHOLD]
    if sample.size > 0 and (sample.min() < -30 or sample.max() > 60):
        data_bs = data.copy()
        data_bs.byteswap(inplace=True)
        sample_bs = data_bs[data_bs < TEMP_MISSING_THRESHOLD]
        if sample_bs.size > 0 and -30 <= sample_bs.min() and sample_bs.max() <= 60:
            print(f"    [{os.path.basename(filepath)}] Byte order corrected.")
            data = data_bs
        else:
            print(f"    WARNING [{os.path.basename(filepath)}]: range "
                  f"[{sample.min():.1f}, {sample.max():.1f}] seems implausible.")

    data = np.where(data >= TEMP_MISSING_THRESHOLD, np.nan, data)
    return data


def load_all_temp_years(directory: str, label: str) -> dict:
    """Loads all GRD files from a directory into {year: ndarray}."""
    files = sorted(
        glob.glob(os.path.join(directory, "*.GRD")) +
        glob.glob(os.path.join(directory, "*.grd"))
    )
    if not files:
        raise FileNotFoundError(f"No .GRD files found in: {directory}")

    print(f"  Loading {label} ({len(files)} files)...")
    out = {}
    for fp in files:
        year = parse_year_from_filename(fp)
        try:
            out[year] = read_grd_year(fp, year)
        except Exception as e:
            print(f"    SKIPPED {os.path.basename(fp)}: {e}")

    print(f"    Loaded {len(out)} years: {min(out)}–{max(out)}")
    return out


def load_all_rain_years(directory: str) -> dict:
    """Loads all rainfall NetCDF files into {year: ndarray (days, lat, lon)}."""
    files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in: {directory}")

    print(f"  Loading Rainfall ({len(files)} files)...")
    out = {}
    for fp in files:
        year = parse_year_from_filename(fp)
        try:
            ds  = xr.open_dataset(fp)
            arr = ds["RAINFALL"].values                 # (time, lat, lon)
            arr = np.where(arr <= -998, np.nan, arr)
            out[year] = arr
            ds.close()
        except Exception as e:
            print(f"    SKIPPED {os.path.basename(fp)}: {e}")

    print(f"    Loaded {len(out)} years: {min(out)}–{max(out)}")
    return out


def validate_and_intersect(tmax_years: dict, tmin_years: dict,
                            rain_years: dict) -> list:
    """
    Validates leap-year day counts for each loaded year and returns
    the sorted list of years present in all three datasets.
    """
    subsection("Validating day counts per year")
    for label, ydict in [("Tmax", tmax_years), ("Tmin", tmin_years),
                          ("Rain", rain_years)]:
        for yr, arr in ydict.items():
            expected = 366 if is_leap(yr) else 365
            actual   = arr.shape[0]
            if actual != expected:
                print(f"    WARNING {label} {yr}: {actual} days, expected {expected}")

    common = sorted(set(tmax_years) & set(tmin_years) & set(rain_years))
    print(f"  Years with complete Tmax+Tmin+Rain: "
          f"{len(common)} ({min(common)}–{max(common)})")
    return common


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — GRID-TO-DISTRICT MAPPING
# ═══════════════════════════════════════════════════════════════════════════

def load_up_districts(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Loads UP district boundaries from a GADM shapefile.
    Auto-detects state and district column names so the function
    works regardless of GADM version or column name drift.
    """
    gdf = gpd.read_file(shapefile_path)
    print(f"  Shapefile: {len(gdf)} features, columns: {list(gdf.columns)}")

    geom_col     = gdf.geometry.name
    non_geom     = [c for c in gdf.columns if c != geom_col]

    # Locate state column
    state_col = None
    if "NAME_1" in gdf.columns and gdf["NAME_1"].astype(str).str.contains(
            STATE_NAME, case=False, na=False).any():
        state_col = "NAME_1"
    else:
        for col in non_geom:
            if gdf[col].astype(str).str.contains(STATE_NAME, case=False, na=False).any():
                state_col = col
                break

    if state_col is None:
        raise ValueError(
            f"Column containing '{STATE_NAME}' not found. "
            f"Available: {list(gdf.columns)}"
        )

    up = gdf[gdf[state_col].astype(str).str.contains(
        STATE_NAME, case=False, na=False)].copy()

    # Locate district column
    district_col = None
    if "NAME_2" in up.columns:
        district_col = "NAME_2"
    else:
        candidates = [c for c in up.columns if c not in (state_col, geom_col)]
        for c in candidates:
            vals = up[c].astype(str)
            if vals.nunique() == len(up) and vals.str.len().mean() > 3:
                district_col = c
                break
        if district_col is None and candidates:
            district_col = candidates[0]

    if district_col is None:
        raise ValueError(f"District column not found. Columns: {list(up.columns)}")

    print(f"  State col = '{state_col}', District col = '{district_col}'")
    print(f"  {len(up)} UP districts loaded")

    up = up.rename(columns={district_col: "district"})[["district", "geometry"]]
    return up.to_crs(epsg=4326)


def build_grid_district_map(districts: gpd.GeoDataFrame,
                             lat_arr: np.ndarray,
                             lon_arr: np.ndarray,
                             label: str = "") -> pd.DataFrame:
    """
    Assigns every grid cell centroid to the district it falls within.
    Districts with no centroid inside them (common at coarse 1° resolution)
    fall back to their nearest grid cell.

    Returns DataFrame with columns: lat_idx, lon_idx, district.
    """
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    lat_idx, lon_idx   = np.meshgrid(np.arange(len(lat_arr)),
                                      np.arange(len(lon_arr)), indexing="ij")

    points = gpd.GeoDataFrame({
        "lat_idx": lat_idx.ravel(),
        "lon_idx": lon_idx.ravel(),
        "lat":     lat_grid.ravel(),
        "lon":     lon_grid.ravel(),
    }, geometry=[Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel())],
       crs="EPSG:4326")

    joined = gpd.sjoin(points, districts, how="left", predicate="within")
    mapped = joined.dropna(subset=["district"])[["lat_idx", "lon_idx", "district"]].copy()

    # Nearest-cell fallback for districts with no grid point inside them
    missing = set(districts["district"]) - set(mapped["district"].unique())
    if missing:
        print(f"  {len(missing)} districts ({label}) need nearest-cell fallback:")
        for d in sorted(missing):
            centroid = districts.loc[districts["district"] == d,
                                     "geometry"].iloc[0].centroid
            dist_sq  = (points["lon"] - centroid.x)**2 + \
                       (points["lat"] - centroid.y)**2
            nearest  = points.iloc[dist_sq.idxmin()]
            mapped   = pd.concat([mapped, pd.DataFrame([{
                "lat_idx":  int(nearest["lat_idx"]),
                "lon_idx":  int(nearest["lon_idx"]),
                "district": d,
            }])], ignore_index=True)
            print(f"    {d} → ({nearest['lat']:.2f}°N, {nearest['lon']:.2f}°E)")

    return mapped.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — DAILY DISTRICT WEATHER PANEL
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_grid_to_district(year_data: dict,
                                grid_map: pd.DataFrame,
                                year: int) -> pd.DataFrame:
    """
    Spatially averages one year's gridded array (days, nlat, nlon)
    to district-daily means using the precomputed mapping table.

    Returns a long DataFrame: date | district | value
    """
    arr    = year_data[year]
    n_days = arr.shape[0]
    dates  = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    rows = []
    for district, grp in grid_map.groupby("district"):
        li   = grp["lat_idx"].values
        loi  = grp["lon_idx"].values
        vals = np.nanmean(arr[:, li, loi], axis=1)   # (n_days,)
        rows.append(pd.DataFrame({"date": dates, "district": district, "value": vals}))

    return pd.concat(rows, ignore_index=True)


def build_daily_panel(tmax_years: dict, tmin_years: dict, rain_years: dict,
                      temp_map: pd.DataFrame, rain_map: pd.DataFrame,
                      common_years: list) -> pd.DataFrame:
    """
    Aggregates each year for all three variables to district-daily,
    merges them and computes Tmean.
    """
    tmax_parts, tmin_parts, rain_parts = [], [], []

    for yr in common_years:
        tmax_parts.append(aggregate_grid_to_district(tmax_years, temp_map, yr)
                          .rename(columns={"value": "tmax"}))
        tmin_parts.append(aggregate_grid_to_district(tmin_years, temp_map, yr)
                          .rename(columns={"value": "tmin"}))
        rain_parts.append(aggregate_grid_to_district(rain_years, rain_map, yr)
                          .rename(columns={"value": "rainfall"}))
        print(f"    Aggregated {yr}")

    tmax_df = pd.concat(tmax_parts, ignore_index=True)
    tmin_df = pd.concat(tmin_parts, ignore_index=True)
    rain_df = pd.concat(rain_parts, ignore_index=True)

    daily = (tmax_df
             .merge(tmin_df,  on=["date", "district"])
             .merge(rain_df,  on=["date", "district"]))

    daily["tmean"]    = (daily["tmax"] + daily["tmin"]) / 2
    daily["year"]     = daily["date"].dt.year
    daily["month"]    = daily["date"].dt.month
    daily["day"]      = daily["date"].dt.day

    # Clip rainfall negatives (grid interpolation artefacts)
    daily["rainfall"] = daily["rainfall"].clip(lower=0)

    # Enforce column order per plan
    col_order = ["district", "date", "year", "month", "day",
                 "tmax", "tmin", "tmean", "rainfall"]
    return daily[col_order].sort_values(["district", "date"]).reset_index(drop=True)


def validate_daily_panel(daily: pd.DataFrame) -> None:
    """Basic validation on the assembled daily panel."""
    subsection("Daily panel validation")

    n_neg_rain = (daily["rainfall"] < 0).sum()
    n_neg_temp = (daily[["tmax", "tmin", "tmean"]] < -50).any(axis=1).sum()
    n_miss = daily[["tmax", "tmin", "tmean", "rainfall"]].isna().sum()

    print(f"    Rows            : {len(daily):,}")
    print(f"    Districts       : {daily['district'].nunique()}")
    print(f"    Date range      : {daily['date'].min()} → {daily['date'].max()}")
    print(f"    Negative rain   : {n_neg_rain}")
    print(f"    Implausible temp: {n_neg_temp}")
    print(f"    Missing values  :\n{n_miss.to_string()}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — MONTHLY AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the daily panel to monthly statistics per the plan.

    Output columns (per plan):
        district, year, month,
        avg_tmax, avg_tmin, avg_tmean,
        monthly_rainfall, rainy_days,
        HDD_10, HDD_15, HDD_18
    """
    grp = daily.groupby(["district", "year", "month"])

    monthly = grp.agg(
        avg_tmax         = ("tmax",     "mean"),
        avg_tmin         = ("tmin",     "mean"),
        avg_tmean        = ("tmean",    "mean"),
        monthly_rainfall = ("rainfall", "sum"),
    ).reset_index()

    # Rainy days (rainfall >= threshold)
    daily["_rainy"] = (daily["rainfall"] >= RAINY_DAY_MM).astype(int)
    rainy = (daily.groupby(["district", "year", "month"])["_rainy"]
             .sum().reset_index().rename(columns={"_rainy": "rainy_days"}))
    daily.drop(columns=["_rainy"], inplace=True)

    monthly = monthly.merge(rainy, on=["district", "year", "month"])

    # HDD per base: monthly sum of max(0, base - tmean)
    for base in HDD_BASES:
        col = f"HDD_{base}"
        daily[col] = np.maximum(0, base - daily["tmean"])
        hdd_m = (daily.groupby(["district", "year", "month"])[col]
                 .sum().reset_index())
        monthly = monthly.merge(hdd_m, on=["district", "year", "month"])
        daily.drop(columns=[col], inplace=True)

    monthly = monthly.sort_values(["district", "year", "month"]).reset_index(drop=True)
    return monthly


def validate_monthly_weather(monthly: pd.DataFrame, common_years: list) -> None:
    subsection("Monthly panel validation")
    n_years  = len(common_years)
    n_months = n_years * 12
    dists    = monthly["district"].unique()
    for d in dists:
        sub = monthly[monthly["district"] == d]
        if len(sub) != n_months:
            print(f"    WARNING {d}: {len(sub)} monthly rows, expected {n_months}")

    n_miss = monthly.drop(columns=["district"]).isna().sum().sum()
    print(f"    Districts    : {len(dists)}")
    print(f"    Total rows   : {len(monthly):,}")
    print(f"    Missing vals : {n_miss}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — YEARLY AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_yearly_weather(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates to yearly summaries per district.

    Output columns:
        district, year,
        avg_tmax, avg_tmin, avg_tmean,
        max_tmax, min_tmin,
        annual_rainfall,
        rainy_days, heavy_rain_days, very_heavy_rain_days,
        HDD_10, HDD_15, HDD_18
    """
    # Temperature and rainfall from daily
    daily["_heavy"]      = (daily["rainfall"] >= 64.5).astype(int)
    daily["_very_heavy"] = (daily["rainfall"] >= 115.6).astype(int)
    daily["_rainy"]      = (daily["rainfall"] >= RAINY_DAY_MM).astype(int)

    yr_grp = daily.groupby(["district", "year"])
    yearly = yr_grp.agg(
        avg_tmax           = ("tmax",         "mean"),
        avg_tmin           = ("tmin",         "mean"),
        avg_tmean          = ("tmean",         "mean"),
        max_tmax           = ("tmax",          "max"),
        min_tmin           = ("tmin",          "min"),
        annual_rainfall    = ("rainfall",      "sum"),
        rainy_days         = ("_rainy",        "sum"),
        heavy_rain_days    = ("_heavy",        "sum"),
        very_heavy_rain_days = ("_very_heavy", "sum"),
    ).reset_index()

    daily.drop(columns=["_heavy", "_very_heavy", "_rainy"], inplace=True)

    # HDD yearly totals from monthly (already computed)
    for base in HDD_BASES:
        col = f"HDD_{base}"
        hdd_yr = (monthly.groupby(["district", "year"])[col]
                  .sum().reset_index())
        yearly = yearly.merge(hdd_yr, on=["district", "year"])

    yearly = yearly.sort_values(["district", "year"]).reset_index(drop=True)
    return yearly


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — SPI CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _fit_spi_group(values: pd.Series) -> pd.Series:
    """
    Fits a zero-inflated gamma distribution to a series of rainfall
    totals (all years for one district×calendar-month combination)
    and returns standardised SPI z-scores aligned to the input index.

    Method: McKee et al. (1993).
    """
    values   = values.astype(float)
    nonzero  = values[values > 0]

    if len(nonzero) < 5:
        return pd.Series(np.nan, index=values.index)

    q             = (values == 0).sum() / len(values)
    shape, _, scale = stats.gamma.fit(nonzero, floc=0)

    def _cdf_to_z(v):
        if v <= 0:
            p = q
        else:
            g = stats.gamma.cdf(v, shape, loc=0, scale=scale)
            p = q + (1 - q) * g
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(stats.norm.ppf(p))

    return values.apply(_cdf_to_z)


def compute_spi(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Computes SPI-1 and SPI-3 from monthly_rainfall per the plan.

    Fit is performed per (district, calendar month) so monsoon and
    non-monsoon seasonality are handled correctly.

    Output columns:
        district, year, month, rainfall_monthly, SPI1, SPI3
    """
    df = monthly[["district", "year", "month", "monthly_rainfall"]].copy()
    df = df.rename(columns={"monthly_rainfall": "rainfall_monthly"})
    df = df.sort_values(["district", "year", "month"]).reset_index(drop=True)

    for ts in SPI_TIMESCALES:
        accum_col = f"_rain_accum_{ts}"
        spi_col   = f"SPI{ts}"

        # Rolling accumulation within each district's chronological series
        df[accum_col] = (
            df.groupby("district")["rainfall_monthly"]
            .transform(lambda s: s.rolling(ts, min_periods=ts).sum())
        )

        spi_vals = pd.Series(np.nan, index=df.index)
        for (dist, mon), grp in df.groupby(["district", "month"]):
            spi_vals.loc[grp.index] = _fit_spi_group(grp[accum_col])

        df[spi_col] = spi_vals
        df.drop(columns=[accum_col], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7 — METADATA
# ═══════════════════════════════════════════════════════════════════════════

def build_metadata(daily: pd.DataFrame, monthly: pd.DataFrame,
                   yearly: pd.DataFrame, spi: pd.DataFrame,
                   common_years: list) -> dict:
    """Constructs a metadata dict describing the preprocessing run."""

    def _miss(df: pd.DataFrame) -> dict:
        miss = df.isna().sum()
        return {col: int(v) for col, v in miss.items() if v > 0}

    meta = {
        "generated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
        "state":             STATE_NAME,
        "analysis_period":   {"start": int(min(common_years)),
                              "end":   int(max(common_years))},
        "years_processed":   [int(y) for y in common_years],
        "n_years":           len(common_years),
        "districts":         sorted(daily["district"].unique().tolist()),
        "n_districts":       int(daily["district"].nunique()),
        "hdd_bases":         HDD_BASES,
        "spi_timescales":    SPI_TIMESCALES,
        "rainy_day_threshold_mm": RAINY_DAY_MM,
        "row_counts": {
            "daily":   int(len(daily)),
            "monthly": int(len(monthly)),
            "yearly":  int(len(yearly)),
            "spi":     int(len(spi)),
        },
        "missing_values": {
            "daily":   _miss(daily),
            "monthly": _miss(monthly),
            "yearly":  _miss(yearly),
            "spi":     _miss(spi),
        },
        "outputs": {
            "district_daily_weather.csv":   "Daily district weather panel",
            "district_monthly_weather.csv": "Monthly aggregated weather stats",
            "district_yearly_weather.csv":  "Yearly aggregated weather stats",
            "district_monthly_spi.csv":     "Monthly SPI-1 and SPI-3",
            "weather_metadata.json":        "This file",
        },
    }
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> dict:
    print("\n" + "=" * 70)
    print("WEATHER PREPROCESSING — UP Weather Derivative Project")
    print("=" * 70)

    # ── PHASE 1: Load & Validate ─────────────────────────────────────────
    section("PHASE 1 — LOAD & VALIDATE DATA")

    subsection("Loading shapefile")
    districts = load_up_districts(SHAPEFILE_PATH)

    subsection("Loading temperature & rainfall files")
    tmax_years = load_all_temp_years(MAXTEMP_DIR, "Tmax")
    tmin_years = load_all_temp_years(MINTEMP_DIR, "Tmin")
    rain_years = load_all_rain_years(RAINFALL_DIR)

    common_years = validate_and_intersect(tmax_years, tmin_years, rain_years)

    # ── PHASE 2: Grid-to-District Mapping ────────────────────────────────
    section("PHASE 2 — GRID-TO-DISTRICT MAPPING")

    print("  Building temperature grid map (1° resolution)...")
    temp_map = build_grid_district_map(districts, TEMP_LAT, TEMP_LON,
                                        label="temp 1°")

    print("  Building rainfall grid map (0.25° resolution)...")
    rain_map = build_grid_district_map(districts, RAIN_LAT, RAIN_LON,
                                        label="rain 0.25°")

    # Optionally save maps for audit / reuse
    temp_map.to_csv(os.path.join(OUTPUT_DIR, "grid_map_temp.csv"), index=False)
    rain_map.to_csv(os.path.join(OUTPUT_DIR, "grid_map_rain.csv"), index=False)
    print("  Grid maps saved.")

    # ── PHASE 3: Daily District Weather Panel ────────────────────────────
    section("PHASE 3 — DAILY DISTRICT WEATHER PANEL")

    print("  Aggregating gridded data to district-daily...")
    daily = build_daily_panel(tmax_years, tmin_years, rain_years,
                               temp_map, rain_map, common_years)
    validate_daily_panel(daily)

    daily_path = os.path.join(OUTPUT_DIR, "district_daily_weather.csv")
    daily.to_csv(daily_path, index=False)
    print(f"\n  Saved → {daily_path}  ({len(daily):,} rows)")

    # ── PHASE 4: Monthly Aggregation ─────────────────────────────────────
    section("PHASE 4 — MONTHLY AGGREGATION")

    monthly = compute_monthly_weather(daily)
    validate_monthly_weather(monthly, common_years)

    monthly_path = os.path.join(OUTPUT_DIR, "district_monthly_weather.csv")
    monthly.to_csv(monthly_path, index=False)
    print(f"\n  Saved → {monthly_path}  ({len(monthly):,} rows)")

    # ── PHASE 5: Yearly Aggregation ──────────────────────────────────────
    section("PHASE 5 — YEARLY AGGREGATION")

    yearly = compute_yearly_weather(daily, monthly)

    yearly_path = os.path.join(OUTPUT_DIR, "district_yearly_weather.csv")
    yearly.to_csv(yearly_path, index=False)
    print(f"  Saved → {yearly_path}  ({len(yearly):,} rows)")

    # ── PHASE 6: SPI Construction ─────────────────────────────────────────
    section("PHASE 6 — SPI CONSTRUCTION")

    spi = compute_spi(monthly)
    nan_spi = spi[["SPI1", "SPI3"]].isna().sum()
    print(f"  SPI NaN counts: SPI1={nan_spi['SPI1']}, SPI3={nan_spi['SPI3']}")

    spi_path = os.path.join(OUTPUT_DIR, "district_monthly_spi.csv")
    spi.to_csv(spi_path, index=False)
    print(f"  Saved → {spi_path}  ({len(spi):,} rows)")

    # ── PHASE 7: Metadata ─────────────────────────────────────────────────
    section("PHASE 7 — METADATA")

    meta      = build_metadata(daily, monthly, yearly, spi, common_years)
    meta_path = os.path.join(OUTPUT_DIR, "weather_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {meta_path}")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    section("PREPROCESSING COMPLETE")
    print(f"""
  Outputs written to: {OUTPUT_DIR}
  ├── district_daily_weather.csv    ({len(daily):,} rows)
  ├── district_monthly_weather.csv  ({len(monthly):,} rows)
  ├── district_yearly_weather.csv   ({len(yearly):,} rows)
  ├── district_monthly_spi.csv      ({len(spi):,} rows)
  ├── weather_metadata.json
  ├── grid_map_temp.csv
  └── grid_map_rain.csv

  Districts : {meta['n_districts']}
  Period    : {meta['analysis_period']['start']}–{meta['analysis_period']['end']}
  HDD bases : {HDD_BASES}
  SPI scales: {SPI_TIMESCALES}

  These files feed directly into weather_feature_engineering.py.

  NOTE: District name spellings in this shapefile may differ from the
  mandi price CSVs (e.g. "Prayagraj" vs "Allahabad", "Ayodhya" vs
  "Faizabad"). Maintain a name-mapping dict before merging datasets.
""")

    return {
        "daily":   daily,
        "monthly": monthly,
        "yearly":  yearly,
        "spi":     spi,
        "meta":    meta,
    }


if __name__ == "__main__":
    results = main()