"""
APY PREPROCESSING MODULE — Phase 0
====================================
Loads district-wise Area / Production / Yield data for a single crop
(default: Wheat, Rabi season) from the renamed yearly CSVs
(2006.csv … 2025.csv), reconciles district names against the GADM
canonical spelling used by the weather module, and outputs a clean
district × year panel ready for feature engineering.

DESIGN DECISIONS (Phase 0 vs Phase 2):
  - Wheat only (CROP config param — change to "Rice" etc. to extend).
  - Rabi season only for wheat.  If a folder has both Rabi and Total rows
    the script keeps Rabi; if no Rabi rows survive the filter it falls
    back to Total and logs a warning so you know.
  - Year label = filename year (2006.csv → year 2006).  This avoids the
    "2005-06" vs 2006 ambiguity and aligns with the year axis used by
    the price and weather modules.
  - State filter: keeps only Uttar Pradesh rows.  Raw data may contain
    other states if the source dump was not pre-filtered.
  - No detrending here — that is a Phase 2 concern.  Phase 0 needs raw
    levels for the feature engineering step.

CROSSWALK SOURCE (APY_TO_GADM):
  Copied verbatim from phase2_step1.py, which was built by diffing the
  actual district lists.  Do not edit without re-verifying against both
  the APY raw files and district_weather_features.csv.

INPUT DIRECTORY STRUCTURE:
  <BASE_DIR>/
    Wheat_APY/
      Wheat_Area_District_wise_1998-2025/
        2006.csv … 2025.csv
      Wheat_Production_District_wise_1998-2025/
        2006.csv … 2025.csv
      Wheat_Yield_District_wise_1998-2025/
        2006.csv … 2025.csv

  Each CSV schema:
    cropname, statename, districtname, cropyear, seasonname,
    <metric_col>, majorcrops, Source

OUTPUT:
  phase0_output/district_apy_panel.csv
    One row per (district, year).
    Columns: district, year, area, production, yield,
             season_used, years_with_all_three

Run:
  python apy_preprocessing.py
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  — edit these paths to match your environment
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = "/Users/sthitpragye/Desktop/Finance/SURGE"
OUT_DIR  = os.path.join(BASE_DIR, "phase0", "output")
os.makedirs(OUT_DIR, exist_ok=True)

CROP          = "Wheat"       # change to Rice / Sugarcane / Pulses to extend
TARGET_SEASON = "Rabi"        # primary season filter for wheat
FALLBACK_SEASON = "Total"     # used if no Rabi rows survive for a folder
STATE_NAME    = "Uttar Pradesh"

YEAR_START = 2006
YEAR_END   = 2025

# Folder paths relative to BASE_DIR
CROP_FOLDERS = {
    "Wheat": {
        "area":       "Wheat_APY/Wheat_Area_District_wise_1998-2025",
        "production": "Wheat_APY/Wheat_Production_District_wise_1998-2025",
        "yield":      "Wheat_APY/Wheat_Yield_District_wise_1998-2025",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# DISTRICT NAME CROSSWALK
# Copied verbatim from phase2_step1.py — built by diffing actual lists.
# APY spelling → GADM (weather module) canonical spelling.
# ═══════════════════════════════════════════════════════════════════════

APY_TO_GADM = {
    "Ayodhya":        "Faizabad",
    "Bara Banki":     "Barabanki",
    "Bhadohi":        "Sant Ravi Das Nagar",
    "Kheri":          "Lakhimpur Kheri",
    "Mahrajganj":     "Maharajganj",
    "Prayagraj":      "Allahabad",
    "Shrawasti":      "Shravasti",
    "Siddharthnagar": "Siddharth Nagar",
}

# Districts legitimately absent from early years — created after 1998 via
# administrative splits.  Not missing data, real history.
DISTRICTS_CREATED_AFTER_1998 = ["Amethi", "Hapur", "Kasganj", "Sambhal", "Shamli"]

# ═══════════════════════════════════════════════════════════════════════
# COLUMN DETECTION
# Same adaptive pattern as phase2_step1.py — never hardcode positions.
# ═══════════════════════════════════════════════════════════════════════

COLUMN_PATTERNS = {
    "district":   ["districtname", "district", "dist_name"],
    "state":      ["statename", "state_name", "state"],
    "season":     ["seasonname", "season"],
    "crop":       ["cropname", "crop"],
    "area":       ["area"],
    "production": ["production"],
    "yield":      ["yield"],
}

def detect_column(columns, patterns):
    """Return first column whose lowercased name contains any pattern."""
    lower = {c: c.lower() for c in columns}
    for pattern in patterns:
        for orig, low in lower.items():
            if pattern in low:
                return orig
    return None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & CONSOLIDATE
# ═══════════════════════════════════════════════════════════════════════

def load_metric_folder(folder_path: str, metric_name: str) -> pd.DataFrame:
    """
    Load all yearly CSVs from one metric folder (area / production / yield).

    Year label strategy: extract from filename (2006.csv → 2006).
    This is unambiguous for the renamed file set and aligns with the
    year axis used by the price and weather modules.

    Season filter: keep TARGET_SEASON rows only.  Fall back to
    FALLBACK_SEASON if none survive after the state filter.
    """
    pattern = os.path.join(folder_path, "*.csv")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: _year_from_filename(f) or 0
    )

    if not files:
        print(f"    WARNING: no CSV files found in {folder_path}")
        return pd.DataFrame()

    target_files = [
        f for f in files
        if YEAR_START <= (_year_from_filename(f) or 0) <= YEAR_END
    ]
    if not target_files:
        print(f"    WARNING: no files in {YEAR_START}–{YEAR_END} range under {folder_path}")
        return pd.DataFrame()

    frames = []
    cols_logged = False

    for fp in target_files:
        year = _year_from_filename(fp)
        if year is None:
            print(f"    SKIPPED {os.path.basename(fp)}: could not parse year from filename")
            continue

        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"    SKIPPED {os.path.basename(fp)}: read error ({exc})")
            continue

        # Adaptive column detection
        district_col = detect_column(df.columns, COLUMN_PATTERNS["district"])
        state_col    = detect_column(df.columns, COLUMN_PATTERNS["state"])
        season_col   = detect_column(df.columns, COLUMN_PATTERNS["season"])
        value_col    = detect_column(df.columns, COLUMN_PATTERNS[metric_name])

        if not cols_logged:
            print(f"    [{os.path.basename(fp)}] detected → "
                  f"district='{district_col}', state='{state_col}', "
                  f"season='{season_col}', value='{value_col}'")
            cols_logged = True

        if district_col is None or value_col is None:
            print(f"    SKIPPED {os.path.basename(fp)}: missing district or "
                  f"{metric_name} column. Present: {list(df.columns)}")
            continue

        # State filter
        if state_col is not None:
            df = df[df[state_col].str.strip().str.lower() == STATE_NAME.lower()].copy()
            if df.empty:
                # Non-fatal: file may legitimately contain only non-UP districts.
                continue

        # Season filter
        season_used = TARGET_SEASON
        if season_col is not None:
            rabi_rows = df[df[season_col].str.strip().str.lower() == TARGET_SEASON.lower()]
            if not rabi_rows.empty:
                df = rabi_rows.copy()
            else:
                # Fall back to Total
                total_rows = df[df[season_col].str.strip().str.lower() == FALLBACK_SEASON.lower()]
                if not total_rows.empty:
                    print(f"    NOTE {os.path.basename(fp)}: no {TARGET_SEASON} rows — "
                          f"using {FALLBACK_SEASON} instead")
                    df = total_rows.copy()
                    season_used = FALLBACK_SEASON
                # else: keep all rows (unusual; column exists but season values differ)

        out = pd.DataFrame()
        out["district"]    = df[district_col].astype(str).str.strip()
        out["year"]        = year                                   # from filename
        out["season_used"] = season_used
        out[metric_name]   = pd.to_numeric(df[value_col], errors="coerce")

        frames.append(out)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["district", metric_name])
    result = result[result["district"].str.strip() != ""]
    return result


def _year_from_filename(fname: str):
    """Extract 4-digit year from filename, e.g. '2006.csv' → 2006."""
    m = re.search(r"(19|20)\d{2}", os.path.basename(fname))
    return int(m.group(0)) if m else None


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — MERGE AREA / PRODUCTION / YIELD
# ═══════════════════════════════════════════════════════════════════════

def build_apy_panel(crop_label: str) -> pd.DataFrame:
    """
    Load the three metric folders, reconcile to a wide (district × year)
    panel with columns area, production, yield.
    """
    folders = CROP_FOLDERS[crop_label]
    metrics = {}

    for metric_name, rel_path in folders.items():
        full_path = os.path.join(BASE_DIR, rel_path)
        print(f"\n  Loading {metric_name} from: {full_path}")
        df = load_metric_folder(full_path, metric_name)
        if df.empty:
            print(f"    -> 0 rows loaded for {metric_name}")
        else:
            print(f"    -> {len(df):,} rows, "
                  f"{df['district'].nunique()} districts, "
                  f"years {df['year'].min()}–{df['year'].max()}")
        metrics[metric_name] = df

    # Each metric df has: district, year, season_used, <metric>
    # Merge on (district, year); season_used should be consistent —
    # flag if it diverges across the three folders.

    base = None
    for metric_name, df in metrics.items():
        if df.empty:
            continue
        keep = df[["district", "year", "season_used", metric_name]].copy()
        if base is None:
            base = keep
        else:
            base = base.merge(keep, on=["district", "year"], how="outer",
                              suffixes=("", f"_{metric_name}"))

    if base is None or base.empty:
        print("  ERROR: no data loaded across any metric folder.")
        return pd.DataFrame()

    # Resolve duplicate season_used columns from outer-merge suffixes
    season_cols = [c for c in base.columns if c.startswith("season_used")]
    if season_cols:
        # Take the first non-null season label per row
        base["season_used"] = base[season_cols].bfill(axis=1).iloc[:, 0]
        base.drop(columns=[c for c in season_cols if c != "season_used"],
                  errors="ignore", inplace=True)

    return base


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — DISTRICT NAME RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════

def reconcile_district_names(df: pd.DataFrame, weather_districts: set) -> pd.DataFrame:
    """
    Apply APY_TO_GADM crosswalk.  Warn about district names that are not
    in the weather module's district set and are not on the known
    post-1998-created list.
    """
    df = df.copy()
    df["district"] = df["district"].str.strip()
    df["district"] = df["district"].replace(APY_TO_GADM)

    if weather_districts:
        unmatched = sorted(set(df["district"].unique()) - weather_districts)
        unexpected = [d for d in unmatched
                      if d not in DISTRICTS_CREATED_AFTER_1998]
        if unexpected:
            print(f"\n  WARNING: districts not matching weather module's GADM set "
                  f"and not in known post-1998 list — verify spelling:\n"
                  f"  {unexpected}")
        known_new = [d for d in unmatched if d in DISTRICTS_CREATED_AFTER_1998]
        if known_new:
            print(f"  NOTE: post-1998 split districts present (expected): {known_new}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — WITHIN-YEAR AGGREGATION & VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def aggregate_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    After name reconciliation, there should be exactly one row per
    (district, year).  If somehow duplicates exist (e.g. two season
    rows survived the filter) aggregate by mean and warn.
    """
    dup_mask = df.duplicated(subset=["district", "year"], keep=False)
    if dup_mask.any():
        n_dup = dup_mask.sum()
        print(f"\n  WARNING: {n_dup} duplicate (district, year) pairs found "
              f"— aggregating by mean.  Inspect season_used column for the "
              f"cause.")
        num_cols = ["area", "production", "yield"]
        num_cols = [c for c in num_cols if c in df.columns]
        df = (df.groupby(["district", "year"], as_index=False)
                .agg({**{c: "mean" for c in num_cols},
                      "season_used": "first"}))

    # Expected shape: 75 districts × 20 years = 1,500 rows (approximate)
    n_districts = df["district"].nunique()
    n_years     = df["year"].nunique()
    print(f"\n  Panel dimensions: {len(df):,} rows, "
          f"{n_districts} districts, {n_years} years "
          f"({df['year'].min()}–{df['year'].max()})")

    # Completeness report
    total_possible = n_districts * n_years
    pct_filled = len(df) / total_possible * 100 if total_possible > 0 else 0
    print(f"  Coverage: {pct_filled:.1f}% of possible district-year cells filled")

    # Negative value check
    for col in ["area", "production", "yield"]:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                print(f"  WARNING: {n_neg} negative values in '{col}' — "
                      f"setting to NaN")
                df.loc[df[col] < 0, col] = np.nan

    # Missing value summary
    for col in ["area", "production", "yield"]:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                print(f"  Missing '{col}': {n_miss} rows "
                      f"({n_miss / len(df) * 100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def save_panel(df: pd.DataFrame, crop_label: str):
    out_path = os.path.join(OUT_DIR, "district_apy_panel.csv")

    col_order = ["district", "year", "area", "production", "yield", "season_used"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order].sort_values(["district", "year"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(sep)
    print(f"APY PREPROCESSING — Phase 0  |  Crop: {CROP}  |  "
          f"Season: {TARGET_SEASON}  |  Years: {YEAR_START}–{YEAR_END}")
    print(sep)

    # Load weather district list for cross-check (optional — proceeds if absent)
    weather_path = os.path.join(BASE_DIR, "phase0_output", "district_weather_features.csv")
    if os.path.exists(weather_path):
        weather_districts = set(
            pd.read_csv(weather_path)["district"].str.strip().unique()
        )
        print(f"  Weather district reference loaded: {len(weather_districts)} districts")
    else:
        weather_districts = set()
        print(f"  NOTE: {weather_path} not found — skipping weather district "
              f"cross-check (will still proceed).")

    # PHASE 1 + 2: Load and merge metrics
    print(f"\n{'─' * 70}")
    print("PHASE 1–2: Loading and merging Area / Production / Yield")
    print(f"{'─' * 70}")
    panel = build_apy_panel(CROP)

    if panel.empty:
        print("\nERROR: No data loaded. Check BASE_DIR and folder paths in CONFIG.")
        return

    # PHASE 3: Reconcile district names
    print(f"\n{'─' * 70}")
    print("PHASE 3: Reconciling district names (APY → GADM)")
    print(f"{'─' * 70}")
    panel = reconcile_district_names(panel, weather_districts)

    # PHASE 4: Validate
    print(f"\n{'─' * 70}")
    print("PHASE 4: Aggregation and validation")
    print(f"{'─' * 70}")
    panel = aggregate_and_validate(panel)

    # PHASE 5: Save
    print(f"\n{'─' * 70}")
    print("PHASE 5: Saving output")
    print(f"{'─' * 70}")
    out_path = save_panel(panel, CROP)

    print(f"""
{sep}
APY PREPROCESSING COMPLETE.

Output: {out_path}
  - One row per (district, year)
  - Columns: district, year, area, production, yield, season_used
  - Crop: {CROP}  |  Season: {TARGET_SEASON}  |  Years: {YEAR_START}–{YEAR_END}

CHECK BEFORE RUNNING FEATURE ENGINEERING:
  1. Confirm detected column names look correct (printed above for each folder).
  2. If any unexpected district warnings appeared, verify APY_TO_GADM crosswalk.
  3. Check coverage % — significantly below 100% means some district-year
     combinations are absent from the raw data (expected for early years in
     post-1998-created districts; unexpected otherwise).
  4. Look at the missing-value counts for area/production/yield.
     The feature engineering module handles NaN rows by excluding them from
     individual feature calculations but does not impute.
{sep}
""")
    return panel


if __name__ == "__main__":
    main()