"""
price_preprocessing.py

Implements price_preprocessing_plan.txt end-to-end:

  Phase 1 - Load & consolidate raw yearly per-district CSVs
  Phase 2 - Monthly aggregation
  Phase 3 - Missing-month imputation (continuous monthly series)
  Phase 4 - Validation
  Phase 5 - Final output -> phase0_output/district_monthly_prices.csv

Expected input layout (Agmarknet-style exports). Two layouts are auto-detected:

  <raw-dir>/                      <raw-dir>/
      Agra/                           Agra.csv
          2006.csv                    Aligarh.csv
          2007.csv                    ...
          ...
      Aligarh/
          ...
   (nested: one folder per           (flat: one CSV per district,
    district, yearly files inside)    already concatenated across years)

Each CSV is expected to contain (column names vary by export, detected
adaptively - see COLUMN_CANDIDATES below):
  - a date column
  - modal / min / max price columns
  - optionally a district_name column (cross-checked against the folder/
    file-derived district name as a sanity check, if present)

Note: this version has no arrivals/quantity tracking - that column does
not exist in the source data, so it has been removed entirely (rather
than carried through as an always-NaN field).

Usage:
    python price_preprocessing.py
    python price_preprocessing.py --base-dir /path/to/SURGE --raw-dir analysing_data_debug/wheat_prices_processed --start-year 2006 --end-year 2025
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Adaptive column-name detection
# Agmarknet exports (especially scraped ones) vary a lot in column naming
# -- "Modal Price (Rs./Quintal)", "Modal_x0020_Price", "modal_price", etc.
# We normalize aggressively and match on substrings, most-specific first.
# ----------------------------------------------------------------------
COLUMN_CANDIDATES = {
    "date": ["price date", "reported date", "arrival date", "date"],
    "modal": ["modal price", "modal_price", "modal"],
    "min": ["min price", "minimum price", "min_price", "min"],
    "max": ["max price", "maximum price", "max_price", "max"],
    "district_name": ["district name", "district_name"],
}

REQUIRED_FIELDS = ["date", "modal", "min", "max"]

ISO_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")


def parse_date_column(series):
    """Robust date parsing.

    ISO-style dates (YYYY-MM-DD, optionally with a time/timezone suffix,
    e.g. '2006-01-04 00:00:00+00:00') are unambiguous and must NOT be
    parsed with dayfirst=True - doing so badly mis-parses them (e.g.
    '2006-01-04' silently becomes April 1st instead of January 4th).

    Other formats (e.g. 'DD/MM/YYYY' from raw Agmarknet HTML exports) ARE
    genuinely ambiguous and default to dayfirst=True, the standard Indian
    convention.

    We detect which case we're in by checking whether the strings start
    with a 4-digit year.
    """
    s = series.astype(str).str.strip()
    iso_like = s.str.match(ISO_DATE_RE)
    frac_iso = iso_like.mean() if len(iso_like) else 0.0

    if frac_iso > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def normalize(col):
    s = str(col)
    s = s.replace("_x0020_", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def find_column(columns, candidates):
    norm_map = {}
    for c in columns:
        norm_map.setdefault(normalize(c), c)
    # exact match first (most specific candidate wins because of list order)
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # substring fallback
    for norm_c, orig_c in norm_map.items():
        for cand in candidates:
            if cand in norm_c:
                return orig_c
    return None


# ----------------------------------------------------------------------
# Phase 1 - Load & consolidate
# ----------------------------------------------------------------------
def discover_districts(raw_root):
    """Return [(district_name, [csv_file_paths]), ...].

    Auto-detects two layouts:
      nested: raw_root/<District>/<year>.csv   (one folder per district)
      flat:   raw_root/<District>.csv          (one file per district)
    Both can coexist; whichever pattern is found contributes districts.
    """
    entries = sorted(os.listdir(raw_root))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(raw_root, e))]
    flat_csvs = [e for e in entries if e.lower().endswith(".csv")
                 and os.path.isfile(os.path.join(raw_root, e))]

    districts = []
    for d in subdirs:
        files = sorted(glob.glob(os.path.join(raw_root, d, "*.csv")))
        if files:
            districts.append((d, files))
    for f in flat_csvs:
        name = os.path.splitext(f)[0]
        districts.append((name, [os.path.join(raw_root, f)]))

    return districts


def load_district_raw(district_name, csv_files, verbose=False):
    """Read every CSV for one district, standardize columns, cross-check
    any internal district_name column, drop exact duplicate records,
    sort chronologically."""
    frames = []
    mismatched_names = set()

    for f in csv_files:
        try:
            raw = pd.read_csv(f)
        except Exception as e:
            print(f"      ! could not read {os.path.basename(f)}: {e}")
            continue
        if raw.empty:
            continue

        col_map = {field: find_column(raw.columns, cands) for field, cands in COLUMN_CANDIDATES.items()}
        missing_required = [f_ for f_ in REQUIRED_FIELDS if col_map[f_] is None]
        if missing_required:
            print(f"      ! skipping {os.path.basename(f)} - missing required column(s) "
                  f"{missing_required} (columns found: {list(raw.columns)})")
            continue

        # Sanity check: if the file carries its own district_name column,
        # make sure it agrees with the folder/file-derived district name.
        if col_map["district_name"] is not None:
            seen = set(raw[col_map["district_name"]].dropna().astype(str).str.strip().unique())
            unexpected = {s for s in seen if s.lower() != district_name.lower()}
            mismatched_names |= unexpected

        out = pd.DataFrame()
        out["date"] = parse_date_column(raw[col_map["date"]])
        out["modal_price"] = pd.to_numeric(raw[col_map["modal"]], errors="coerce")
        out["min_price"] = pd.to_numeric(raw[col_map["min"]], errors="coerce")
        out["max_price"] = pd.to_numeric(raw[col_map["max"]], errors="coerce")

        out = out.dropna(subset=["date"])
        if not out.empty:
            frames.append(out)

    if mismatched_names:
        print(f"      \u26a0 internal district_name column disagrees with folder name "
              f"'{district_name}' - found also: {sorted(mismatched_names)}")

    if not frames:
        return pd.DataFrame(columns=["date", "modal_price", "min_price", "max_price",
                                      "year", "month"])

    df = pd.concat(frames, ignore_index=True)

    # Phase 1 step 6: remove duplicate records (exact duplicate rows only -
    # different markets reporting on the same date are legitimate, separate
    # observations and must NOT be collapsed here).
    df = df.drop_duplicates()

    # Phase 1 step 7: sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Phase 1 step 5: extract year, month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


# ----------------------------------------------------------------------
# Phase 2 - Monthly aggregation
# ----------------------------------------------------------------------
def aggregate_monthly(daily_df, district_name):
    cols = ["district", "year", "month", "mean_modal_price", "mean_min_price",
            "mean_max_price", "mean_price_spread", "reporting_days"]
    if daily_df.empty:
        return pd.DataFrame(columns=cols)

    d = daily_df.copy()
    d["spread"] = d["max_price"] - d["min_price"]

    grouped = d.groupby(["year", "month"]).agg(
        mean_modal_price=("modal_price", "mean"),
        mean_min_price=("min_price", "mean"),
        mean_max_price=("max_price", "mean"),
        mean_price_spread=("spread", "mean"),
        reporting_days=("date", "nunique"),
    ).reset_index()

    grouped.insert(0, "district", district_name)
    return grouped[cols]


# ----------------------------------------------------------------------
# Phase 3 - Missing month imputation
# ----------------------------------------------------------------------
def impute_missing_months(monthly_df, district_name, start_year, end_year):
    """Reindex onto a continuous (year, month) grid for [start_year, end_year]
    and fill gaps per the plan's five cases."""

    full_index = pd.MultiIndex.from_product(
        [range(start_year, end_year + 1), range(1, 13)], names=["year", "month"]
    )
    full = pd.DataFrame(index=full_index).reset_index()

    merged = full.merge(monthly_df.drop(columns=["district"], errors="ignore"),
                         on=["year", "month"], how="left")
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    # Lock in is_imputed BEFORE any filling: a month is "imputed" iff there
    # was no observed record for it at all.
    merged["is_imputed"] = merged["mean_modal_price"].isna().astype(int)
    merged["reporting_days"] = merged["reporting_days"].fillna(0)

    value_cols = ["mean_modal_price", "mean_min_price", "mean_max_price"]

    # --- Case 5: entire missing year -> interpolate month-wise using the
    #     same calendar month in the neighbouring years; if only one
    #     neighbour exists, use that neighbour's value. ---
    year_all_missing = merged.groupby("year")["mean_modal_price"].apply(lambda s: s.isna().all())
    fully_missing_years = set(year_all_missing[year_all_missing].index)

    for col in value_cols:
        for yr in sorted(fully_missing_years):
            prev_vals = merged.loc[merged["year"] == yr - 1, ["month", col]].set_index("month")[col]
            next_vals = merged.loc[merged["year"] == yr + 1, ["month", col]].set_index("month")[col]
            for m in range(1, 13):
                idx = merged.index[(merged["year"] == yr) & (merged["month"] == m)]
                if len(idx) == 0:
                    continue
                pv = prev_vals.get(m, np.nan)
                nv = next_vals.get(m, np.nan)
                if pd.notna(pv) and pd.notna(nv):
                    merged.loc[idx, col] = (pv + nv) / 2
                elif pd.notna(pv):
                    merged.loc[idx, col] = pv
                elif pd.notna(nv):
                    merged.loc[idx, col] = nv
                # else: leave NaN, resolved by the fallback pass below

    # --- Cases 1-4: gaps inside an otherwise-present series ---
    #   Case 2 (missing first month(s)):  copy first available following value
    #   Case 3 (missing last month(s)):   copy previous available value
    #   Case 1 (single interior gap) &
    #   Case 4 (multiple consecutive interior gaps): linear interpolation
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    for col in value_cols:
        s = merged[col]

        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        if first_valid is not None and first_valid > 0:
            s.iloc[:first_valid] = s.iloc[first_valid]
        if last_valid is not None and last_valid < len(s) - 1:
            s.iloc[last_valid + 1:] = s.iloc[last_valid]

        s = s.interpolate(method="linear", limit_direction="both")
        merged[col] = s

    # Recompute spread from the now-imputed min/max so it stays internally
    # consistent (spread is derived, not independently imputed).
    merged["mean_price_spread"] = merged["mean_max_price"] - merged["mean_min_price"]

    merged["reporting_days"] = merged["reporting_days"].fillna(0).astype(int)
    merged.loc[merged["reporting_days"] == 0, "is_imputed"] = 1
    merged.insert(0, "district", district_name)

    cols = ["district", "year", "month", "mean_modal_price", "mean_min_price",
            "mean_max_price", "mean_price_spread", "reporting_days", "is_imputed"]
    return merged[cols]


# ----------------------------------------------------------------------
# Phase 4 - Validation
# ----------------------------------------------------------------------
def validate_panel(df):
    print("\n[Phase 4] Validation checks")
    all_passed = True

    def ok(msg):
        print(f"  \u2713 {msg}")

    def bad(msg):
        nonlocal all_passed
        print(f"  \u2717 {msg}")
        all_passed = False

    month_counts = df.groupby(["district", "year"])["month"].nunique()
    bad_counts = month_counts[month_counts != 12]
    if len(bad_counts) > 0:
        bad(f"{len(bad_counts)} (district, year) groups do not have exactly 12 months.")
    else:
        ok("Every (district, year) has exactly 12 months.")

    is_sorted = all(
        g["year"].mul(12).add(g["month"]).is_monotonic_increasing
        for _, g in df.groupby("district")
    )
    ok("Chronological ordering verified within each district.") if is_sorted else bad("Chronological order violated for at least one district.")

    dup = int(df.duplicated(subset=["district", "year", "month"]).sum())
    ok("No duplicate (district, year, month) rows.") if dup == 0 else bad(f"{dup} duplicate (district, year, month) rows.")

    price_cols = ["mean_modal_price", "mean_min_price", "mean_max_price"]
    miss_price = int(df[price_cols].isna().sum().sum())
    ok("No missing price values.") if miss_price == 0 else bad(f"{miss_price} missing price values.")

    neg_price = int((df[price_cols] < 0).sum().sum())
    ok("No negative prices.") if neg_price == 0 else bad(f"{neg_price} negative price values.")

    neg_days = int((df["reporting_days"] < 0).sum())
    ok("reporting_days >= 0 for every row.") if neg_days == 0 else bad(f"{neg_days} rows with reporting_days < 0.")

    inconsistent = df[(df["reporting_days"] == 0) & (df["is_imputed"] != 1)]
    if len(inconsistent) == 0:
        ok("is_imputed flag consistent with reporting_days == 0 rows.")
    else:
        bad(f"{len(inconsistent)} rows with reporting_days == 0 but is_imputed != 1.")

    return all_passed


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Price preprocessing pipeline (price_preprocessing_plan.txt)")
    parser.add_argument("--base-dir", default="/Users/sthitpragye/Desktop/Finance/SURGE")
    parser.add_argument("--raw-dir", default="analysing_data_debug/wheat_prices_processed",
                         help="Folder of per-district CSVs (flat or nested), relative to base-dir")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    raw_root = os.path.join(args.base_dir, args.raw_dir)
    out_dir = os.path.join(args.base_dir, "phase0_output")
    out_file = os.path.join(out_dir, "district_monthly_prices.csv")

    if not os.path.isdir(raw_root):
        print(f"ERROR: raw data folder not found at {raw_root}")
        print("Pass --base-dir / --raw-dir to point at the correct location.")
        sys.exit(1)

    districts = discover_districts(raw_root)
    if not districts:
        print(f"ERROR: no CSV files found under {raw_root} "
              f"(checked for both <District>/<year>.csv and <District>.csv layouts).")
        sys.exit(1)

    print("=" * 70)
    print("PRICE PREPROCESSING PIPELINE  (price_preprocessing_plan.txt)")
    print("=" * 70)
    print(f"Raw data root : {raw_root}")
    print(f"Districts     : {len(districts)}")
    print(f"Analysis years: {args.start_year}-{args.end_year}")

    all_monthly = []
    no_data_districts = []

    for i, (district, files) in enumerate(districts, 1):
        print(f"\n[{i}/{len(districts)}] {district}")

        daily = load_district_raw(district, files, verbose=args.verbose)
        if daily.empty:
            print("    ! no usable records found - skipping district.")
            no_data_districts.append(district)
            continue
        print(f"    loaded {len(daily)} daily records spanning "
              f"{daily['date'].min().date()} to {daily['date'].max().date()}")

        monthly = aggregate_monthly(daily, district)
        imputed = impute_missing_months(monthly, district, args.start_year, args.end_year)

        n_imputed = int(imputed["is_imputed"].sum())
        print(f"    monthly panel: {len(imputed)} rows ({n_imputed} imputed months)")

        all_monthly.append(imputed)

    if not all_monthly:
        print("\nERROR: no district produced usable data. Nothing to write.")
        sys.exit(1)

    final_df = pd.concat(all_monthly, ignore_index=True)
    final_df = final_df.sort_values(["district", "year", "month"]).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Districts processed : {final_df['district'].nunique()}")
    print(f"Total rows          : {len(final_df)}")
    if no_data_districts:
        print(f"Districts with NO usable raw data ({len(no_data_districts)}): {no_data_districts}")

    passed = validate_panel(final_df)

    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(out_file, index=False)
    print(f"\nWritten to: {out_file}")

    print("\n" + "=" * 70)
    if passed:
        print("RESULT: \u2705 Preprocessing complete - all Phase 4 checks passed.")
    else:
        print("RESULT: \u26a0\ufe0f Preprocessing complete, but some Phase 4 checks failed (see above).")
    print("=" * 70)

    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()