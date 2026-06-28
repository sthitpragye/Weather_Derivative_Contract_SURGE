"""
RAW DATA AUDIT — UP Mandi Price CSVs
========================================================================
Scans every district/year CSV under UP_WHEAT_2006_2025/ without
modifying any file.  Produces three outputs:

  audit_output/file_audit.csv        — one row per CSV, all flags
  audit_output/column_variants.csv   — every column name seen across
                                       all files (helps spot naming drift)
  audit_output/audit_summary.txt     — human-readable summary

Run this BEFORE price_preprocessing.py.  Fix or document whatever the
audit surfaces, then re-run the audit to confirm a clean bill of health.

Checks performed
----------------
FILE LEVEL
  F1  File unreadable (encoding / parse error)
  F2  File empty (zero bytes)
  F3  Header-only (no data rows after header)
  F4  Year in filename doesn't match year(s) found in the date column

COLUMN LEVEL
  C1  One or more expected columns missing
  C2  Column names differ from the standard set (possible rename/typo)
  C3  Unexpected extra columns (informational, not an error)

DATE LEVEL
  D1  Date column not found or not parseable
  D2  One or more date values invalid / NaT after parsing
  D3  Dates outside the expected analysis window (2006–2025)
  D4  More than one date format detected within the file
  D5  Duplicate date entries (same date appears more than once)

PRICE LEVEL
  P1  Non-numeric values in price columns
  P2  Negative price values
  P3  Zero price values
  P4  Price outliers detected (value > Q3 + 3×IQR or < Q1 - 3×IQR,
      per column, computed file-by-file)
  P5  min_price > max_price on any row (price inversion)
  P6  modal_price outside [min_price, max_price] on any row

ARRIVALS LEVEL
  A1  Non-numeric values in arrivals column
  A2  Negative arrivals
  A3  100% of rows have zero arrivals (market never reported activity)
  A4  Arrivals column missing entirely

CONSISTENCY
  K1  Suspicious price jumps: any single-day abs change > 3× rolling
      median absolute deviation (flags likely data-entry errors)
"""

import os
import io
import glob
import re
import warnings
import chardet
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR    = "/Users/sthitpragye/Desktop/Finance/SURGE"
# PRICE_DIR   = os.path.join(BASE_DIR, "UP_WHEAT_2006_2025")
# AUDIT_DIR   = os.path.join(BASE_DIR, "audit_output")
PRICE_DIR   = os.path.join(BASE_DIR, "analysing_data_debug","wheat_prices_processed")
AUDIT_DIR   = os.path.join(BASE_DIR, "analysing_data_debug", "audit_output_debug")

YEAR_START, YEAR_END = 2006, 2025

ALLOWED_COLUMNS = {
    "date",
    "commodity_id",
    "census_state_id",
    "census_district_id",
    "market_id",
    "min_price",
    "max_price",
    "modal_price",
    "district_name",
    "market_name",
    "year",
    "is_imputed",
    "imputed_from_year",
}

# Canonical column names the preprocessing module expects.
# The audit will try to map whatever it finds to these.
CANONICAL = {
    "date"        : ["date", "Date", "DATE", "report_date", "price_date",
                     "Arrival Date", "arrival_date", "ArrivalDate"],
    "min_price"   : ["min_price", "Min Price", "min price", "MinPrice",
                     "Minimum Price", "minimum_price", "min_modal_price"],
    "max_price"   : ["max_price", "Max Price", "max price", "MaxPrice",
                     "Maximum Price", "maximum_price", "max_modal_price"],
    "modal_price" : ["modal_price", "Modal Price", "modal price", "ModalPrice",
                     "Modal_Price", "modal", "Modal"],
    # "arrivals"    : ["arrivals", "Arrivals", "ARRIVALS", "arrival",
    #                  "Arrival", "total_arrivals", "Total Arrivals"],
}

OPTIONAL_COLUMNS = {
    "year",
    "is_imputed",
    "imputed_from_year",
}

YEAR_FILE_RE = re.compile(r"^(\d{4})\.csv$", re.IGNORECASE)

# Outlier detection threshold: flag if value > Q3 + N*IQR or < Q1 - N*IQR
OUTLIER_IQR_MULTIPLIER = 3.0

# Price jump threshold (× rolling MAD)
JUMP_MAD_MULTIPLIER = 3.0

# List of flags considered "critical" (for summary reporting)
CRITICAL_FLAGS = [
    "F1_unreadable",
    "F2_empty_file",
    "F3_header_only",
    "F4_year_mismatch",
    "C1_missing_cols",
    "D1_date_unparseable",
    "D2_invalid_dates",
    "D3_dates_out_of_range",
    "D5_duplicate_dates",
    "P1_non_numeric_prices",
    "P2_negative_prices",
    "P5_price_inversion",
    "P6_modal_out_of_range",
    # "A1_non_numeric_arr",
    # "A2_negative_arrivals",
]

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def detect_encoding(filepath: str) -> str:
    with open(filepath, "rb") as f:
        raw = f.read(50_000)  # sample first 50 KB
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def try_read_csv(filepath: str) -> tuple[pd.DataFrame | None, str]:
    """
    Attempts to read a CSV trying common encodings.
    Returns (DataFrame or None, encoding_used_or_error_message).
    """
    encodings = [detect_encoding(filepath), "utf-8", "latin-1", "cp1252"]
    seen = set()
    for enc in encodings:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            return df, enc
        except Exception:
            continue
    return None, "unreadable"


def normalise_col_name(raw: str) -> str:
    """Lowercase, strip, collapse spaces/underscores."""
    return re.sub(r"[\s_]+", "_", raw.strip().lower())


def find_canonical(cols: list[str]) -> dict[str, str | None]:
    """
    Returns {canonical_name: actual_col_in_file} for each expected
    column.  None if not found.
    """
    col_lower = {normalise_col_name(c): c for c in cols}
    mapping = {}
    for canon, variants in CANONICAL.items():
        found = None
        for v in variants:
            key = normalise_col_name(v)
            if key in col_lower:
                found = col_lower[key]
                break
        mapping[canon] = found
    return mapping


def to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce to numeric, handling comma-formatted numbers like "1,500".
    """
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def detect_date_formats(s: pd.Series, sample: int = 200) -> list[str]:
    """
    Sniffs the date format(s) present in a string series.
    Returns list of unique strftime patterns found.
    """
    candidates = [
        "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%m-%y",
        "%m/%d/%Y",
    ]
    detected = set()
    sample_vals = s.dropna().head(sample).astype(str).tolist()
    for val in sample_vals:
        for fmt in candidates:
            try:
                datetime.strptime(val.strip(), fmt)
                detected.add(fmt)
                break
            except ValueError:
                continue
    return list(detected)


def outlier_count(s: pd.Series) -> int:
    s = s.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
    return int(((s < lo) | (s > hi)).sum())


def suspicious_jumps(s: pd.Series) -> int:
    s = s.dropna().reset_index(drop=True)
    if len(s) < 5:
        return 0
    diffs = s.diff().abs()
    rolling_mad = diffs.rolling(7, min_periods=3).median()
    threshold = JUMP_MAD_MULTIPLIER * rolling_mad
    flags = diffs > threshold
    return int(flags.sum())


# ═══════════════════════════════════════════════════════════════════════
# AUDIT SINGLE FILE
# ═══════════════════════════════════════════════════════════════════════

def audit_file(filepath: str, district: str, filename: str) -> dict:
    """
    Audits one CSV file. Returns a dict of flags and metadata.
    """
    m = YEAR_FILE_RE.match(filename)
    filename_year = int(m.group(1)) if m else None

    rec = {
        "district"       : district,
        "filename"        : filename,
        "filename_year"   : filename_year,
        "file_size_bytes" : os.path.getsize(filepath),
        "encoding_used"   : None,
        "n_rows"          : None,
        "n_cols"          : None,
        "columns_found"   : None,
        # Flags
        "F1_unreadable"         : False,
        "F2_empty_file"         : False,
        "F3_header_only"        : False,
        "F4_year_mismatch"      : False,
        "C1_missing_cols"       : False,
        "C2_nonstandard_names"  : False,
        "C3_extra_cols"         : False,
        "D1_date_unparseable"   : False,
        "D2_invalid_dates"      : False,
        "D3_dates_out_of_range" : False,
        "D4_mixed_date_formats" : False,
        "D5_duplicate_dates"    : False,
        "P1_non_numeric_prices" : False,
        "P2_negative_prices"    : False,
        "P3_zero_prices"        : False,
        "P4_price_outliers"     : False,
        "P5_price_inversion"    : False,
        "P6_modal_out_of_range" : False,
        # "A1_non_numeric_arr"    : False,
        # "A2_negative_arrivals"  : False,
        # "A3_all_zero_arrivals"  : False,
        # "A4_arrivals_missing"   : False,
        "K1_suspicious_jumps"   : False,
        # Detail fields
        "missing_cols"          : "",
        "nonstandard_cols"      : "",
        "extra_cols"            : "",
        "n_invalid_dates"       : 0,
        "n_out_of_range_dates"  : 0,
        "date_formats_detected" : "",
        "n_duplicate_dates"     : 0,
        "n_price_outliers"      : 0,
        "n_price_inversions"    : 0,
        "n_modal_violations"    : 0,
        "n_suspicious_jumps"    : 0,
        "years_in_data"         : "",
        "total_issues"          : 0,
        "critical_issues"       : 0,
        "warning_issues"        : 0,
        "total_issues"          : 0,
    }

    # ── F2: empty file
    if rec["file_size_bytes"] == 0:
        rec["F2_empty_file"] = True
        rec["total_issues"]  = 1
        return rec

    # ── Read file
    df, enc_or_err = try_read_csv(filepath)
    if df is None:
        rec["F1_unreadable"] = True
        rec["encoding_used"] = enc_or_err
        rec["total_issues"]  = 1
        return rec

    rec["encoding_used"] = enc_or_err
    rec["n_rows"]         = len(df)
    rec["n_cols"]         = len(df.columns)
    rec["columns_found"]  = "|".join(df.columns.tolist())
    rec["is_imputed_file"] = (
        "is_imputed" in df.columns and
        df["is_imputed"].fillna(False).any()
    )

    # ── F3: header only
    if len(df) == 0:
        rec["F3_header_only"] = True
        rec["total_issues"]   = 1
        return rec

    # ── Column mapping
    col_map = find_canonical(df.columns.tolist())

    missing_canon = [k for k, v in col_map.items() if v is None]
    if missing_canon:
        rec["C1_missing_cols"]  = True
        rec["missing_cols"]     = "|".join(missing_canon)

    nonstandard = [
        f"{v}→{k}" for k, v in col_map.items()
        if v is not None and v != k
    ]
    if nonstandard:
        rec["C2_nonstandard_names"] = True
        rec["nonstandard_cols"]     = "|".join(nonstandard)

    extra = [
        c for c in df.columns
        if normalise_col_name(c) not in {
            normalise_col_name(col)
            for col in ALLOWED_COLUMNS
        }
    ]

    if extra:
        rec["C3_extra_cols"] = True
        rec["extra_cols"] = "|".join(extra)

    # ── DATE checks
    date_col = col_map.get("date")
    if date_col is None:
        rec["D1_date_unparseable"] = True
    else:
        raw_dates = df[date_col].astype(str)

        # Detect formats
        fmts = detect_date_formats(df[date_col])
        rec["date_formats_detected"] = "|".join(fmts)
        if len(fmts) > 1:
            rec["D4_mixed_date_formats"] = True

        parsed = pd.to_datetime(df[date_col],
                                errors="coerce", utc=True)
        n_nat = int(parsed.isna().sum())
        if n_nat == len(df):
            rec["D1_date_unparseable"] = True
        elif n_nat > 0:
            rec["D2_invalid_dates"] = True
            rec["n_invalid_dates"]  = n_nat

        valid_parsed = parsed.dropna()
        if len(valid_parsed):
            years_in_data = sorted(valid_parsed.dt.year.unique().tolist())
            rec["years_in_data"] = "|".join(map(str, years_in_data))

            # F4: year mismatch between filename and data
            if filename_year and filename_year not in years_in_data:
                rec["F4_year_mismatch"] = True

            # D3: dates outside analysis window
            n_oor = int(((valid_parsed.dt.year < YEAR_START) |
                         (valid_parsed.dt.year > YEAR_END)).sum())
            if n_oor:
                rec["D3_dates_out_of_range"] = True
                rec["n_out_of_range_dates"]  = n_oor

            # D5: duplicate dates
            dup_mask = df.duplicated(
                subset=[date_col, "market_id"],
                keep=False
            )

            n_dup = int(dup_mask.sum())

            if n_dup:
                rec["D5_duplicate_dates"] = True
                rec["n_duplicate_dates"] = n_dup

    # ── PRICE checks
    for price_key in ["min_price", "max_price", "modal_price"]:
        col = col_map.get(price_key)
        if col is None:
            continue

        raw = df[col]
        num = to_numeric_series(raw)

        # P1: non-numeric
        n_non_num = int(num.isna().sum() - raw.isna().sum())
        if n_non_num > 0:
            rec["P1_non_numeric_prices"] = True

        # P2: negative
        if (num < 0).any():
            rec["P2_negative_prices"] = True

        # P3: zeros
        if (num == 0).any():
            rec["P3_zero_prices"] = True

        # P4: outliers
        n_out = outlier_count(num)
        rec["n_price_outliers"] += n_out
        if n_out > 0:
            rec["P4_price_outliers"] = True

    # P5: min > max (price inversion)
    min_col   = col_map.get("min_price")
    max_col   = col_map.get("max_price")
    modal_col = col_map.get("modal_price")

    if min_col and max_col:
        mn = to_numeric_series(df[min_col])
        mx = to_numeric_series(df[max_col])
        n_inv = int((mn > mx).sum())
        if n_inv:
            rec["P5_price_inversion"] = True
            rec["n_price_inversions"] = n_inv

        # P6: modal outside [min, max]
        if modal_col:
            mo = to_numeric_series(df[modal_col])
            n_viol = int(((mo < mn) | (mo > mx)).sum())
            if n_viol:
                rec["P6_modal_out_of_range"] = True
                rec["n_modal_violations"]    = n_viol

    # # ── ARRIVALS checks
    # arr_col = col_map.get("arrivals")
    # if arr_col is None:
    #     rec["A4_arrivals_missing"] = True
    # else:
    #     raw_arr = df[arr_col]
    #     num_arr = to_numeric_series(raw_arr)

    #     n_non_num_arr = int(num_arr.isna().sum() - raw_arr.isna().sum())
    #     if n_non_num_arr > 0:
    #         rec["A1_non_numeric_arr"] = True

    #     if (num_arr < 0).any():
    #         rec["A2_negative_arrivals"] = True

    #     if num_arr.notna().any() and (num_arr.fillna(0) == 0).all():
    #         rec["A3_all_zero_arrivals"] = True

    # ── K1: suspicious modal price jumps (within each market)

    if modal_col and "market_id" in df.columns:

        total_jumps = 0

        for _, market_df in df.groupby("market_id"):

            market_df = market_df.sort_values(date_col)

            mo = to_numeric_series(market_df[modal_col])

            total_jumps += suspicious_jumps(mo)

        rec["n_suspicious_jumps"] = total_jumps

        if total_jumps > 0:
            rec["K1_suspicious_jumps"] = True

    # ── Issue counts

    rec["critical_issues"] = sum(
        int(rec[f]) for f in CRITICAL_FLAGS
    )

    rec["warning_issues"] = sum([
        int(rec["C2_nonstandard_names"]),
        int(rec["C3_extra_cols"]),
        int(rec["D4_mixed_date_formats"]),
        int(rec["P3_zero_prices"]),
        int(rec["P4_price_outliers"]),
        # int(rec["A3_all_zero_arrivals"]),
        # int(rec["A4_arrivals_missing"]),
        int(rec["K1_suspicious_jumps"]),
    ])

    rec["total_issues"] = (
        rec["critical_issues"] +
        rec["warning_issues"]
    )

    return rec


# ═══════════════════════════════════════════════════════════════════════
# COLUMN VARIANTS TRACKER
# ═══════════════════════════════════════════════════════════════════════

def collect_column_variants(audit_rows: list[dict]) -> pd.DataFrame:
    """
    Aggregates every unique column name seen across all files and shows
    which canonical name it maps to (or 'UNKNOWN').
    """
    from collections import Counter

    col_counter = Counter()
    for row in audit_rows:
        if row["columns_found"]:
            for col in row["columns_found"].split("|"):
                col_counter[col.strip()] += 1

    records = []
    for col_name, count in sorted(col_counter.items(),
                                   key=lambda x: -x[1]):
        # Find canonical mapping
        canon = "UNKNOWN"
        for cname, variants in CANONICAL.items():
            if normalise_col_name(col_name) in [normalise_col_name(v)
                                                  for v in variants]:
                canon = cname
                break
        records.append({
            "column_as_found"  : col_name,
            "maps_to_canonical": canon,
            "file_count"       : count,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════

def write_summary(audit_df: pd.DataFrame, out_path: str):
    flag_cols = [
        c for c in audit_df.columns
        if (
            audit_df[c].dtype == bool and
            re.match(r"^[FCDPAK]\d+_", c)
        )
    ]
    
    critical = (audit_df["critical_issues"] > 0).sum()

    warnings = (
        (audit_df["critical_issues"] == 0) &
        (audit_df["warning_issues"] > 0)
    ).sum()

    clean = (
        (audit_df["critical_issues"] == 0) &
        (audit_df["warning_issues"] == 0)
    ).sum()

    lines = [
        "=" * 70,
        "RAW DATA AUDIT SUMMARY",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Price dir : {PRICE_DIR}",
        "=" * 70,
        "",
        
        f"Total files scanned   : {len(audit_df)}",
        f"Total districts       : {audit_df['district'].nunique()}",
        f"Files with ANY issue  : {(audit_df['total_issues'] > 0).sum()}",
        f"Fully clean files     : {(audit_df['total_issues'] == 0).sum()}",
        f"Files with critical issues : {critical}",
        f"Files with warnings only   : {warnings}",
        f"Fully clean files          : {clean}",
        f"Imputed files         : {audit_df['is_imputed_file'].sum()}",
        "",
        "─" * 70,
        "FLAG BREAKDOWN",
        "─" * 70,
    ]

    flag_descriptions = {
        "F1_unreadable"        : "File unreadable (encoding/parse error)",
        "F2_empty_file"        : "File completely empty (0 bytes)",
        "F3_header_only"       : "File has header but no data rows",
        "F4_year_mismatch"     : "Year in filename ≠ year(s) in date column",
        "C1_missing_cols"      : "One or more expected columns absent",
        "C2_nonstandard_names" : "Column names differ from standard set",
        "C3_extra_cols"        : "Unexpected extra columns present (info only)",
        "D1_date_unparseable"  : "Date column absent or entirely unparseable",
        "D2_invalid_dates"     : "Some date values are NaT after parsing",
        "D3_dates_out_of_range": f"Dates outside {YEAR_START}–{YEAR_END}",
        "D4_mixed_date_formats": "Multiple date formats within one file",
        "D5_duplicate_dates"   : "Duplicate date entries",
        "P1_non_numeric_prices": "Non-numeric values in price columns",
        "P2_negative_prices"   : "Negative price values",
        "P3_zero_prices"       : "Zero price values",
        "P4_price_outliers"    : f"Price outliers (> Q3+{OUTLIER_IQR_MULTIPLIER}×IQR or < Q1-{OUTLIER_IQR_MULTIPLIER}×IQR)",
        "P5_price_inversion"   : "min_price > max_price on some rows",
        "P6_modal_out_of_range": "modal_price outside [min, max] on some rows",
        # "A1_non_numeric_arr"   : "Non-numeric values in arrivals column",
        # "A2_negative_arrivals" : "Negative arrivals",
        # "A3_all_zero_arrivals" : "All arrival values are zero",
        # "A4_arrivals_missing"  : "Arrivals column entirely absent",
        "K1_suspicious_jumps"  : f"Suspicious price jumps (>{JUMP_MAD_MULTIPLIER}× rolling MAD)",
    }

    for fc in flag_cols:
        n = audit_df[fc].sum()
        if n > 0:
            desc = flag_descriptions.get(fc, fc)
            lines.append(f"  {fc:30s} {n:4d} files   {desc}")

    lines += [
        "",
        "─" * 70,
        "TOP 20 MOST PROBLEMATIC FILES",
        "─" * 70,
    ]
    top20 = (
    audit_df[
            (audit_df["critical_issues"] > 0) |
            (audit_df["warning_issues"] > 0)
        ]
        .sort_values(
            ["critical_issues", "warning_issues"],
            ascending=False
        )
             .head(20)[["district", "filename", "total_issues"]])
    for _, row in top20.iterrows():
        lines.append(f"  {row['district']:30s}  {row['filename']:12s}  "
                     f"{row['total_issues']} issue(s)")

    lines += [
        "",
        "─" * 70,
        "DISTRICTS WITH MOST ISSUES",
        "─" * 70,
    ]
    dist_issues = (audit_df.groupby("district")["total_issues"]
                   .sum().sort_values(ascending=False).head(15))
    for dist, n in dist_issues.items():
        lines.append(f"  {dist:35s} {n:4d} total issues")

    lines += ["", "=" * 70]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RAW DATA AUDIT — UP Mandi Price CSVs")
    print("=" * 70)

    os.makedirs(AUDIT_DIR, exist_ok=True)

    # Collect all district directories
    district_dirs = sorted(
        d for d in glob.glob(os.path.join(PRICE_DIR, "*"))
        if os.path.isdir(d)
    )
    if not district_dirs:
        raise FileNotFoundError(f"No district folders found in {PRICE_DIR}")

    print(f"Districts found: {len(district_dirs)}")

    audit_rows = []
    total_files = 0

    for d in district_dirs:
        district_name = os.path.basename(d)
        csv_files = sorted(glob.glob(os.path.join(d, "*.csv")))
        for fp in csv_files:
            fname = os.path.basename(fp)
            if not YEAR_FILE_RE.match(fname):
                continue  # skip non-year files like _imputation_log.csv
            total_files += 1
            rec = audit_file(fp, district_name, fname)
            audit_rows.append(rec)

        print(f"  {district_name}: {len(csv_files)} file(s) scanned")

    print(f"\nTotal files audited: {total_files}")

    # ── Outputs
    audit_df = pd.DataFrame(audit_rows)

    file_audit_path = os.path.join(AUDIT_DIR, "file_audit.csv")
    audit_df.to_csv(file_audit_path, index=False)
    print(f"\nFile audit saved  → {file_audit_path}")

    col_variants = collect_column_variants(audit_rows)
    col_path = os.path.join(AUDIT_DIR, "column_variants.csv")
    col_variants.to_csv(col_path, index=False)
    print(f"Column variants   → {col_path}")

    summary_path = os.path.join(AUDIT_DIR, "audit_summary.txt")
    write_summary(audit_df, summary_path)
    print(f"Summary report    → {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()