"""
FILL MISSING PRICE YEARS — UP Mandi Price Data Imputation
========================================================================
For each district folder under a crop's price-data directory
(e.g. UP_WHEAT_2006_2025/<district>/<year>.csv), detects years with no
data (file missing, or file present but empty / header-only) and fills
them by carrying forward (or backward) the nearest year that does have
real data, re-dating every row to the missing year and scaling
min_price / max_price / modal_price by cumulative CPI inflation between
the source year and the target year.

WHY THIS APPROACH:
  - Mandi price *levels* drift with general inflation even when local
    supply/demand patterns repeat similarly year to year. Carrying the
    raw rupee figures forward unadjusted would understate prices in
    later filled years; carrying them adjusted keeps the filled data
    roughly consistent with the surrounding real (non-imputed) years.
  - Exact daily mandi reporting dates differ year to year (markets
    don't report on identical calendar dates), so there is no clean
    daily one-to-one mapping across years. The best tractable
    approximation — per your instruction — is to reuse the *previous*
    (or nearest available) year's calendar of reporting dates, simply
    relabeling the year.

FILL DIRECTION:
  - Forward fill is preferred: a missing year Y is built from year Y-1
    (which may itself be a previously-filled year — chains work
    automatically since years are processed in increasing order).
  - Backward fill is used only for *leading* gaps with no earlier real
    year to anchor to (e.g. district has no 2006-2008 data at all but
    starts in 2009) — those are built from the nearest later year,
    deflated backward.
  - A district with NO real data anywhere in 2006-2025 cannot be filled
    at all (nothing to anchor to) and is skipped with a warning.

INFLATION SERIES:
  Annual India WPI (Wholesale Price Index, all commodities; Office of
  Economic Adviser), representing the inflation rate from year Y-1 to
  year Y, mapped from fiscal-year to calendar-year. WPI is more
  volatile than CPI and went negative in 2015 and 2023 — swap in a
  wheat/cereal-specific WPI sub-index in INFLATION_RATES below if you
  want an even more targeted adjustment.

OUTPUT:
  - One new <year>.csv written into each district folder for every
    year that was filled. Existing real-data files are NEVER
    overwritten.
  - A single audit log `_imputation_log.csv` written at the top of each
    crop directory, recording exactly what was filled, from where, and
    what inflation factor was applied — so Phase 2 can identify/exclude
    imputed years if needed.

USAGE:
    python fill_missing_price_years.py
"""

import os
import re
import glob
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = "/Users/sthitpragye/Desktop/Finance/SURGE"

# One entry per crop's price-data root. Add more paths here as you get
# rice / pulses / sugarcane mandi data in the same folder layout.
CROP_DIRS = [
    # os.path.join(BASE_DIR, "UP_WHEAT_2006_2025"),
    os.path.join(BASE_DIR, "analysing_data_debug", "wheat_prices_processed")
]

YEAR_START, YEAR_END = 2006, 2025

PRICE_COLS = ["min_price", "max_price", "modal_price"]
PRICE_DECIMALS = 2          # rounding applied to inflation-adjusted prices

# India WPI annual inflation rate, year Y vs year Y-1 (Office of Economic
# Adviser, fiscal-year figures mapped to the calendar year containing
# 9 of the fiscal year's 12 months, e.g. FY2008-09 -> 2008). WPI is more
# volatile than CPI and went negative in 2015 and 2023.
INFLATION_RATES = {
    2006: 0.0470, 2007: 0.0440, 2008: 0.0810, 2009: 0.0380, 2010: 0.0960,
    2011: 0.0890, 2012: 0.0740, 2013: 0.0600, 2014: 0.0200, 2015: -0.0370,
    2016: 0.0170, 2017: 0.0300, 2018: 0.0430, 2019: 0.0170, 2020: 0.0130,
    2021: 0.1300, 2022: 0.0940, 2023: -0.0070, 2024: 0.0200, 2025: 0.0250,
}
YEAR_FILE_RE = re.compile(r"^(\d{4})\.csv$")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def inflation_factor(source_year: int, target_year: int) -> float:
    """
    Cumulative multiplicative inflation factor to apply to a price from
    source_year so it is expressed in target_year terms.
    target_year > source_year -> compounds forward (scales up).
    target_year < source_year -> compounds backward (scales down).
    """
    factor = 1.0
    if target_year > source_year:
        for y in range(source_year + 1, target_year + 1):
            factor *= (1 + INFLATION_RATES[y])
    elif target_year < source_year:
        for y in range(target_year + 1, source_year + 1):
            factor /= (1 + INFLATION_RATES[y])
    return factor


def shift_row_date(ts: pd.Timestamp, target_year: int):
    """
    Re-dates a timestamp to target_year, keeping month/day/time/tz.
    Returns pd.NaT for the one case where this is impossible: a source
    Feb 29 being re-dated into a target_year that isn't a leap year.
    """
    try:
        return ts.replace(year=target_year)
    except ValueError:
        return pd.NaT


def discover_years(district_dir: str) -> dict:
    """
    Returns {year: filepath} for every YYYY.csv in district_dir that
    actually contains at least one data row (i.e. not missing and not
    empty/header-only).
    """
    found = {}
    for fp in glob.glob(os.path.join(district_dir, "*.csv")):
        m = YEAR_FILE_RE.match(os.path.basename(fp))
        if not m:
            continue
        year = int(m.group(1))
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if len(df) > 0:
            found[year] = fp
    return found


def build_filled_year(source_df: pd.DataFrame, source_year: int,
                       target_year: int):
    """
    Builds the filled DataFrame for target_year from source_df
    (source_year's actual/previously-filled data).
    Returns (new_df, n_dropped_feb29, inflation_factor_used).
    """
    df = source_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].apply(lambda ts: shift_row_date(ts, target_year))

    n_dropped = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).copy()

    factor = inflation_factor(source_year, target_year)
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = (df[col] * factor).round(PRICE_DECIMALS)

    df["year"] = target_year
    df["is_imputed"] = True
    df["imputed_from_year"] = source_year

    return df, n_dropped, factor


# ═══════════════════════════════════════════════════════════════════════
# Main per-district fill logic
# ═══════════════════════════════════════════════════════════════════════

def fill_district(district_dir: str, district_name: str) -> list:
    """
    Fills missing years for one district folder. Returns a list of
    log-row dicts (one per year filled) for the audit log.
    """
    available = discover_years(district_dir)   # {year: filepath}
    all_years = set(range(YEAR_START, YEAR_END + 1))
    missing = sorted(all_years - set(available))

    if not missing:
        print(f"  {district_name}: complete ({YEAR_START}-{YEAR_END} all present)")
        return []

    if not available:
        print(f"  {district_name}: SKIPPED — no real data in any year, "
              f"nothing to anchor an imputation to.")
        return []

    # working[year] = DataFrame (real or filled), keyed by year
    working = {y: pd.read_csv(fp) for y, fp in available.items()}
    log_rows = []

    # ── Pass 1: forward fill (prev year -> this year); chains naturally
    #    since the loop runs in increasing year order.
    for y in range(YEAR_START + 1, YEAR_END + 1):
        if y in working:
            continue
        if (y - 1) in working:
            new_df, n_dropped, factor = build_filled_year(working[y - 1], y - 1, y)
            working[y] = new_df
            log_rows.append(dict(
                district=district_name, year_filled=y, source_year=y - 1,
                direction="forward", inflation_factor=round(factor, 4),
                rows_written=len(new_df), rows_dropped_feb29=n_dropped))

    # ── Pass 2: backward fill for leading gaps with no earlier anchor
    for y in range(YEAR_END - 1, YEAR_START - 1, -1):
        if y in working:
            continue
        if (y + 1) in working:
            new_df, n_dropped, factor = build_filled_year(working[y + 1], y + 1, y)
            working[y] = new_df
            log_rows.append(dict(
                district=district_name, year_filled=y, source_year=y + 1,
                direction="backward", inflation_factor=round(factor, 4),
                rows_written=len(new_df), rows_dropped_feb29=n_dropped))

    # ============================================================
    # Write every filled year; never overwrite real-data files
    # ============================================================

    for row in log_rows:

        y = row["year_filled"]
        out_path = os.path.join(district_dir, f"{y}.csv")

        should_write = True

        if os.path.exists(out_path):
            try:
                existing_df = pd.read_csv(out_path)

                if len(existing_df) > 0:
                    # Real data exists -> never overwrite
                    print(
                        f"    WARNING: {out_path} contains real data — skipping."
                    )
                    should_write = False

                else:
                    # Header-only file -> replace it
                    os.remove(out_path)

            except Exception:
                # Corrupt/unreadable -> replace it
                os.remove(out_path)

        if should_write:
            working[y].to_csv(out_path, index=False)

    filled_years = sorted(r["year_filled"] for r in log_rows)
    still_missing = sorted(all_years - set(working))
    suffix = f" — still missing {still_missing} (no anchor available)" if still_missing else ""
    print(f"  {district_name}: filled {len(filled_years)} year(s) {filled_years}{suffix}")

    return log_rows


# ═══════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════

def main():
    for crop_dir in CROP_DIRS:
        if not os.path.isdir(crop_dir):
            print(f"SKIPPING — directory not found: {crop_dir}")
            continue

        crop_label = os.path.basename(crop_dir)
        print("=" * 70)
        print(f"FILLING MISSING PRICE YEARS — {crop_label}")
        print("=" * 70)

        district_dirs = sorted(
            d for d in glob.glob(os.path.join(crop_dir, "*"))
            if os.path.isdir(d)
        )
        if not district_dirs:
            print(f"  No district subfolders found in {crop_dir}")
            continue

        all_log_rows = []
        for d in district_dirs:
            district_name = os.path.basename(d)
            all_log_rows.extend(fill_district(d, district_name))

        log_path = os.path.join(crop_dir, "_imputation_log.csv")
        if all_log_rows:
            pd.DataFrame(all_log_rows).to_csv(log_path, index=False)
            print(f"\n  Imputation log ({len(all_log_rows)} rows) -> {log_path}")
        else:
            print(f"\n  Nothing needed filling in {crop_label}.")
        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()