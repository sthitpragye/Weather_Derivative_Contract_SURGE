
"""
d5_duplicate_dates.py
---------------------

Investigate D5 duplicate (date, market_id) groups across all yearly CSVs.

Output:
    analyse_data_debug/debug_output/d5_duplicate_dates.csv

This script DOES NOT modify any data.
"""

import os
import re
import glob
import warnings
from collections import Counter

import chardet
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/sthitpragye/Desktop/Finance/SURGE"
# PRICE_DIR = os.path.join(BASE_DIR, "UP_WHEAT_2006_2025")
PRICE_DIR = os.path.join(BASE_DIR, "analysing_data_debug", "d5_duplicates")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysing_data_debug", "debug_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUTPUT_DIR, "d5_duplicate_dates.csv")

YEAR_FILE_RE = re.compile(r"^(\d{4})\.csv$", re.I)

CANONICAL = {
    "date": ["date", "Date", "DATE", "report_date", "Arrival Date"],
    "market_id": ["market_id", "Market ID"],
    "market_name": ["market_name", "Market Name"],
    "min_price": ["min_price", "Min Price"],
    "max_price": ["max_price", "Max Price"],
    "modal_price": ["modal_price", "Modal Price"],
    "is_imputed": ["is_imputed"],
}


def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read(50000)
    return chardet.detect(raw).get("encoding") or "utf-8"


def read_csv(path):
    encs = [detect_encoding(path), "utf-8", "latin-1", "cp1252"]
    tried = set()
    for e in encs:
        if e in tried:
            continue
        tried.add(e)
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except Exception:
            pass
    return None


def norm(x):
    return re.sub(r"[\s_]+", "_", x.strip().lower())


def mapping(cols):
    lower = {norm(c): c for c in cols}
    out = {}
    for c, vars_ in CANONICAL.items():
        out[c] = None
        for v in vars_:
            if norm(v) in lower:
                out[c] = lower[norm(v)]
                break
    return out


def classify(group, cols):
    if group.drop_duplicates().shape[0] == 1:
        return "Exact Duplicate"

    pcols = [c for c in cols if c in group.columns]

    if group[pcols].isna().any().any():
        complete = group[pcols].notna().sum(axis=1)
        if complete.max() != complete.min():
            return "Missing Value Duplicate"

    if group[pcols].nunique(dropna=False).max() > 1:
        return "Price Conflict"

    return "Mixed"


records = []

files_scanned = 0
files_with_dup = 0
dup_groups = 0
dup_rows = 0
types = Counter()
largest = 0

district_dirs = sorted(
    d for d in glob.glob(os.path.join(PRICE_DIR, "*"))
    if os.path.isdir(d)
)

for district_dir in district_dirs:

    district = os.path.basename(district_dir)

    for fp in sorted(glob.glob(os.path.join(district_dir, "*.csv"))):

        fname = os.path.basename(fp)

        if not YEAR_FILE_RE.match(fname):
            continue

        files_scanned += 1

        df = read_csv(fp)
        if df is None:
            continue

        mp = mapping(df.columns)

        # Drop rows where min_price or max_price is 0
        if mp["min_price"] is not None and mp["max_price"] is not None:
            min_price = pd.to_numeric(df[mp["min_price"]], errors="coerce")
            max_price = pd.to_numeric(df[mp["max_price"]], errors="coerce")

            df = df.loc[(min_price != 0) & (max_price != 0)].copy()

        if mp["date"] is None or mp["market_id"] is None:
            continue

        date_col = mp["date"]
        market_col = mp["market_id"]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        mask = df.duplicated([date_col, market_col], keep=False)

        if not mask.any():
            continue

        files_with_dup += 1

        dup_df = df.loc[mask].copy()

        for (d, m), grp in dup_df.groupby([date_col, market_col]):

            dup_groups += 1
            dup_rows += len(grp)
            largest = max(largest, len(grp))

            dtype = classify(
                grp,
                [mp["min_price"], mp["max_price"], mp["modal_price"]],
            )
            types[dtype] += 1

            identical = grp.drop_duplicates().shape[0] == 1

            for idx, row in grp.iterrows():

                records.append({
                    "district": district,
                    "filename": fname,
                    "market_id": m,
                    "market_name": row.get(mp["market_name"], np.nan),
                    "date": d,
                    "duplicate_group_size": len(grp),
                    "duplicate_type": dtype,
                    "identical_rows": identical,
                    "row_index": idx,
                    "min_price": row.get(mp["min_price"], np.nan),
                    "max_price": row.get(mp["max_price"], np.nan),
                    "modal_price": row.get(mp["modal_price"], np.nan),
                    "is_imputed": row.get(mp["is_imputed"], np.nan),
                })

out = pd.DataFrame(records)
out.to_csv(OUT_FILE, index=False)

print("=" * 60)
print("D5 DUPLICATE DATE ANALYSIS")
print("=" * 60)
print(f"Files scanned               : {files_scanned}")
print(f"Files with duplicates       : {files_with_dup}")
print(f"Duplicate groups            : {dup_groups}")
print(f"Duplicate rows              : {dup_rows}")
print()
for k, v in sorted(types.items()):
    print(f"{k:25s}: {v}")
print(f"\nLargest duplicate group     : {largest}")
print(f"\nOutput saved to\n{OUT_FILE}")
