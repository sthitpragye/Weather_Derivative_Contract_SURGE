
"""
d5_duplicate_dates.py (Resolution Version)

Creates cleaned copies of all CSVs after resolving duplicate
(date, market_id) records.

Resolution rules
----------------
1. Exact Duplicate:
   Keep first row.

2. Missing Value Duplicate:
   Keep row with maximum non-null values.

3. Price Conflict:
   min_price   = minimum of conflicting min_price values
   max_price   = maximum of conflicting max_price values
   modal_price = round((min_price + max_price)/2)

Original files are NEVER modified.

Output:
analysing_data_debug/
    d5_duplicates/
        <district>/<year>.csv

    debug_output/
        d5_duplicate_resolution_log.csv
"""

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np
import chardet

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/sthitpragye/Desktop/Finance/SURGE"
SOURCE_DIR = os.path.join(BASE_DIR, "UP_WHEAT_2006_2025")
OUT_DIR = os.path.join(BASE_DIR, "analysing_data_debug", "d5_duplicates")
LOG_DIR = os.path.join(BASE_DIR, "analysing_data_debug", "debug_output")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

YEAR_RE = re.compile(r"^\d{4}\.csv$", re.I)

CANONICAL = {
    "date":["date"],
    "market_id":["market_id"],
    "min_price":["min_price"],
    "max_price":["max_price"],
    "modal_price":["modal_price"],
}

def detect_encoding(fp):
    with open(fp,"rb") as f:
        raw=f.read(50000)
    return chardet.detect(raw).get("encoding") or "utf-8"

def read_csv(fp):
    for enc in [detect_encoding(fp),"utf-8","latin-1","cp1252"]:
        try:
            return pd.read_csv(fp,encoding=enc,low_memory=False)
        except:
            pass
    return None

def norm(s):
    return re.sub(r"[\s_]+","_",s.strip().lower())

def map_cols(cols):
    d={}
    low={norm(c):c for c in cols}
    for k,v in CANONICAL.items():
        d[k]=None
        for x in v:
            if norm(x) in low:
                d[k]=low[norm(x)]
                break
    return d

logs=[]

for district_path in sorted(glob.glob(os.path.join(SOURCE_DIR,"*"))):
    if not os.path.isdir(district_path):
        continue

    district=os.path.basename(district_path)
    out_district=os.path.join(OUT_DIR,district)
    os.makedirs(out_district,exist_ok=True)

    for fp in sorted(glob.glob(os.path.join(district_path,"*.csv"))):

        fname=os.path.basename(fp)
        if not YEAR_RE.match(fname):
            continue

        df=read_csv(fp)
        if df is None:
            continue

        m=map_cols(df.columns)

        if m["min_price"] is not None and m["max_price"] is not None:
            min_price = pd.to_numeric(df[m["min_price"]], errors="coerce")
            max_price = pd.to_numeric(df[m["max_price"]], errors="coerce")

            df = df.loc[
                min_price.notna() &
                max_price.notna() &
                (min_price != 0) &
                (max_price != 0)
            ].copy()

        if None in [m["date"],m["market_id"]]:
            df.to_csv(os.path.join(out_district,fname),index=False)
            continue

        date_col=m["date"]
        market_col=m["market_id"]

        df[date_col]=pd.to_datetime(df[date_col],errors="coerce")

        keep_rows=[]
        processed=set()

        grouped=df.groupby([date_col,market_col],dropna=False)

        for key,g in grouped:

            idxs=tuple(g.index)

            if len(g)==1:
                keep_rows.append(g.iloc[0])
                continue

            if idxs in processed:
                continue
            processed.add(idxs)

            before=len(g)

            if g.drop_duplicates().shape[0]==1:
                keep_rows.append(g.iloc[0])
                logs.append({
                    "district":district,
                    "file":fname,
                    "date":key[0],
                    "market_id":key[1],
                    "type":"Exact Duplicate",
                    "rows_before":before,
                    "rows_after":1
                })
                continue

            nonnull=g.notna().sum(axis=1)

            if nonnull.max()!=nonnull.min():
                row=g.loc[nonnull.idxmax()]
                keep_rows.append(row)
                logs.append({
                    "district":district,
                    "file":fname,
                    "date":key[0],
                    "market_id":key[1],
                    "type":"Missing Value Duplicate",
                    "rows_before":before,
                    "rows_after":1
                })
                continue

            row=g.iloc[0].copy()

            if all(m[x] is not None for x in ["min_price", "max_price", "modal_price"]):

                mins = pd.to_numeric(g[m["min_price"]], errors="coerce")
                maxs = pd.to_numeric(g[m["max_price"]], errors="coerce")


                # Drop rows where either min_price or max_price is missing
                valid = (mins.notna() & maxs.notna() &
                    (mins != 0) & (maxs != 0)
                )

                g = g.loc[valid].copy()
                mins = mins.loc[valid]
                maxs = maxs.loc[valid]

                # If no valid rows remain, drop this duplicate group
                if len(g) == 0:
                    logs.append({
                        "district": district,
                        "file": fname,
                        "date": key[0],
                        "market_id": key[1],
                        "type": "Dropped (Missing min/max price)",
                        "rows_before": before,
                        "rows_after": 0
                    })
                    continue

                new_min = float(mins.min())
                new_max = float(maxs.max())
                new_modal = round((new_min + new_max) / 2)

                row = g.iloc[0].copy()
                row[m["min_price"]] = new_min
                row[m["max_price"]] = new_max
                row[m["modal_price"]] = new_modal

            keep_rows.append(row)

            logs.append({
                "district":district,
                "file":fname,
                "date":key[0],
                "market_id":key[1],
                "type":"Price Conflict",
                "rows_before":before,
                "rows_after":1
            })

        cleaned = pd.DataFrame(keep_rows, columns=df.columns)

        if date_col in cleaned.columns:
            cleaned=cleaned.sort_values([date_col,market_col]).reset_index(drop=True)

        cleaned.to_csv(os.path.join(out_district,fname),index=False)

pd.DataFrame(logs).to_csv(
    os.path.join(LOG_DIR,"d5_duplicate_resolution_log.csv"),
    index=False
)

print("Finished.")
print(f"Cleaned files : {OUT_DIR}")
print(f"Log file      : {os.path.join(LOG_DIR,'d5_duplicate_resolution_log.csv')}")
