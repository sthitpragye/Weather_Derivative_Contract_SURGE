from pathlib import Path
import pandas as pd

# ==========================================
# CONFIG
# ==========================================

INPUT_ROOT = Path("analysing_data_debug/d5_duplicates")
OUTPUT_FILE = Path("analysing_data_debug/P3_zero_prices.csv")

PRICE_COLUMNS = ["min_price", "max_price", "modal_price"]

# ==========================================
# MAIN
# ==========================================

records = []

csv_files = sorted(INPUT_ROOT.rglob("*.csv"))

print(f"Scanning {len(csv_files)} CSV files...\n")

for csv_path in csv_files:

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        continue

    existing_cols = [c for c in PRICE_COLUMNS if c in df.columns]

    if not existing_cols:
        continue

    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    zero_mask = False

    for col in existing_cols:
        zero_mask |= (df[col] == 0)

    n_zero_rows = int(zero_mask.sum())

    if n_zero_rows > 0:

        records.append({
            "district": csv_path.parent.name,
            "file": csv_path.name,
            "relative_path": str(csv_path.relative_to(INPUT_ROOT)),
            "zero_price_rows": n_zero_rows,
            "total_rows": len(df)
        })

report = pd.DataFrame(records)
print(f"Number of records found: {len(records)}")
report = report.sort_values(
    ["district_name", "market_name", "file"],
    ignore_index=True
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print(f"Files with zero prices : {len(report)}")
print(f"Report saved to        : {OUTPUT_FILE}")
print("=" * 60)