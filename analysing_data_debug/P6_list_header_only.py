"""
P6_list_header_only.py

Lists all CSV files that contain only a header row (no data).

Output:
    analysing_data_debug/P6_header_only_files.csv
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = PROJECT_ROOT / "analysing_data_debug" / "wheat_prices_processed"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "analysing_data_debug"
    / "P6_header_only_files.csv"
)

# ============================================================
# SCAN
# ============================================================

csv_files = sorted(INPUT_ROOT.rglob("*.csv"))

print(f"Scanning {len(csv_files)} CSV files...\n")

header_only = []

for csv_file in csv_files:

    relative_path = csv_file.relative_to(INPUT_ROOT)

    try:
        df = pd.read_csv(csv_file)

        if df.empty:
            header_only.append({
                "file": str(relative_path),
                "rows": 0,
                "columns": len(df.columns),
                "column_names": ", ".join(df.columns)
            })
            print(f"HEADER ONLY : {relative_path}")

    except Exception as e:
        print(f"ERROR : {relative_path}")
        print(e)

# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(header_only)
result.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print(f"Total CSV files scanned : {len(csv_files)}")
print(f"Header-only files found : {len(result)}")
print(f"List saved to           : {OUTPUT_FILE}")
print("=" * 60)