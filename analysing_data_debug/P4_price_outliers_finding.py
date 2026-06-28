"""
P4_price_outliers_finding.py

Lists all CSV files flagged with P4_price_outliers.

Output:
    analysing_data_debug/P4_price_outliers_files.csv
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "analysing_data_debug"
    / "audit_output_debug"
    / "file_audit.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "analysing_data_debug"
    / "P4_price_outliers_files.csv"
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

if "P4_price_outliers" not in df.columns:
    raise ValueError("Column 'P4_price_outliers' not found.")

# ============================================================
# FILTER
# ============================================================

outliers = df[df["P4_price_outliers"] == True].copy()

# Keep all original columns
outliers.to_csv(OUTPUT_FILE, index=False)

print(f"Found {len(outliers)} files with P4_price_outliers.")
print(f"Saved to:\n{OUTPUT_FILE}")