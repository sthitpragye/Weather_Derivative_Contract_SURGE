"""
plot_up_clusters.py

Plots Uttar Pradesh district clusters using GeoPandas.

Output:
    phase1/output/plots/up_cluster_map.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/Users/sthitpragye/Desktop/Finance/SURGE")

GEOJSON = Path(
    "/Users/sthitpragye/Desktop/Finance/SURGE/phase1/geoBoundaries-IND-ADM2.geojson"
)

RUN = "auto"        # change to "k4", "k5", etc.

CLUSTERS = (
    PROJECT_ROOT
    / "phase1"
    / "output"
    / RUN
    / "data"
    / "district_cluster_assignments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "phase1"
    / "output"
    / RUN
    / "plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "up_cluster_map.png"

# ============================================================
# LOAD
# ============================================================

gdf = gpd.read_file(GEOJSON)
clusters = pd.read_csv(CLUSTERS)

gdf["shapeName"] = gdf["shapeName"].str.strip()
clusters["district"] = clusters["district"].str.strip()

# ============================================================
# MODERN -> HISTORICAL NAME MAPPING
# ============================================================

name_map = {
    "Amroha": "Jyotiba Phule Nagar",
    "Allahabad": "Allahabad",          # if your CSV uses Prayagraj, map Prayagraj -> Allahabad
    "Prayagraj": "Allahabad",
    "Ayodhya": "Faizabad",
    "Barabanki": "Bara Banki",
    "Sant Ravi Das Nagar": "Sant Ravidas Nagar (Bhadohi)",
    "Hathras": "Mahamaya Nagar",
    "Kasganj": "Kanshiram Nagar",
    "Lakhimpur Kheri": "Kheri",
    "Maharajganj": "Mahrajganj",
    "Shamli": "Samli",
    "Shravasti": "Shrawasti",
    "Siddharth Nagar": "Siddharthnagar",
}

clusters["shapeName"] = clusters["district"].replace(name_map)

# ============================================================
# MERGE
# ============================================================

merged = gdf.merge(
    clusters,
    on="shapeName",
    how="inner",
)

# ============================================================
# KEEP ONLY UTTAR PRADESH GEOMETRIES
# ============================================================

points = merged.geometry.representative_point()

merged["lon"] = points.x
merged["lat"] = points.y

merged = merged[
    (merged.lon > 76)
    & (merged.lon < 85.5)
    & (merged.lat > 23.5)
    & (merged.lat < 31.5)
].copy()

print(f"Districts plotted : {len(merged)}")

# ============================================================
# COLOURS
# ============================================================

cluster_colors = {
    1: "#4E79A7",
    2: "#E15759",
    3: "#59A14F",
    4: "#F28E2B",
    5: "#B07AA1",
    6: "#76B7B2",
    7: "#EDC948",
    8: "#FF9DA7",
    9: "#9C755F",
    10:"#BAB0AC",
}

merged["colour"] = merged.cluster_id.map(cluster_colors)

# Remove Himachal Hamirpur
merged = merged[
    ~(
        (merged["district"] == "Hamirpur") &
        (merged["lat"] > 30)
    )
]

# Remove Rajasthan Pratapgarh
merged = merged[
    ~(
        (merged["district"] == "Pratapgarh") &
        (merged["lon"] < 77)
    )
]

# Remove Chhattisgarh Balrampur
merged = merged[
    ~(
        (merged["district"] == "Balrampur") &
        (merged["lat"] < 24.5)
    )
]

assert not merged.duplicated(subset=["district"]).any(), \
    "Duplicate districts remain."

assert len(merged) == 75, \
    f"Expected 75 districts, got {len(merged)}"

print(f"Final districts plotted: {len(merged)}")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10,10))

merged.plot(
    ax=ax,
    color=merged["colour"],
    edgecolor="black",
    linewidth=0.8,
)

# ============================================================
# LEGEND
# ============================================================

handles = []

for c in sorted(merged.cluster_id.unique()):
    handles.append(
        Patch(
            facecolor=cluster_colors[c],
            edgecolor="black",
            label=f"Cluster {c}",
        )
    )

ax.legend(
    handles=handles,
    title="Cluster",
    loc="lower left",
    fontsize=11,
)

if RUN == "auto":
    title = "Uttar Pradesh District Clusters (Automatic K)"
else:
    title = f"Uttar Pradesh District Clusters ({RUN.upper()})"

ax.set_title(
    title,
    fontsize=18,
    weight="bold",
)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight",
)

print(f"\nSaved to:\n{OUTPUT}")