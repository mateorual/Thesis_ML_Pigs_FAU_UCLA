"""
UMAP visualization — all treatments combined (B / U / S).

Generates two sets of 5 plots (10 total):
  - 4-stage: Pre / Early Post / Mid Post / Late Post  (suffix _4stages)
  - 3-stage: Pre / Early Post / Late Post             (suffix _3stages)

  - Color  = Treatment (Bilateral / Unilateral / Scar)
  - Marker = Stage
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent  # UMAP_visualization/ (parent of this scripts/ dir)
DATA_FILE = BASE_DIR / "Consolidated_Features_Groups_S_U_B_v1.xlsx"
OUT_DIR   = BASE_DIR / "umap_plots" / "All_Treatments"

# ── Aesthetics ─────────────────────────────────────────────────────────────
TREATMENT_COLORS = {
    "B": "#9467BD",   # purple  – Bilateral
    "U": "#FF7F0E",   # orange  – Unilateral
    "S": "#17BECF",   # teal    – Scar
}
TREATMENT_LABEL = {"B": "Bilateral", "U": "Unilateral", "S": "Scar"}

STAGE_MARKERS = {
    "Pre":        "o",
    "Early Post": "s",
    "Mid Post":   "^",
    "Late Post":  "D",
}

# ── Stage variants ─────────────────────────────────────────────────────────
VARIANTS = {
    "4stages": ["Pre", "Early Post", "Mid Post", "Late Post"],
    "3stages": ["Pre", "Early Post", "Late Post"],
}

# ── UMAP configurations ────────────────────────────────────────────────────
UMAP_CONFIGS = [
    {"n_neighbors": 15, "min_dist": 0.1},
    {"n_neighbors": 30, "min_dist": 0.1},
    {"n_neighbors": 15, "min_dist": 0.3},
    {"n_neighbors": 50, "min_dist": 0.1},
    {"n_neighbors": 30, "min_dist": 0.5},
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_excel(DATA_FILE, sheet_name="Features_Data")

META_COLS    = ["Squeal", "Subject", "Pig_ID", "Treatment", "Week", "Num_Cycles", "Stage"]
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"  {len(df)} samples, {len(feature_cols)} features")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Main loop: variant × config ────────────────────────────────────────────
for variant_name, stage_order in VARIANTS.items():
    subset = (
        df[df["Stage"].isin(stage_order)]
        .copy()
        .reset_index(drop=True)
    )

    X = subset[feature_cols].values
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    print(f"\n[{variant_name}]  {len(subset)} samples, stages: {stage_order}")

    for cfg_idx, cfg in enumerate(UMAP_CONFIGS, start=1):
        nn = cfg["n_neighbors"]
        md = cfg["min_dist"]
        print(f"  config {cfg_idx}/5  n_neighbors={nn}, min_dist={md} ...", end=" ", flush=True)

        embedding = umap.UMAP(
            n_neighbors=nn,
            min_dist=md,
            n_components=2,
            random_state=42,
        ).fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))

        for treatment in ["B", "U", "S"]:
            for stage in stage_order:
                mask = (subset["Treatment"] == treatment) & (subset["Stage"] == stage)
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=TREATMENT_COLORS[treatment],
                    marker=STAGE_MARKERS[stage],
                    s=40,
                    alpha=0.65,
                    linewidths=0.3,
                    edgecolors="k",
                    zorder=2,
                )

        # Legend: Treatment → colors
        color_patches = [
            mpatches.Patch(color=TREATMENT_COLORS[t], label=TREATMENT_LABEL[t])
            for t in ["B", "U", "S"]
        ]
        # Legend: Stage → marker shapes (dummy scatter)
        shape_handles = [
            ax.scatter([], [], c="grey", marker=STAGE_MARKERS[s], s=55, label=s)
            for s in stage_order
        ]

        leg1 = ax.legend(
            handles=color_patches, title="Treatment",
            loc="upper left", framealpha=0.85, fontsize=8, title_fontsize=9,
        )
        ax.legend(
            handles=shape_handles, title="Stage",
            loc="lower left", framealpha=0.85, fontsize=8, title_fontsize=9,
        )
        ax.add_artist(leg1)

        ax.set_title(
            f"UMAP — All Treatments\n"
            f"n_neighbors={nn},  min_dist={md}",
            fontsize=11,
        )
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        fig.tight_layout()

        fname = (
            f"umap_all_treatments_{variant_name}"
            f"_cfg{cfg_idx}_nn{nn}_md{str(md).replace('.', 'p')}.png"
        )
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {fname}")

print("\nDone. All plots saved.")
