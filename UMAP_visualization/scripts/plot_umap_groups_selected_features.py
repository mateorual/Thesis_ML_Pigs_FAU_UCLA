"""
UMAP visualization per treatment group (B/U/S) — selected features dataset.

Generates two sets of 5 plots per group (10 total per group):
  - 4-stage: Pre / Early Post / Mid Post / Late Post  (suffix _4stages)
  - 3-stage: Pre / Early Post / Late Post             (suffix _3stages)

Colors shared across both sets:
  Pre=Green, Early Post=Red, Late Post=Blue, Mid Post=Orange
Marker = Pig_ID
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
DATA_FILE = BASE_DIR / "Selected_Features_Groups_S_U_B_v1.xlsx"
OUT_DIR   = BASE_DIR / "umap_plots_selected_features"

TREATMENT_FOLDER = {"B": "Bilateral", "U": "Unilateral", "S": "Scar"}
TREATMENT_LABEL  = {"B": "Bilateral", "U": "Unilateral", "S": "Scar"}

# ── Aesthetics ─────────────────────────────────────────────────────────────
STAGE_COLORS = {
    "Pre":        "#2CA02C",   # green
    "Early Post": "#D62728",   # red
    "Mid Post":   "#FF7F0E",   # orange
    "Late Post":  "#1F77B4",   # blue
}
MARKERS = ["o", "s", "^", "D"]   # up to 4 pigs per group

# ── Stage variants ─────────────────────────────────────────────────────────
VARIANTS = {
    "4stages": ["Pre", "Early Post", "Mid Post", "Late Post"],
    "3stages": ["Pre", "Early Post", "Late Post"],
}

# ── UMAP configurations to compare ────────────────────────────────────────
UMAP_CONFIGS = [
    {"n_neighbors": 15, "min_dist": 0.1},    # config 1 – default-ish
    {"n_neighbors": 30, "min_dist": 0.1},    # config 2 – larger neighbourhood
    {"n_neighbors": 15, "min_dist": 0.3},    # config 3 – more spread
    {"n_neighbors": 50, "min_dist": 0.1},    # config 4 – global structure
    {"n_neighbors": 30, "min_dist": 0.5},    # config 5 – balanced spread
    {"n_neighbors": 30, "min_dist": 0.05},   # config 6 – tight clusters
    {"n_neighbors": 15, "min_dist": 0.5},    # config 7 – local, spread out
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_excel(DATA_FILE, sheet_name="Features_Data")

META_COLS    = ["Squeal", "Subject", "Pig_ID", "Treatment", "Week", "Num_Cycles", "Stage"]
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"  {len(df)} samples, {len(feature_cols)} features")


def make_plot(ax, embedding, subset, stage_order, pig_ids, pig_marker, treatment, nn, md):
    for pig in pig_ids:
        for stage in stage_order:
            mask = (subset["Pig_ID"] == pig) & (subset["Stage"] == stage)
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=STAGE_COLORS[stage],
                marker=pig_marker[pig],
                s=40,
                alpha=0.75,
                linewidths=0.3,
                edgecolors="k",
                zorder=2,
            )

    color_patches = [
        mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in stage_order
    ]
    shape_handles = [
        ax.scatter([], [], c="grey", marker=pig_marker[p], s=55, label=p)
        for p in pig_ids
    ]

    leg1 = ax.legend(
        handles=color_patches, title="Stage",
        loc="upper left", framealpha=0.85, fontsize=8, title_fontsize=9,
    )
    ax.legend(
        handles=shape_handles, title="Pig ID",
        loc="lower left", framealpha=0.85, fontsize=8, title_fontsize=9,
    )
    ax.add_artist(leg1)

    ax.set_title(
        f"UMAP — {TREATMENT_LABEL[treatment]}\n"
        f"n_neighbors={nn},  min_dist={md}",
        fontsize=11,
    )
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


# ── Main loop: treatment × variant × config ────────────────────────────────
for treatment, folder_name in TREATMENT_FOLDER.items():
    out_folder = OUT_DIR / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n[{TREATMENT_LABEL[treatment]}]")

    for variant_name, stage_order in VARIANTS.items():
        subset = (
            df[(df["Treatment"] == treatment) & (df["Stage"].isin(stage_order))]
            .copy()
            .reset_index(drop=True)
        )
        pig_ids    = sorted(subset["Pig_ID"].unique())
        pig_marker = {pig: MARKERS[i] for i, pig in enumerate(pig_ids)}

        X = subset[feature_cols].values
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = StandardScaler().fit_transform(X)

        print(f"  [{variant_name}] {len(subset)} samples, stages: {stage_order}")

        for cfg_idx, cfg in enumerate(UMAP_CONFIGS, start=1):
            nn = cfg["n_neighbors"]
            md = cfg["min_dist"]
            print(f"    config {cfg_idx}/{len(UMAP_CONFIGS)}  n_neighbors={nn}, min_dist={md} ...", end=" ", flush=True)

            embedding = umap.UMAP(
                n_neighbors=nn,
                min_dist=md,
                n_components=2,
                random_state=42,
            ).fit_transform(X)

            fig, ax = plt.subplots(figsize=(8, 6))
            make_plot(ax, embedding, subset, stage_order, pig_ids, pig_marker, treatment, nn, md)
            fig.tight_layout()

            fname = (
                f"umap_{folder_name.lower()}_{variant_name}"
                f"_cfg{cfg_idx}_nn{nn}_md{str(md).replace('.', 'p')}.png"
            )
            fig.savefig(out_folder / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved -> {fname}")

print("\nDone. All plots saved.")
