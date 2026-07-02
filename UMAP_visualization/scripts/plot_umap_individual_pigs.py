"""
UMAP visualization per individual pig (3-stage: Pre / Early Post / Late Post).

For each pig: 5 UMAP configs × 1 plot, saved to
  umap_plots/<TreatmentFolder>/<Pig_ID>/
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
OUT_DIR   = BASE_DIR / "umap_plots"

TREATMENT_FOLDER = {"B": "Bilateral", "U": "Unilateral", "S": "Scar"}

# ── Aesthetics ─────────────────────────────────────────────────────────────
STAGE_ORDER  = ["Pre", "Early Post", "Late Post"]
STAGE_COLORS = {
    "Pre":        "#2CA02C",   # green
    "Early Post": "#D62728",   # red
    "Late Post":  "#1F77B4",   # blue
}
MARKERS = ["o", "s", "^", "D"]   # must match order in plot_umap_groups.py

# ── UMAP configurations ────────────────────────────────────────────────────
UMAP_CONFIGS = [
    {"n_neighbors":  5, "min_dist": 0.1},
    {"n_neighbors":  5, "min_dist": 0.3},
    {"n_neighbors": 10, "min_dist": 0.1},
    {"n_neighbors": 10, "min_dist": 0.3},
    {"n_neighbors": 15, "min_dist": 0.1},
    {"n_neighbors":  5, "min_dist": 0.5},
    {"n_neighbors": 15, "min_dist": 0.3},
    {"n_neighbors": 15, "min_dist": 0.05},
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_excel(DATA_FILE, sheet_name="Features_Data")

META_COLS    = ["Squeal", "Subject", "Pig_ID", "Treatment", "Week", "Num_Cycles", "Stage"]
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"  {len(df)} samples, {len(feature_cols)} features")

# ── Main loop: treatment → pig → config ───────────────────────────────────
for treatment, folder_name in TREATMENT_FOLDER.items():
    pig_ids = sorted(
        df[(df["Treatment"] == treatment) & (df["Stage"].isin(STAGE_ORDER))]["Pig_ID"].unique()
    )
    pig_marker_map = {pig: MARKERS[i] for i, pig in enumerate(pig_ids)}
    print(f"\n[{folder_name}]  pigs: {pig_ids}")

    for pig in pig_ids:
        out_folder = OUT_DIR / folder_name / pig
        out_folder.mkdir(parents=True, exist_ok=True)

        subset = (
            df[
                (df["Treatment"] == treatment)
                & (df["Pig_ID"] == pig)
                & (df["Stage"].isin(STAGE_ORDER))
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(subset) < 10:
            print(f"  Skipping {pig} — too few samples ({len(subset)})")
            continue

        X = subset[feature_cols].values
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = StandardScaler().fit_transform(X)

        print(f"  {pig}  ({len(subset)} samples)")

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

            for stage in STAGE_ORDER:
                mask = subset["Stage"] == stage
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=STAGE_COLORS[stage],
                    marker=pig_marker_map[pig],
                    s=40,
                    alpha=0.75,
                    linewidths=0.3,
                    edgecolors="k",
                    zorder=2,
                )

            color_patches = [
                mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER
            ]
            ax.legend(
                handles=color_patches, title="Stage",
                loc="upper left", framealpha=0.85, fontsize=8, title_fontsize=9,
            )

            pig_label = pig.replace("&", "and")
            ax.set_title(
                f"UMAP — {folder_name} / {pig}\n"
                f"n_neighbors={nn},  min_dist={md}",
                fontsize=11,
            )
            ax.set_xlabel("UMAP 1", fontsize=10)
            ax.set_ylabel("UMAP 2", fontsize=10)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            fig.tight_layout()

            safe_pig = pig.replace("&", "and").replace(" ", "_").lower()
            fname = (
                f"umap_{safe_pig}_3stages"
                f"_cfg{cfg_idx}_nn{nn}_md{str(md).replace('.', 'p')}.png"
            )
            fig.savefig(out_folder / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved -> {fname}")

print("\nDone. All per-pig plots saved.")
