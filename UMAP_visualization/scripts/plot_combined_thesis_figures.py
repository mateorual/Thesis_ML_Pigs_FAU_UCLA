"""
Combined multi-panel thesis figure for each ThesisFigure folder.

Layout:
  Left  (Panel A) : large group/treatment UMAP plot
  Right (Panels B+): individual pig UMAP plots stacked vertically

Output per folder:
  combined_figure_all_{T}.png / .pdf      for umap_plots/
  combined_figure_selected_{T}.png / .pdf for umap_plots_selected_features/
where T ∈ {B, S, U}.
"""

import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent  # UMAP_visualization/ (parent of this scripts/ dir)

FOLDER_TO_CODE = {"Bilateral": "B", "Scar": "S", "Unilateral": "U"}
PANEL_LABELS   = list("ABCDEFGH")


def output_stem(thesis_dir: Path) -> str:
    """Return the combined-figure filename stem for this ThesisFigure folder."""
    treatment_folder = thesis_dir.parent          # e.g. …/Bilateral
    root_folder      = thesis_dir.parent.parent   # e.g. …/umap_plots_selected_features
    code = FOLDER_TO_CODE.get(treatment_folder.name, treatment_folder.name)
    kind = "selected" if "selected" in root_folder.name else "all"
    return f"combined_figure_{kind}_{code}"


def pig_sort_key(path: Path) -> int:
    """Sort pig plots by the leading number in the pig name (e.g. pig_10and11 → 10)."""
    m = re.search(r"umap_pig_(\d+)", path.name)
    return int(m.group(1)) if m else 999


def make_combined(thesis_dir: Path) -> None:
    pngs = [p for p in thesis_dir.glob("*.png") if "combined_figure" not in p.name]

    group_plots = [p for p in pngs if not re.search(r"umap_pig_", p.name)]
    pig_plots   = sorted(
        [p for p in pngs if re.search(r"umap_pig_", p.name)],
        key=pig_sort_key,
    )

    if not group_plots:
        print(f"  WARNING: no group plot found — skipping {thesis_dir.parent.name}.")
        return
    if not pig_plots:
        print(f"  WARNING: no pig plots found — skipping {thesis_dir.parent.name}.")
        return

    group_plot = group_plots[0]
    n_pigs     = len(pig_plots)

    print(f"  Group ({n_pigs} pigs): {group_plot.name}")
    for p in pig_plots:
        print(f"    {p.name}")

    # ── Figure & GridSpec ──────────────────────────────────────────────────
    # Height scales with number of pig panels; minimum 9 inches.
    fig_h = max(n_pigs * 3.4, 9.0)
    fig   = plt.figure(figsize=(15, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        n_pigs, 2,
        width_ratios=[1.55, 1],
        hspace=0.05,
        wspace=0.04,
        left=0.01, right=0.99,
        top=0.97,  bottom=0.01,
    )

    # ── Panel A: group plot (left, full height) ────────────────────────────
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(mpimg.imread(group_plot))
    ax_main.axis("off")
    ax_main.text(
        0.013, 0.988, "A",
        transform=ax_main.transAxes,
        fontsize=20, fontweight="bold", va="top", ha="left",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=3),
    )

    # ── Panels B–E: individual pig plots (right column, stacked) ──────────
    for i, pig_path in enumerate(pig_plots):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(mpimg.imread(pig_path))
        ax.axis("off")
        ax.text(
            0.013, 0.988, PANEL_LABELS[i + 1],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=3),
        )

    # ── Save ──────────────────────────────────────────────────────────────
    stem = output_stem(thesis_dir)
    for ext in ("png", "pdf"):
        out = thesis_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved -> {out.name}")

    plt.close(fig)


# ── Run across all ThesisFigure folders ───────────────────────────────────
thesis_dirs = sorted(BASE_DIR.rglob("ThesisFigure"))
print(f"Found {len(thesis_dirs)} ThesisFigure folders.\n")

for td in thesis_dirs:
    label = f"{td.parent.parent.name} / {td.parent.name}"
    print(f"[{label}]")
    make_combined(td)
    print()

print("Done.")
