"""Visualizations for stability selection results."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.use("Agg")  # non-interactive backend
logger = logging.getLogger("feature_selection")


# ── Feature-family colour palette ──────────────────────────────────────────
# Families derived from the extraction pipeline:
#   a_gat      → GAT perturbation / noise measures
#   b_dfg      → DFG cepstral-peak / harmonics measures
#   c_temporal → Temporal (amplitude, energy, ZCR)
#   d_spectral → Spectral shape (centroid, bandwidth, contrast …)
#   e1_mfcc    → MFCC + Δ + ΔΔ
#   e2_gfcc    → GFCC + Δ + ΔΔ
#   e3_plpcc   → PLPCC + Δ + ΔΔ
#   e5_pncc    → PNCC + Δ + ΔΔ
#   f_lpc      → LPC coefficients, LSF, LPCC
#   i_ratio    → Spectral ratios (LHR, SPI)
#   z_ucla     → UCLA parameters (F0, Jitter, Shimmer, Q50, Flux)
#   z2_schlegel21 → Schlegel21 parameters (Q50_2/10/min, PF, Q25, Dur, HNR)

FAMILY_COLORS = {
    "GAT":        "#E57373",  # red
    "DFG":        "#64B5F6",  # blue
    "Temporal":   "#81C784",  # green
    "Spectral":   "#FFD54F",  # amber
    "MFCC":       "#F06292",  # pink
    "GFCC":       "#BA68C8",  # purple
    "PLPCC":      "#4DD0E1",  # cyan
    "PNCC":       "#AED581",  # lime
    "LPC":        "#FF8A65",  # deep orange
    "Ratio":      "#4DB6AC",  # teal
    "UCLA":       "#FFF176",  # yellow
    "Schlegel21": "#A5D6A7",  # light green
    "Other":      "#B0BEC5",  # grey
}

# GAT features that cannot be identified by prefix alone
_GAT_EXACT = frozenset({
    "f0_mean", "f0_std",
    "mean_jit", "jit_p", "jit_factor", "jit_ratio",
    "ppf", "pvi", "ppq3", "ppq5", "ppq11", "rap_v1", "rap_v2",
    "mean_shim", "shim_p", "apf", "avi", "apq3", "apq5", "apq11",
    "epf", "epq3", "epq5", "epq11",
    "harmonics_intensity", "hnr", "spectral_flatness",
    "wmc_mean", "wmc_max", "cpp",
    "snr1_mean", "snr1_std", "nne",
})

# Temporal features (exact match — short names that would clash with prefixes)
_TEMPORAL_EXACT = frozenset({
    "a_mean", "a_std", "e_mean", "e_std", "zcr_mean", "zcr_std",
})


def _get_family(feature: str) -> str:
    """Return the feature-family name for a given feature column name."""
    f = feature.lower()

    # Schlegel21 — capitalised prefix in the original dataset
    if feature.startswith("Schlegel21_") or f.startswith("schlegel21_"):
        return "Schlegel21"

    # UCLA
    if f.startswith("ucla_"):
        return "UCLA"

    # Spectral ratios (i_ratio)
    if f.startswith(("lhr_", "spi_")):
        return "Ratio"

    # LPC family (f_lpc): lpc_, lsf_, lpcc_
    if f.startswith(("lpc_", "lsf_", "lpcc_")):
        return "LPC"

    # Cepstral families — check delta orders before base prefix
    if f.startswith(("ddpncc_", "dpncc_", "pncc_")):
        return "PNCC"
    if f.startswith(("ddplpcc_", "dplpcc_", "plpcc_")):
        return "PLPCC"
    if f.startswith(("ddgfcc_", "dgfcc_", "gfcc_")):
        return "GFCC"
    if f.startswith(("ddmfcc_", "dmfcc_", "mfcc_")):
        return "MFCC"

    # Spectral shape (d_spectral): s_ prefix
    if f.startswith("s_"):
        return "Spectral"

    # Temporal (c_temporal): exact short names
    if f in _TEMPORAL_EXACT:
        return "Temporal"

    # DFG (b_dfg): cpp_*, cpm_*, cpps_*, cpms_*
    if f.startswith(("cpp_", "cpm_", "cpps_", "cpms_")):
        return "DFG"

    # GAT (a_gat): exact set
    if f in _GAT_EXACT:
        return "GAT"

    return "Other"


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path.name}")


def plot_stability_curve(combined: pd.DataFrame, group_name: str, output_dir: Path):
    """Feature rank vs mean selection probability (mean across l1_ratios)."""
    df = combined.sort_values("mean_pi", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["rank"], df["mean_pi"], linewidth=1.2, color="steelblue")
    ax.axhline(0.70, color="red", linestyle="--", linewidth=0.9, label="π = 0.70")
    ax.axhline(0.50, color="orange", linestyle="--", linewidth=0.9, label="π = 0.50")
    ax.set_xlabel("Feature rank (by mean π)")
    ax.set_ylabel("Mean selection probability (π)")
    ax.set_title(f"Stability curve — Group {group_name}")
    ax.legend()
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, 1.05)
    _save(fig, output_dir / "stability_curve_mean_pi.png")


def _add_family_legend(ax, features: list):
    """Add a family-colour legend to an axes."""
    families_present = sorted({_get_family(f) for f in features})
    patches = [
        mpatches.Patch(color=FAMILY_COLORS.get(fam, FAMILY_COLORS["Other"]), label=fam)
        for fam in families_present
    ]
    ax.legend(
        handles=patches,
        title="Feature family",
        loc="lower right",
        fontsize=6,
        title_fontsize=7,
        framealpha=0.8,
    )


def plot_stability_heatmap(combined: pd.DataFrame, group_name: str,
                            output_dir: Path, top_n: int = 50):
    """Heatmap of selection probabilities for top features × l1_ratio values,
    with y-axis tick labels coloured by feature family."""
    l1_cols = [c for c in combined.columns if c.startswith("pi_l1ratio_")]
    if not l1_cols:
        logger.warning("No l1ratio columns found for heatmap")
        return

    top = combined.nlargest(top_n, "mean_pi")[["feature"] + l1_cols].set_index("feature")
    top.columns = [c.replace("pi_l1ratio_", "l1=") for c in top.columns]

    fig_h = max(8, top_n * 0.22)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.heatmap(
        top,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        linewidths=0.3,
        cbar_kws={"label": "Selection probability (π)"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"Stability heatmap (top {top_n}) — Group {group_name}")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)

    # Colour y-axis labels by feature family
    for tick_label in ax.get_yticklabels():
        feat = tick_label.get_text()
        fam = _get_family(feat)
        tick_label.set_color(FAMILY_COLORS.get(fam, FAMILY_COLORS["Other"]))

    _add_family_legend(ax, list(top.index))
    fig.tight_layout()
    _save(fig, output_dir / "stability_heatmap_top50.png")


def plot_uncapped_vs_capped(comparison: pd.DataFrame,
                             group_name: str, output_dir: Path):
    """Scatter of uncapped vs subject-capped mean stability probability."""
    if comparison is None or comparison.empty:
        return

    flagged = comparison[comparison["pseudoreplication_flag"]]
    not_flagged = comparison[~comparison["pseudoreplication_flag"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        not_flagged["mean_pi_uncapped"],
        not_flagged["mean_pi_subject_capped"],
        alpha=0.5,
        s=25,
        color="steelblue",
        label="Not flagged",
    )
    ax.scatter(
        flagged["mean_pi_uncapped"],
        flagged["mean_pi_subject_capped"],
        alpha=0.8,
        s=40,
        color="crimson",
        marker="^",
        label=f"Flagged (Δπ > 0.20, n={len(flagged)})",
    )

    # Annotate top flagged features
    for _, row in flagged.nlargest(10, "mean_pi_uncapped").iterrows():
        ax.annotate(
            row["feature"],
            (row["mean_pi_uncapped"], row["mean_pi_subject_capped"]),
            fontsize=6,
            textcoords="offset points",
            xytext=(4, 2),
        )

    lim = max(comparison["mean_pi_uncapped"].max(),
              comparison["mean_pi_subject_capped"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean π (uncapped)")
    ax.set_ylabel("Mean π (subject-capped)")
    ax.set_title(f"Uncapped vs subject-capped stability — Group {group_name}")
    ax.legend()
    _save(fig, output_dir / "uncapped_vs_subject_capped.png")


def plot_class_specific_heatmap(per_l1_dfs: dict, combined: pd.DataFrame,
                                 group_name: str, output_dir: Path, top_n: int = 30):
    """
    Heatmap of class-specific selection probabilities for top features.
    Averages class-specific pi across l1_ratios.
    """
    # Get class-specific columns from any l1 result
    first_df = next(iter(per_l1_dfs.values()))
    class_pi_cols = [c for c in first_df.columns if c.startswith("pi_") and
                     not c.startswith("pi_any") and "l1ratio" not in c]
    if not class_pi_cols:
        logger.warning("No class-specific pi columns found for class heatmap")
        return

    top_features = combined.nlargest(top_n, "mean_pi")["feature"].tolist()

    # Average class-specific pi across l1_ratios
    avg_class_pi = None
    n = len(per_l1_dfs)
    for df in per_l1_dfs.values():
        sub = df.set_index("feature").loc[
            [f for f in top_features if f in df["feature"].values], class_pi_cols
        ]
        if avg_class_pi is None:
            avg_class_pi = sub / n
        else:
            avg_class_pi = avg_class_pi.add(sub / n, fill_value=0)

    if avg_class_pi is None or avg_class_pi.empty:
        return

    avg_class_pi.columns = [c.replace("pi_", "") for c in avg_class_pi.columns]

    fig_h = max(6, top_n * 0.22)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(
        avg_class_pi,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="Blues",
        linewidths=0.3,
        cbar_kws={"label": "Class-specific π (avg over l1_ratios)"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"Class-specific stability (top {top_n}) — Group {group_name}")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)

    # Colour y-axis labels by feature family
    for tick_label in ax.get_yticklabels():
        feat = tick_label.get_text()
        fam = _get_family(feat)
        tick_label.set_color(FAMILY_COLORS.get(fam, FAMILY_COLORS["Other"]))

    _add_family_legend(ax, list(avg_class_pi.index))
    fig.tight_layout()
    _save(fig, output_dir / "class_specific_stability_top30.png")


def make_all_plots(combined: pd.DataFrame, per_l1_dfs: dict,
                   comparison: pd.DataFrame, group_name: str, output_dir: Path):
    """Generate all required plots for one group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_stability_curve(combined, group_name, output_dir)
    plot_stability_heatmap(combined, group_name, output_dir)
    plot_uncapped_vs_capped(comparison, group_name, output_dir)
    plot_class_specific_heatmap(per_l1_dfs, combined, group_name, output_dir)
    logger.info(f"[{group_name}] All plots saved to {output_dir}")
