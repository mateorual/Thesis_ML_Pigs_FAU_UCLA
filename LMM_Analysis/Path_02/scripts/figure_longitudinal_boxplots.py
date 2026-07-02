# figure_longitudinal_boxplots.py
#
# Longitudinal distribution boxplots — adapted from Figure 03 V2 style.
# One figure per treatment group (S and U), layout: 5 rows (parameters) x N cols (pig pairs).
# Each cell: greyscale boxplot + stripplot, with Wilcoxon rank-sum significance brackets
# (BY correction applied per pig-pair x parameter combination).
#
# Outputs (box_plots/):
#   longitudinal_boxplots_S.png / .pdf
#   longitudinal_boxplots_U.png / .pdf
#   longitudinal_significance_BY.xlsx

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
# BASE is the repository's Path_02/ folder (parent of this scripts/ directory)
BASE = Path(__file__).resolve().parent.parent
INPUT_FILE = os.path.join(BASE, "Selected_Features_Groups_S_U_v1.xlsx")
OUT_DIR    = os.path.join(BASE, "box_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel(INPUT_FILE, sheet_name="Features_Data")
df.columns = df.columns.str.strip()

pig_name_map = {
    "Elvis":              "Pig 1 (M)",
    "Cher and Adele":     "Pig 8&9 (F)",
    "Barry and Stevie":   "Pig 16&17 (M)",
    "Snoop and Dre":      "Pig 18&19 (M)",
    "Michael and Prince": "Pig 10&11 (M)",
    "Paul and John":      "Pig 14&15 (M)",
    "Tina and Aretha":    "Pig 20&21 (F)",
}
df = df.rename(columns={"Pig_ID": "Pair"})
df["Pair"] = df["Pair"].map(lambda x: pig_name_map.get(x, x))

time_order = ["Pre", "Early Post", "Mid Post", "Late Post"]
df["Stage"] = pd.Categorical(df["Stage"], categories=time_order, ordered=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
outcomes = [
    "gfcc_4_std",
    "pncc_4_std",
    "s_entropy_mean",
    "s_flux_mean",
    "Schlegel21_HNR",
]
y_labels = {
    "gfcc_4_std":     "gfcc_4_std (a.u.)",
    "pncc_4_std":     "pncc_4_std (a.u.)",
    "s_entropy_mean": "s_entropy_mean (a.u.)",
    "s_flux_mean":    "s_flux_mean (a.u.)",
    "Schlegel21_HNR": "Schlegel21_HNR (dB)",
}

# Stage colors — ColorBrewer RdBu-inspired: blue = baseline, red = late post
stage_palette = {
    "Pre":        "#2166AC",
    "Early Post": "#74ADD1",
    "Mid Post":   "#FDAE61",
    "Late Post":  "#D73027",
}

# ── Treatment groups ───────────────────────────────────────────────────────────
group_config = {
    "S": {
        "pairs": [
            "Pig 1 (M)", "Pig 8&9 (F)", "Pig 16&17 (M)", "Pig 18&19 (M)"
        ],
        "label": "Scar (S)",
    },
    "U": {
        "pairs": [
            "Pig 10&11 (M)", "Pig 14&15 (M)", "Pig 20&21 (F)"
        ],
        "label": "Unilateral (U)",
    },
}

# ── Helper: significance symbol ────────────────────────────────────────────────
def p_to_symbol(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ── Helper: Wilcoxon rank-sum tests, BY-corrected, for one pig-pair × outcome ──
# Returns DataFrame of rows with p_BY and reject columns (ALL comparisons, not just sig.)
def run_tests(sub_df, outcome):
    groups = [g for g in time_order if g in sub_df["Stage"].values]
    rows, pvals = [], []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x = sub_df.loc[sub_df["Stage"] == g1, outcome].dropna().values
            y = sub_df.loc[sub_df["Stage"] == g2, outcome].dropna().values
            if len(x) >= 2 and len(y) >= 2:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
            else:
                p = np.nan
            rows.append({"Group1": g1, "Group2": g2, "p_raw": p})
            pvals.append(p)

    if not rows:
        return pd.DataFrame(columns=["Group1", "Group2", "p_raw", "p_BY", "reject"])

    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    p_adj = np.full(len(pvals), np.nan)
    reject = np.zeros(len(pvals), dtype=bool)

    if valid.sum() >= 1:
        rej_v, pcor_v, _, _ = multipletests(pvals[valid], method="fdr_by")
        p_adj[valid]  = pcor_v
        reject[valid] = rej_v

    result = pd.DataFrame(rows)
    result["p_BY"]   = p_adj
    result["reject"] = reject
    return result

# ── Pre-compute all tests (avoid running twice for ylim + plotting) ─────────────
# tests[trt][pair][outcome] = full result DataFrame
tests = {}
for trt, cfg in group_config.items():
    tests[trt] = {}
    df_trt = df[df["Treatment"] == trt]
    for pair in cfg["pairs"]:
        tests[trt][pair] = {}
        sub_pair = df_trt[df_trt["Pair"] == pair]
        for outcome in outcomes:
            sub = sub_pair[["Stage", outcome]].dropna()
            tests[trt][pair][outcome] = run_tests(sub, outcome)

# ── Helper: compute shared y-limits per outcome with bracket headroom ──────────
def compute_ylimits(df_trt, pairs, outcomes, tests_trt):
    ylims = {}
    for outcome in outcomes:
        v = df_trt[outcome].dropna()
        if v.empty:
            ylims[outcome] = (0, 1)
            continue
        d_min, d_max = float(v.min()), float(v.max())
        d_range = max(d_max - d_min, 1e-9)

        needed_tops = []
        for pair in pairs:
            sub = df_trt.loc[df_trt["Pair"] == pair, outcome].dropna()
            local_max = float(sub.max()) if len(sub) > 0 else d_max

            res = tests_trt[pair][outcome]
            sig = res[res["reject"]]
            n_sig = int(sig["reject"].sum())

            bracket_top = (local_max
                           + d_range * 0.10
                           + n_sig   * d_range * 0.25
                           + d_range * 0.12)
            needed_tops.append(bracket_top)

        ylims[outcome] = (
            d_min - d_range * 0.05,
            max(needed_tops) if needed_tops else d_max + d_range * 0.20
        )
    return ylims

# ── Helper: draw ordered significance brackets ─────────────────────────────────
def draw_significance(ax, res_df, local_max, data_range):
    sig = res_df[res_df["reject"]].copy()
    if sig.empty:
        return

    sig["x1"]   = sig["Group1"].apply(lambda g: time_order.index(g))
    sig["x2"]   = sig["Group2"].apply(lambda g: time_order.index(g))
    sig["span"] = (sig["x2"] - sig["x1"]).abs()
    sig = sig.sort_values(["span", "x1"], ascending=[True, True])

    spacing = data_range * 0.25
    tip_h   = spacing   * 0.20
    y_base  = local_max + data_range * 0.10

    for _, r in sig.iterrows():
        symbol = p_to_symbol(r["p_BY"])
        if symbol == "":
            continue
        x1, x2 = r["x1"], r["x2"]
        ax.plot([x1, x1, x2, x2],
                [y_base, y_base + tip_h, y_base + tip_h, y_base],
                lw=1.0, c="black")
        ax.text((x1 + x2) / 2, y_base + tip_h, symbol,
                ha="center", va="bottom", fontsize=10)
        y_base += spacing

# ── Main: one figure per treatment group ───────────────────────────────────────
all_stats = []

for trt, cfg in group_config.items():
    pairs   = cfg["pairs"]
    n_cols  = len(pairs)
    n_rows  = len(outcomes)
    df_trt  = df[df["Treatment"] == trt].copy()
    ylims   = compute_ylimits(df_trt, pairs, outcomes, tests[trt])

    sns.set_style("white")
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.4 * n_rows),
        sharey="row"
    )
    # Always 2-D for consistent indexing
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # No figure title — figures will have captions in the thesis

    for c, pair in enumerate(pairs):
        for r, outcome in enumerate(outcomes):
            ax  = axes[r, c]
            sub = df_trt.loc[df_trt["Pair"] == pair, ["Stage", outcome]].dropna()

            sns.boxplot(
                data=sub, x="Stage", y=outcome,
                order=time_order,
                hue="Stage", hue_order=time_order,
                palette=stage_palette,
                dodge=False,
                ax=ax,
                showfliers=False,
                linewidth=1.0,
            )
            sns.stripplot(
                data=sub, x="Stage", y=outcome,
                order=time_order,
                ax=ax,
                color="0.2",
                size=2.5,
                jitter=True,
                alpha=0.55,
            )

            ax.set_ylim(ylims[outcome])
            ax.tick_params(axis="y", labelsize=9)
            if r == n_rows - 1:
                ax.tick_params(axis="x", labelsize=9, rotation=35)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
            else:
                ax.tick_params(axis="x", labelbottom=False)

            # Titles and axis labels
            if r == 0:
                ax.set_title(pair, fontsize=11, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(y_labels[outcome], fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Stage" if r == n_rows - 1 else "", fontsize=9)

            # Significance brackets
            res = tests[trt][pair][outcome]
            v_trt   = df_trt[outcome].dropna()
            d_range = max(float(v_trt.max()) - float(v_trt.min()), 1e-9)
            local_max = float(sub[outcome].max()) if len(sub) > 0 else ylims[outcome][1]
            draw_significance(ax, res, local_max, d_range)

            # Collect significant results
            sig = res[res["reject"]].copy()
            if not sig.empty:
                sig["Treatment"] = trt
                sig["Pair"]      = pair
                sig["Outcome"]   = outcome
                all_stats.append(sig)

    # Strip any per-panel legends seaborn may have added
    for row in axes:
        for cell in row:
            if cell.get_legend() is not None:
                cell.get_legend().remove()

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor=stage_palette[s], edgecolor="0.3", label=s)
        for s in time_order
    ]
    plt.tight_layout()
    fig.legend(
        handles=legend_handles,
        title="Stage",
        loc="lower center",
        ncol=4,
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0),
    )
    fig.subplots_adjust(bottom=0.08)
    base = os.path.join(OUT_DIR, f"longitudinal_boxplots_{trt}")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf",           bbox_inches="tight")
    plt.close()
    print(f"Saved: {base}.png / .pdf")

# ── Export significance table ──────────────────────────────────────────────────
if all_stats:
    stats_out = pd.concat(all_stats, ignore_index=True)
    cols_front = ["Treatment", "Pair", "Outcome", "Group1", "Group2",
                  "p_raw", "p_BY", "reject"]
    stats_out = stats_out[[c for c in cols_front if c in stats_out.columns]]
    out_xl = os.path.join(OUT_DIR, "longitudinal_significance_BY.xlsx")
    stats_out.to_excel(out_xl, index=False)
    print(f"Significance table saved: {out_xl}")
else:
    print("No significant results found.")

print("Done. Outputs in:", OUT_DIR)
