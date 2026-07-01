"""Summarize selected features across treatment groups.

This script reads the per-group ``features_for_LMM.csv`` files produced by
``run_feature_selection.py`` and writes:

- per-group selected-feature tables with feature-family labels
- S/U/B overlap and union CSV files
- Venn diagram and family-breakdown plots
- a compact Markdown summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3, venn3_circles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import FAMILY_COLORS, _get_family  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create selected-feature summaries for S, U, and B."
    )
    parser.add_argument(
        "--results-su",
        default="results/S_U",
        help="Result folder containing S/final and U/final outputs.",
    )
    parser.add_argument(
        "--results-b",
        default="results/B",
        help="Result folder containing B/final outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/selected_feature_summaries",
        help="Output directory for summary tables and plots.",
    )
    return parser.parse_args()


def load_group_features(path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"{path} must contain a 'feature' column.")
    df["family"] = df["feature"].apply(_get_family)
    df["group"] = group
    return df


def intersection_df(dfs: dict[str, pd.DataFrame], groups: list[str]) -> pd.DataFrame:
    sets = {g: set(dfs[g]["feature"]) for g in dfs}
    features = set.intersection(*[sets[g] for g in groups])
    rows = []
    for feature in sorted(features):
        row = {"feature": feature, "family": _get_family(feature)}
        for group in groups:
            sub = dfs[group].set_index("feature")
            row[f"mean_pi_{group}"] = round(float(sub.loc[feature, "mean_pi_uncapped"]), 4)
            row[f"decision_{group}"] = sub.loc[feature, "final_decision"]
        rows.append(row)
    return pd.DataFrame(rows)


def union_df(dfs: dict[str, pd.DataFrame], groups: list[str]) -> pd.DataFrame:
    sets = {g: set(dfs[g]["feature"]) for g in dfs}
    features = set.union(*[sets[g] for g in groups])
    rows = []
    for feature in sorted(features):
        row = {"feature": feature, "family": _get_family(feature)}
        for group in groups:
            sub = dfs[group].set_index("feature")
            if feature in sub.index:
                row[f"mean_pi_{group}"] = round(float(sub.loc[feature, "mean_pi_uncapped"]), 4)
                row[f"selected_{group}"] = True
            else:
                row[f"mean_pi_{group}"] = ""
                row[f"selected_{group}"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(dfs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    sets = {g: set(dfs[g]["feature"]) for g in ["S", "U", "B"]}

    fig, ax = plt.subplots(figsize=(7, 6))
    venn = venn3(
        subsets=[sets["S"], sets["U"], sets["B"]],
        set_labels=("Scar (S)", "Unilateral (U)", "Bilateral (B)"),
        set_colors=("#4CAF50", "#EF5350", "#42A5F5"),
        alpha=0.55,
        ax=ax,
    )
    venn3_circles(subsets=[sets["S"], sets["U"], sets["B"]], linewidth=1.2, ax=ax)
    if venn.set_labels:
        for text in venn.set_labels:
            if text:
                text.set_fontsize(12)
                text.set_fontweight("bold")
    if venn.subset_labels:
        for text in venn.subset_labels:
            if text:
                text.set_fontsize(11)
    fig.tight_layout()
    fig.savefig(output_dir / "venn_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    family_counts = pd.DataFrame(
        {group: dfs[group]["family"].value_counts() for group in ["S", "U", "B"]}
    ).fillna(0).astype(int)
    family_counts = family_counts.reindex(sorted(family_counts.index), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = range(3)
    bottom = [0, 0, 0]
    for family in family_counts.index:
        values = [family_counts.loc[family, group] for group in ["S", "U", "B"]]
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=FAMILY_COLORS.get(family, "#B0BEC5"),
            label=family,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(["Scar (S)", "Unilateral (U)", "Bilateral (B)"])
    ax.set_ylabel("Number of selected features")
    ax.legend(title="Feature family", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, max(bottom) + 2)
    for i, total in enumerate(bottom):
        ax.text(i, total + 0.2, str(total), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "family_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    return df.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "S": ROOT / args.results_su / "S" / "final" / "features_for_LMM.csv",
        "U": ROOT / args.results_su / "U" / "final" / "features_for_LMM.csv",
        "B": ROOT / args.results_b / "B" / "final" / "features_for_LMM.csv",
    }
    dfs = {group: load_group_features(path, group) for group, path in paths.items()}

    for group, df in dfs.items():
        df.to_csv(output_dir / f"feature_summary_{group}.csv", index=False)

    inter_all = intersection_df(dfs, ["S", "U", "B"])
    inter_su = intersection_df(dfs, ["S", "U"])
    inter_ub = intersection_df(dfs, ["U", "B"])
    inter_sb = intersection_df(dfs, ["S", "B"])

    inter_su_only = inter_su[~inter_su["feature"].isin(inter_all["feature"])]
    inter_ub_only = inter_ub[~inter_ub["feature"].isin(inter_all["feature"])]
    inter_sb_only = inter_sb[~inter_sb["feature"].isin(inter_all["feature"])]

    inter_all.to_csv(output_dir / "intersection_all_three.csv", index=False)
    inter_su_only.to_csv(output_dir / "intersection_S_U.csv", index=False)
    inter_ub_only.to_csv(output_dir / "intersection_U_B.csv", index=False)
    inter_sb_only.to_csv(output_dir / "intersection_S_B.csv", index=False)

    union_su = union_df(dfs, ["S", "U"])
    union_all = union_df(dfs, ["S", "U", "B"])
    union_su.to_csv(output_dir / "union_S_U.csv", index=False)
    union_all.to_csv(output_dir / "union_S_U_B.csv", index=False)

    write_plots(dfs, output_dir)

    lines = [
        "# Feature Selection Summary: Groups S, U, B",
        "",
        "## Candidate Counts",
        "",
        "| Group | Strong | Moderate | Total for LMM |",
        "|-------|--------|----------|---------------|",
    ]
    for group in ["S", "U", "B"]:
        df = dfs[group]
        n_strong = int((df["final_decision"] == "Strong candidate").sum())
        n_moderate = int((df["final_decision"] == "Moderate candidate").sum())
        lines.append(f"| {group} | {n_strong} | {n_moderate} | {len(df)} |")

    lines.extend(["", "## Selected Features Per Group", ""])
    for group in ["S", "U", "B"]:
        cols = [
            "feature",
            "family",
            "mean_pi_uncapped",
            "alpha_consistency_label",
            "final_decision",
        ]
        lines.extend([f"### Group {group}", "", to_markdown(dfs[group][cols]), ""])

    lines.extend(
        [
            "## Overlap Tables",
            "",
            f"### S and U and B ({len(inter_all)} features)",
            "",
            to_markdown(inter_all),
            "",
            f"### S and U only ({len(inter_su_only)} features, excluding B)",
            "",
            to_markdown(inter_su_only),
            "",
            f"### U and B only ({len(inter_ub_only)} features, excluding S)",
            "",
            to_markdown(inter_ub_only),
            "",
            f"### S and B only ({len(inter_sb_only)} features, excluding U)",
            "",
            to_markdown(inter_sb_only),
            "",
            "## Union Tables",
            "",
            f"### S or U ({len(union_su)} features)",
            "",
            to_markdown(union_su),
            "",
            f"### S or U or B ({len(union_all)} features)",
            "",
            to_markdown(union_all),
            "",
        ]
    )
    (output_dir / "feature_selection_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Summary outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
