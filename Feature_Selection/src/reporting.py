"""Generate the final candidate feature tables and Markdown report."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("feature_selection")


def build_final_table(combined: pd.DataFrame, comparison: pd.DataFrame,
                       boruta_df: pd.DataFrame, output_dir: Path,
                       group_name: str) -> pd.DataFrame:
    """
    Merge all signals into the final candidate feature table.
    Applies decision rules to assign final_decision label.
    Saves final_candidate_features.csv and features_for_LMM.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = combined[["feature", "mean_pi", "max_pi",
                   "n_l1ratios_pi_ge_0.70", "alpha_consistency_label"]].copy()
    df = df.rename(columns={"mean_pi": "mean_pi_uncapped", "max_pi": "max_pi_uncapped"})

    # Merge subject-capped comparison
    if comparison is not None and not comparison.empty:
        cap_cols = ["feature", "mean_pi_subject_capped", "delta_pi", "pseudoreplication_flag"]
        df = df.merge(comparison[cap_cols], on="feature", how="left")
    else:
        df["mean_pi_subject_capped"] = np.nan
        df["delta_pi"] = np.nan
        df["pseudoreplication_flag"] = False

    df["pseudoreplication_flag"] = df["pseudoreplication_flag"].fillna(False)
    df["delta_pi_capped"] = df["delta_pi"]
    df = df.drop(columns=["delta_pi"], errors="ignore")

    # Merge Boruta
    if boruta_df is not None and not boruta_df.empty:
        df = df.merge(
            boruta_df[["feature", "boruta_decision", "boruta_rank"]],
            on="feature", how="left"
        )
        df["boruta_decision"] = df["boruta_decision"].fillna("Not run")
    else:
        df["boruta_decision"] = "Not run"
        df["boruta_rank"] = np.nan

    # Apply decision rules
    def _decision(row):
        pi_cap = row["mean_pi_subject_capped"] if not np.isnan(
            row.get("mean_pi_subject_capped", float("nan"))) else 0.0
        flag = bool(row["pseudoreplication_flag"])
        label = row["alpha_consistency_label"]

        if label == "High" and pi_cap >= 0.50 and not flag:
            return "Strong candidate", "High alpha-consistency, stable under capping"
        elif label in ("High", "Moderate") and pi_cap >= 0.40 and not flag:
            return "Moderate candidate", "Moderate alpha-consistency, stable under capping"
        elif flag:
            return "Exploratory only", "Stability drops after subject-stage capping"
        elif label == "High" and flag:
            return "Exploratory only", "High pi but pseudoreplication flagged"
        else:
            return "Exclude", "Low stability or no useful secondary support"

    decisions = df.apply(_decision, axis=1, result_type="expand")
    df["final_decision"] = decisions[0]
    df["reason"] = decisions[1]

    # Column order
    col_order = [
        "feature", "mean_pi_uncapped", "max_pi_uncapped",
        "n_l1ratios_pi_ge_0.70", "alpha_consistency_label",
        "mean_pi_subject_capped", "delta_pi_capped", "pseudoreplication_flag",
        "boruta_decision", "boruta_rank", "final_decision", "reason",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values("mean_pi_uncapped", ascending=False).reset_index(drop=True)

    df.to_csv(output_dir / "final_candidate_features.csv", index=False)

    # Subset for LMM
    lmm_mask = df["final_decision"].isin({"Strong candidate", "Moderate candidate"})
    df_lmm = df[lmm_mask][["feature", "mean_pi_uncapped", "alpha_consistency_label",
                              "final_decision"]].reset_index(drop=True)
    df_lmm.to_csv(output_dir / "features_for_LMM.csv", index=False)

    n_strong = (df["final_decision"] == "Strong candidate").sum()
    n_mod = (df["final_decision"] == "Moderate candidate").sum()
    n_exp = (df["final_decision"] == "Exploratory only").sum()
    logger.info(
        f"[{group_name}] Final decisions: {n_strong} strong, {n_mod} moderate, "
        f"{n_exp} exploratory-only, "
        f"{(df['final_decision']=='Exclude').sum()} excluded"
    )
    return df


def write_report(output_dir: Path, args, version_info: dict,
                 summary_counts: dict, group_results: dict,
                 conv_summaries: dict):
    """
    Write the Markdown feature-selection report.
    group_results: dict keyed by treatment group, each with final_df, n_orig, n_nzv, n_corr.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = output_dir / "feature_selection_report.md"

    lines = []
    a = lines.append

    a("# Feature Selection Report")
    a("")
    a(f"**Generated:** {now}")
    a("")
    a("## 1. Dataset")
    a(f"- **Path:** `{args.input}`")
    a(f"- **Sheet:** `{args.sheet}`")
    a("")
    a("## 2. Python Environment")
    a("| Package | Version |")
    a("|---------|---------|")
    for pkg, ver in version_info.items():
        a(f"| {pkg} | {ver} |")
    a("")
    a("## 3. Dataset Counts")
    for label, cnt_str in summary_counts.items():
        a(f"### {label}")
        a("```")
        a(cnt_str)
        a("```")
        a("")

    a("## 4. Analysis Goal")
    a(
        "Exploratory feature selection to identify acoustic features that discriminate "
        "between Pre, Early Post, and Late Post stages separately for each requested "
        "treatment group. Selected features are candidates for downstream "
        "Linear Mixed Model (LMM) analysis."
    )
    a("")
    a("## 5. Independence Assumption")
    a(
        "> Feature selection treated squeals as independent observations; this is an "
        "acknowledged limitation of the exploratory stage. All inferential conclusions "
        "should be drawn exclusively from subsequent LMMs that account for "
        "subject-level dependence."
    )
    a("")

    a("## 6. Preprocessing")
    a(f"- **Near-zero variance threshold:** `{args.variance_threshold}`")
    a(f"- **High-correlation threshold:** `{args.correlation_threshold}`")
    a(
        "\n> Near-zero-variance and high-correlation filtering were applied once before "
        "stability selection as unsupervised preprocessing steps using thresholds fixed "
        "a priori. Because these filters do not use class labels, they are unlikely to "
        "introduce label leakage in the exploratory feature-selection stage. Any future "
        "predictive-performance estimation would require the full preprocessing pipeline "
        "to be nested within the training folds."
    )
    a("")

    a("## 7. Stability Selection Settings")
    a(f"- **B (replicates):** `{args.B}`")
    a(f"- **l1_ratio grid:** `{args.l1_ratios}`")
    a(f"- **C grid:** `np.logspace(-3, 2, 30)` ({30} values)")
    a(f"- **C selection rule:** 1-standard-error rule (5-fold stratified CV on full dataset, "
      f"applied once per l1_ratio before the resampling loop)")
    a(f"- **Subsample fraction:** `{args.subsample_fraction}` (stratified by class)")
    a(f"- **Class weighting:** `balanced`")
    a(f"- **Coefficient threshold:** `1e-8`")
    a(f"- **Random state:** `{args.random_state}`")
    a("")

    a("## 8. Subject-Stage-Capped Sensitivity Analysis")
    a(f"- **Cap per Subject × Stage cell:** `{args.subject_stage_cap}`")
    a(
        "\nFeatures whose stability drops by more than 0.20 after subject-stage capping "
        "are flagged as potentially subject-dominated. They are marked as "
        "'Exploratory only' rather than removed."
    )
    a("")

    a("## 9. Boruta")
    boruta_ran = getattr(args, "run_boruta", False)
    if boruta_ran:
        a("Boruta was requested. See individual group sections for results.")
        a(
            "\n> Boruta was used as a complementary nonlinear/all-relevant selector. "
            "Agreement between Boruta and elastic-net stability selection was interpreted "
            "as additional support. Boruta rejection was not used to discard elastic-net-"
            "selected features, because random-forest importance can be difficult to "
            "interpret in the presence of strongly correlated acoustic predictors."
        )
    else:
        a("Boruta was not requested (`--run-boruta` not passed).")
    a("")

    a("## 10. Feature Count Summary")
    a("| Group | Original | After NZV | After Corr. | Strong | Moderate | Exploratory |")
    a("|-------|----------|-----------|-------------|--------|----------|-------------|")
    for grp, res in group_results.items():
        n_strong = (res["final_df"]["final_decision"] == "Strong candidate").sum()
        n_mod = (res["final_df"]["final_decision"] == "Moderate candidate").sum()
        n_exp = (res["final_df"]["final_decision"] == "Exploratory only").sum()
        a(f"| {grp} | {res['n_orig']} | {res['n_nzv']} | {res['n_corr']} "
          f"| {n_strong} | {n_mod} | {n_exp} |")
    a("")

    a("## 11. Convergence Warnings")
    for grp, by_lr in conv_summaries.items():
        a(f"### Group {grp}")
        for lr, count in by_lr.items():
            a(f"- l1_ratio={lr}: {count} warnings")
    a("")

    a("## 12. Top Candidate Features")
    for grp, res in group_results.items():
        a(f"### Group {grp}")
        top = res["final_df"][
            res["final_df"]["final_decision"].isin({"Strong candidate", "Moderate candidate"})
        ][["feature", "mean_pi_uncapped", "alpha_consistency_label",
           "mean_pi_subject_capped", "pseudoreplication_flag", "final_decision"]].head(20)
        if top.empty:
            a("No strong or moderate candidates found.")
        else:
            a(top.to_markdown(index=False))
        a("")

    a("## 13. Interpretation Note")
    a(
        "Candidate features listed above are proposed for downstream LMM testing. "
        "Feature-selection results are **not inferential**: they do not constitute "
        "statistical evidence of group differences. All biological or statistical "
        "conclusions must be drawn from LMMs that correctly account for the "
        "repeated-measures structure of the data."
    )
    a("")

    a("## 14. Output Files")
    a("### Data summaries")
    a("- `results/data_summary/counts_by_treatment.csv`")
    a("- `results/data_summary/counts_by_treatment_stage.csv`")
    a("- `results/data_summary/counts_by_subject_treatment_stage.csv`")
    for grp in group_results:
        a(f"### Group {grp}")
        a(f"- `results/{grp}/preprocessing/nzv_removed_features.csv`")
        a(f"- `results/{grp}/preprocessing/features_after_nzv.csv`")
        a(f"- `results/{grp}/preprocessing/correlation_removed_features.csv`")
        a(f"- `results/{grp}/preprocessing/features_after_correlation_filter.csv`")
        a(f"- `results/{grp}/stability_selection/stability_scores_combined.csv`")
        a(f"- `results/{grp}/stability_selection/chosen_C_values.csv`")
        a(f"- `results/{grp}/subject_capped/stability_scores_combined_subject_capped.csv`")
        a(f"- `results/{grp}/subject_capped/uncapped_vs_subject_capped_comparison.csv`")
        a(f"- `results/{grp}/boruta/boruta_results.csv`")
        a(f"- `results/{grp}/final/final_candidate_features.csv`")
        a(f"- `results/{grp}/final/features_for_LMM.csv`")
        a(f"- `results/{grp}/plots/stability_curve_mean_pi.png`")
        a(f"- `results/{grp}/plots/stability_heatmap_top50.png`")
        a(f"- `results/{grp}/plots/uncapped_vs_subject_capped.png`")
        a(f"- `results/{grp}/plots/class_specific_stability_top30.png`")
    a("- `results/run_log.txt`")
    a("- `results/feature_selection_report.md`")
    a("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {report_path}")
    return report_path
