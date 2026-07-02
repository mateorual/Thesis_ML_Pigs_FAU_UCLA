"""
Exploratory feature selection pipeline for pig squeal acoustic features.

Runs separately for Scar (S) and Unilateral (U) groups:
  1. Data loading and validation
  2. Near-zero-variance and high-correlation preprocessing
  3. Multinomial elastic-net stability selection across l1_ratio grid
  4. Subject-stage-capped sensitivity analysis
  5. Optional Boruta secondary selector
  6. Final candidate feature table
  7. Visualizations and Markdown report

IMPORTANT — Statistical assumption:
  Feature selection treats squeals as independent observations.
  This is an acknowledged limitation of the exploratory stage.
  All inferential conclusions must be drawn from subsequent LMMs
  that account for subject-level dependence.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# ── make src importable when called from any directory ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, make_dirs, get_version_info
from src.data_loading import load_and_validate, prepare_group
from src.preprocessing import run_preprocessing
from src.stability_selection import run_stability_selection
from src.subject_capped import run_subject_capped, build_comparison
from src.boruta_selector import run_boruta
from src.plotting import make_all_plots
from src.reporting import build_final_table, write_report


# ── sanity checks ──────────────────────────────────────────────────────────

def run_sanity_checks(group_data: dict, X_clean: np.ndarray,
                       feature_names: list, group_name: str, logger):
    """Assertions that guard against common implementation errors."""
    meta = group_data["metadata"]
    y = group_data["y"]
    orig_features = group_data["feature_names"]

    # Correct row count
    assert len(X_clean) == len(y), \
        f"[{group_name}] Row count mismatch: X={len(X_clean)}, y={len(y)}"

    # Only 3 classes
    classes = np.unique(y)
    assert len(classes) == 3, \
        f"[{group_name}] Expected 3 classes, got {list(classes)}"

    # No metadata in feature matrix
    meta_cols_set = set(meta.columns)
    feat_set = set(feature_names)
    overlap = meta_cols_set & feat_set
    assert not overlap, \
        f"[{group_name}] Metadata columns in feature matrix: {overlap}"

    # All cleaned features exist in original set
    orig_set = set(orig_features)
    extras = feat_set - orig_set
    assert not extras, \
        f"[{group_name}] Feature names not in original dataset: {list(extras)[:5]}"

    # No NaNs
    assert not np.isnan(X_clean).any(), \
        f"[{group_name}] NaNs found in feature matrix after preprocessing"

    logger.info(f"[{group_name}] All sanity checks passed.")


# ── group pipeline ─────────────────────────────────────────────────────────

def process_group(group_name: str, group_data: dict, args, output_dir: Path,
                   logger) -> dict:
    """Run the full pipeline for one treatment group. Returns summary dict."""
    dirs = make_dirs(output_dir, group_name)

    X_orig = group_data["X"]
    y = group_data["y"]
    groups = group_data["groups"]
    feature_names_orig = group_data["feature_names"]
    n_orig = len(feature_names_orig)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing group: {group_name}")
    logger.info(f"{'='*60}")

    # ── 1. Preprocessing ───────────────────────────────────────────────────
    X_clean, feature_names = run_preprocessing(
        X_orig, feature_names_orig,
        args.variance_threshold, args.correlation_threshold,
        dirs["preprocessing"],
    )
    n_nzv_kept = len(feature_names)   # after NZV (before corr)
    n_corr_kept = len(feature_names)  # after corr (final)

    # Reconstruct intermediate count from saved files
    import pandas as pd
    try:
        nzv_df = pd.read_csv(dirs["preprocessing"] / "features_after_nzv.csv")
        n_nzv_kept = len(nzv_df)
    except Exception:
        pass

    run_sanity_checks(group_data, X_clean, feature_names, group_name, logger)

    C_grid = np.logspace(-3, 2, 30)

    # ── 2. Stability selection ─────────────────────────────────────────────
    per_l1_dfs, combined, conv_summary, chosen_C_map = run_stability_selection(
        X_clean, y, groups, feature_names,
        l1_ratio_grid=args.l1_ratios,
        C_grid=C_grid,
        B=args.B,
        subsample_fraction=args.subsample_fraction,
        coef_threshold=1e-8,
        random_state=args.random_state,
        output_dir=dirs["stability"],
        group_name=group_name,
        n_jobs=-1,
    )

    # Validate selection probabilities are in [0, 1]
    assert combined["mean_pi"].between(0, 1).all(), \
        f"[{group_name}] Selection probabilities out of [0, 1] range"

    # ── 3. Subject-capped sensitivity analysis ─────────────────────────────
    try:
        combined_capped = run_subject_capped(
            X_clean, y, groups, feature_names,
            l1_ratio_grid=args.l1_ratios,
            C_grid=C_grid,
            B=args.B,
            cap=args.subject_stage_cap,
            coef_threshold=1e-8,
            random_state=args.random_state,
            output_dir=dirs["subject_capped"],
            group_name=group_name,
            n_jobs=-1,
            chosen_C_map=chosen_C_map,
        )
        comparison = build_comparison(combined, combined_capped, dirs["subject_capped"])
    except Exception as e:
        logger.error(f"[{group_name}] Subject-capped analysis failed: {e}")
        combined_capped = None
        comparison = None

    # Verify cap constraint from comparison
    if comparison is not None:
        # Just log the flagged count as a proxy check
        n_flagged = comparison["pseudoreplication_flag"].sum()
        logger.info(
            f"[{group_name}] Pseudoreplication flags: {n_flagged}/{len(comparison)}"
        )

    # ── 4. Boruta (optional) ───────────────────────────────────────────────
    if getattr(args, "run_boruta", False):
        boruta_df = run_boruta(
            X_clean, y, feature_names,
            random_state=args.random_state,
            output_dir=dirs["boruta"],
            group_name=group_name,
        )
    else:
        import pandas as pd
        boruta_df = pd.DataFrame({
            "feature": feature_names,
            "boruta_decision": "Not run",
            "boruta_rank": float("nan"),
        })
        boruta_df.to_csv(dirs["boruta"] / "boruta_results.csv", index=False)

    # ── 5. Final candidate table ───────────────────────────────────────────
    final_df = build_final_table(
        combined, comparison, boruta_df,
        dirs["final"], group_name,
    )

    # ── 6. Plots ───────────────────────────────────────────────────────────
    try:
        make_all_plots(combined, per_l1_dfs, comparison, group_name, dirs["plots"])
    except Exception as e:
        logger.error(f"[{group_name}] Plotting failed: {e}")

    return {
        "final_df": final_df,
        "combined": combined,
        "comparison": comparison,
        "conv_summary": conv_summary,
        "n_orig": n_orig,
        "n_nzv": n_nzv_kept,
        "n_corr": n_corr_kept,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Treatment-specific feature selection for pig squeal acoustic features."
    )
    p.add_argument("--input", required=True,
                   help="Path to the Excel input file")
    p.add_argument("--sheet", default="Features_Data",
                   help="Sheet name (default: Features_Data)")
    p.add_argument("--output-dir", default="results",
                   help="Root output directory (default: results)")
    p.add_argument("--B", type=int, default=500,
                   help="Number of stability-selection replicates (default: 500)")
    p.add_argument("--l1-ratios", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9],
                   help="Elastic-net l1_ratio values (default: 0.3 0.5 0.7 0.9)")
    p.add_argument("--subsample-fraction", type=float, default=0.70,
                   help="Fraction of data per replicate (default: 0.70)")
    p.add_argument("--subject-stage-cap", type=int, default=15,
                   help="Max squeals per Subject×Stage cell in capped run (default: 15)")
    p.add_argument("--correlation-threshold", type=float, default=0.95,
                   help="Pearson r threshold for correlation filter (default: 0.95)")
    p.add_argument("--variance-threshold", type=float, default=1e-8,
                   help="Variance threshold for NZV filter (default: 1e-8)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--run-boruta", action="store_true",
                   help="Run optional Boruta secondary selector")
    p.add_argument("--groups", nargs="+", default=["S", "U", "B"],
                   help="Treatment groups to process (default: S U B)")
    return p.parse_args()


# ── entry point ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Feature selection pipeline starting")
    logger.info(f"Arguments: {vars(args)}")

    version_info = get_version_info()
    logger.info(f"Package versions: {version_info}")

    # ── Load and validate data ─────────────────────────────────────────────
    summary_dir = output_dir / "data_summary"
    df_full = load_and_validate(args.input, args.sheet, summary_dir)

    # Build summary count strings for the report
    import pandas as pd
    summary_counts = {
        "Counts by Treatment": (
            df_full.groupby("Treatment").size().rename("n_squeals").to_string()
        ),
        "Counts by Treatment × Stage": (
            df_full.groupby(["Treatment", "Stage"]).size().rename("n_squeals").to_string()
        ),
    }

    # ── Process each group ─────────────────────────────────────────────────
    group_results = {}
    conv_summaries = {}

    for group_name in args.groups:
        group_data = prepare_group(df_full, group_name)
        result = process_group(group_name, group_data, args, output_dir, logger)
        group_results[group_name] = result
        conv_summaries[group_name] = result["conv_summary"]

    # ── Report ─────────────────────────────────────────────────────────────
    write_report(
        output_dir, args, version_info, summary_counts, group_results, conv_summaries
    )

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Feature selection complete.")
    print()
    print("Main outputs:")
    print(f"  {output_dir / 'feature_selection_report.md'}")
    for group_name in args.groups:
        print(f"  {output_dir / group_name / 'final' / 'features_for_LMM.csv'}")
    print()
    print("Reminder:")
    print("  Feature selection was exploratory and treated squeals as independent.")
    print("  Use the selected features as candidates for downstream LMM analysis.")
    print("=" * 60)


if __name__ == "__main__":
    # Required for joblib loky backend on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
