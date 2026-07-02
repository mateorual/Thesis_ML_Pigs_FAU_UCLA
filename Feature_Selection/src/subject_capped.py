"""
Subject-stage-capped sensitivity analysis.

For each replicate, samples up to `cap` squeals per Subject×Class cell,
then runs the same multinomial elastic-net stability selection pipeline.
Features whose stability drops strongly (delta_pi > 0.20) after capping
are flagged as potentially subject-dominated.

C selection: the same global pre-selected C used in the main stability run
is reused here (or re-selected once per l1_ratio on the full dataset).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from .stability_selection import (
    select_C_global,
    _run_one_replicate_capped,
    _aggregate_replicates,
    _create_combined_table,
)

logger = logging.getLogger("feature_selection")


def run_subject_capped(X, y, groups, feature_names, l1_ratio_grid, C_grid,
                        B, cap, coef_threshold, random_state,
                        output_dir: Path, group_name: str, n_jobs: int = -1,
                        chosen_C_map: dict = None):
    """
    Run subject-capped stability selection for all l1_ratios.
    Returns combined_capped_df.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_l1_dfs = {}

    for l1_ratio in l1_ratio_grid:
        logger.info(
            f"[{group_name}] Subject-capped (cap={cap}): l1_ratio={l1_ratio}, B={B}"
        )

        # Reuse pre-selected C from main run if available, else re-select
        if chosen_C_map and l1_ratio in chosen_C_map:
            chosen_C = chosen_C_map[l1_ratio]
            logger.info(
                f"[{group_name}] Capped l1_ratio={l1_ratio}: reusing C={chosen_C:.4f} from main run"
            )
        else:
            chosen_C = select_C_global(X, y, C_grid, l1_ratio, random_state)
            logger.info(
                f"[{group_name}] Capped l1_ratio={l1_ratio}: using C={chosen_C:.4f}"
            )

        rep_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_one_replicate_capped)(
                b, X, y, groups, cap, l1_ratio, chosen_C,
                coef_threshold, random_state + b * 997 + 50000
            )
            for b in tqdm(
                range(B), desc=f"  [{group_name}] capped l1={l1_ratio}", leave=True
            )
        )

        valid = [r for r in rep_results if r is not None]
        n_skipped = B - len(valid)
        if n_skipped > 0:
            logger.warning(
                f"[{group_name}] Capped l1_ratio={l1_ratio}: "
                f"{n_skipped}/{B} replicates skipped (insufficient samples)"
            )

        if not valid:
            logger.error(
                f"[{group_name}] All capped replicates failed for l1_ratio={l1_ratio}"
            )
            continue

        df_scores, classes, total_conv, B_actual, _, _ = \
            _aggregate_replicates(valid, feature_names)
        df_scores["l1_ratio"] = l1_ratio
        per_l1_dfs[l1_ratio] = df_scores

        if total_conv > 0:
            logger.warning(
                f"[{group_name}] Capped l1_ratio={l1_ratio}: "
                f"{total_conv} convergence warnings"
            )

        df_scores.to_csv(
            output_dir / f"capped_stability_scores_l1ratio_{l1_ratio}.csv", index=False
        )
        logger.info(
            f"[{group_name}] Capped l1_ratio={l1_ratio} done. "
            f"Valid replicates: {B_actual}/{B}"
        )

    if not per_l1_dfs:
        raise RuntimeError(f"[{group_name}] Subject-capped analysis produced no results.")

    combined = _create_combined_table(per_l1_dfs, feature_names)
    combined.to_csv(
        output_dir / "stability_scores_combined_subject_capped.csv", index=False
    )
    return combined


def build_comparison(combined_uncapped: pd.DataFrame,
                      combined_capped: pd.DataFrame,
                      output_dir: Path) -> pd.DataFrame:
    """
    Compare uncapped vs subject-capped stability scores.
    Flags features where mean_pi drops by more than 0.20 after capping.
    """
    unc = combined_uncapped[["feature", "mean_pi"]].rename(
        columns={"mean_pi": "mean_pi_uncapped"}
    )
    cap = combined_capped[["feature", "mean_pi"]].rename(
        columns={"mean_pi": "mean_pi_subject_capped"}
    )
    comp = unc.merge(cap, on="feature", how="outer").fillna(0.0)
    comp["delta_pi"] = comp["mean_pi_uncapped"] - comp["mean_pi_subject_capped"]
    comp["pseudoreplication_flag"] = comp["delta_pi"] > 0.20

    def _interp(row):
        if not row["pseudoreplication_flag"]:
            return "Stable after subject-stage capping"
        elif row["delta_pi"] > 0.35:
            return "Potential subject-dominated / exploratory only"
        else:
            return "Reduced after capping"

    comp["interpretation"] = comp.apply(_interp, axis=1)
    comp = comp.sort_values("mean_pi_uncapped", ascending=False).reset_index(drop=True)
    comp.to_csv(output_dir / "uncapped_vs_subject_capped_comparison.csv", index=False)
    return comp
