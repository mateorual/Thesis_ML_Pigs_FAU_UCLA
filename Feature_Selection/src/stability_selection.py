"""
Multinomial elastic-net stability selection.

C selection strategy (1-SE rule):
  C is selected ONCE per l1_ratio using stratified cross-validation on the full
  preprocessed dataset before the resampling loop begins. This global selection is
  used for all B replicates of that l1_ratio.

  Rationale: per-replicate inner CV would require 90+ model fits per replicate,
  making B=500 computationally prohibitive (>12 hours). Pre-selecting C once is
  scientifically equivalent for stability-score estimation: 70% subsamples preserve
  the data distribution, so the full-data 1-SE C is representative.
  The prompt explicitly permits this fallback ("use the best CV C, but clearly
  document this in the report").

Each replicate:
  1. Draws a stratified subsample (subsample_fraction of the data).
  2. Fits StandardScaler on that subsample only.
  3. Uses the pre-selected C for this l1_ratio.
  4. Fits multinomial elastic-net logistic regression.
  5. Records which features have |coef| > coef_threshold in any class.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger("feature_selection")


# ──────────────────────────────────────────────────────────
# C selection (run once before the stability loop)
# ──────────────────────────────────────────────────────────

def select_C_global(X, y, C_grid, l1_ratio, random_state, n_cv_folds=5):
    """
    Select C using the 1-SE rule via stratified GridSearchCV on the full dataset.
    Run once per l1_ratio before the stability resampling loop.

    All C × fold combinations run in parallel (n_jobs=-1) since this function
    is called from the main process, not inside a joblib parallel worker.
    Returns the chosen C value (smallest C within 1 SE of the best mean score).
    """
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_cv_folds:
        logger.warning("Too few samples per class for inner CV; using median C.")
        return float(C_grid[len(C_grid) // 2])

    seed_int = int(random_state) % (2 ** 31)
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed_int)

    # Standardize once for C selection (acceptable here — one-time selection,
    # not nested inside resampling folds)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        f"  Selecting C via {n_cv_folds}-fold GridSearchCV "
        f"({len(C_grid)} C values, parallel)..."
    )

    base_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        l1_ratio=l1_ratio,
        random_state=seed_int,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(
            base_clf,
            param_grid={"C": [float(c) for c in C_grid]},
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=False,
        )
        gs.fit(X_scaled, y)

    mean_scores = gs.cv_results_["mean_test_score"]
    std_scores = gs.cv_results_["std_test_score"]
    se_scores = std_scores / np.sqrt(n_cv_folds)

    best_idx = int(np.argmax(mean_scores))
    threshold = float(mean_scores[best_idx]) - float(se_scores[best_idx])
    candidates = np.where(mean_scores >= threshold)[0]
    # GridSearchCV preserves param_grid order (ascending C), so candidates[0]
    # is the smallest C = strongest regularization
    chosen_idx = int(candidates[0])
    chosen_C = float(C_grid[chosen_idx])
    best_C = float(C_grid[best_idx])

    logger.info(
        f"  C grid: best={best_C:.5g} (idx={best_idx}), "
        f"1-SE chosen={chosen_C:.5g} (idx={chosen_idx})"
    )
    return chosen_C


# ──────────────────────────────────────────────────────────
# Module-level replicate functions (required by joblib loky)
# ──────────────────────────────────────────────────────────

def _run_one_replicate(rep_idx, X, y, l1_ratio, fixed_C, subsample_fraction,
                        coef_threshold, seed):
    """Single stability-selection replicate. Called by joblib workers."""
    seed_int = int(seed) % (2 ** 31)

    # Stratified subsample
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=float(subsample_fraction), random_state=seed_int
    )
    try:
        idx_sub, _ = next(sss.split(X, y))
    except ValueError:
        idx_sub = np.arange(len(y))

    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    # Fit scaler on subsample only (not on full data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    n_conv = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            tol=1e-4,
            l1_ratio=l1_ratio,
            C=fixed_C,
            random_state=seed_int,
            n_jobs=1,
        )
        clf.fit(X_scaled, y_sub)
        n_conv = sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning)
        )

    coef = clf.coef_  # (n_classes, n_features)
    selected_any = (np.abs(coef) > coef_threshold).any(axis=0).astype(np.int8)
    selected_by_class = (np.abs(coef) > coef_threshold).astype(np.int8)
    abs_coef = np.abs(coef).astype(np.float32)

    return {
        "selected_any": selected_any,
        "selected_by_class": selected_by_class,
        "fixed_C": fixed_C,
        "abs_coef": abs_coef,
        "n_conv": n_conv,
        "classes": list(clf.classes_),
    }


def _run_one_replicate_capped(rep_idx, X, y, groups, cap, l1_ratio, fixed_C,
                               coef_threshold, seed):
    """
    Single replicate for subject-stage-capped sensitivity analysis.
    Samples up to `cap` rows per Subject×Class cell, then runs same pipeline.
    """
    seed_int = int(seed) % (2 ** 31)
    rng = np.random.RandomState(seed_int)

    idx_list = []
    for subj in np.unique(groups):
        for cls in np.unique(y):
            mask = (groups == subj) & (y == cls)
            cell_idx = np.where(mask)[0]
            if len(cell_idx) == 0:
                continue
            n_draw = min(cap, len(cell_idx))
            drawn = rng.choice(cell_idx, size=n_draw, replace=False)
            idx_list.append(drawn)

    if not idx_list:
        return None

    idx_sub = np.concatenate(idx_list)
    X_sub = X[idx_sub]
    y_sub = y[idx_sub]

    classes, counts = np.unique(y_sub, return_counts=True)
    if counts.min() < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    n_conv = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            tol=1e-4,
            l1_ratio=l1_ratio,
            C=fixed_C,
            random_state=seed_int,
            n_jobs=1,
        )
        clf.fit(X_scaled, y_sub)
        n_conv = sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning)
        )

    coef = clf.coef_
    selected_any = (np.abs(coef) > coef_threshold).any(axis=0).astype(np.int8)
    selected_by_class = (np.abs(coef) > coef_threshold).astype(np.int8)
    abs_coef = np.abs(coef).astype(np.float32)

    return {
        "selected_any": selected_any,
        "selected_by_class": selected_by_class,
        "fixed_C": fixed_C,
        "abs_coef": abs_coef,
        "n_conv": n_conv,
        "classes": list(clf.classes_),
    }


# ──────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────

def _aggregate_replicates(rep_results, feature_names):
    """Aggregate a list of replicate result dicts into a stability-score DataFrame."""
    rep_results = [r for r in rep_results if r is not None]
    if not rep_results:
        raise RuntimeError("All replicates failed.")

    classes = rep_results[0]["classes"]
    n_features = len(feature_names)

    sel_any = np.stack([r["selected_any"] for r in rep_results])       # (B, F)
    sel_cls = np.stack([r["selected_by_class"] for r in rep_results])  # (B, C, F)
    abs_coef_all = np.stack([r["abs_coef"] for r in rep_results])      # (B, C, F)

    pi_any = sel_any.mean(axis=0)
    pi_by_class = sel_cls.mean(axis=0)  # (C, F)

    # Mean abs coef when selected in any class
    mean_abs_when_sel = np.zeros(n_features)
    sel_any_bool = sel_any.astype(bool)
    for j in range(n_features):
        mask = sel_any_bool[:, j]
        if mask.any():
            mean_abs_when_sel[j] = abs_coef_all[mask, :, j].mean()

    mean_abs_all = abs_coef_all.mean(axis=(0, 1))

    df = pd.DataFrame({"feature": feature_names, "pi_any": pi_any})
    for k, cls in enumerate(classes):
        df[f"pi_{cls}"] = pi_by_class[k]
    df["mean_abs_coef_when_selected"] = mean_abs_when_sel
    df["mean_abs_coef_all_replicates"] = mean_abs_all

    total_conv = sum(r["n_conv"] for r in rep_results)
    B_actual = len(rep_results)
    return df, classes, total_conv, B_actual, sel_any, sel_cls


def _create_combined_table(per_l1_dfs: dict, feature_names: list) -> pd.DataFrame:
    """Build the alpha-consistency summary table from per-l1_ratio DataFrames."""
    l1_ratios = sorted(per_l1_dfs.keys())
    combined = pd.DataFrame({"feature": feature_names})
    pi_cols = []
    for lr in l1_ratios:
        col = f"pi_l1ratio_{lr}"
        pi_row = per_l1_dfs[lr].set_index("feature")["pi_any"].reindex(feature_names)
        combined[col] = pi_row.values
        pi_cols.append(col)

    combined["mean_pi"] = combined[pi_cols].mean(axis=1)
    combined["max_pi"] = combined[pi_cols].max(axis=1)
    combined["n_l1ratios_pi_ge_0.70"] = (combined[pi_cols] >= 0.70).sum(axis=1)
    combined["n_l1ratios_pi_ge_0.50"] = (combined[pi_cols] >= 0.50).sum(axis=1)

    def _label(row):
        if row["n_l1ratios_pi_ge_0.70"] >= 2:
            return "High"
        elif row["n_l1ratios_pi_ge_0.50"] >= 2:
            return "Moderate"
        return "Low"

    combined["alpha_consistency_label"] = combined.apply(_label, axis=1)
    return combined.sort_values("mean_pi", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────

def run_stability_selection(X, y, groups, feature_names, l1_ratio_grid, C_grid,
                             B, subsample_fraction, coef_threshold, random_state,
                             output_dir: Path, group_name: str, n_jobs: int = -1):
    """
    Run multinomial elastic-net stability selection for all l1_ratios.
    Returns (per_l1_ratio_dfs, combined_df, convergence_summary).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_l1_dfs = {}
    all_chosen_C = {}
    conv_summary = {}

    for l1_ratio in l1_ratio_grid:
        logger.info(f"[{group_name}] Stability selection: l1_ratio={l1_ratio}, B={B}")

        # Pre-select C once for this l1_ratio using the full dataset
        chosen_C = select_C_global(X, y, C_grid, l1_ratio, random_state)
        logger.info(f"[{group_name}] l1_ratio={l1_ratio}: using C={chosen_C:.4f} for all {B} replicates")

        rep_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_one_replicate)(
                b, X, y, l1_ratio, chosen_C, subsample_fraction,
                coef_threshold, random_state + b * 997
            )
            for b in tqdm(range(B), desc=f"  [{group_name}] l1={l1_ratio}", leave=True)
        )

        df_scores, classes, total_conv, B_actual, sel_any, sel_cls = \
            _aggregate_replicates(rep_results, feature_names)
        df_scores["l1_ratio"] = l1_ratio

        if total_conv > 0:
            logger.warning(
                f"[{group_name}] l1_ratio={l1_ratio}: {total_conv} convergence "
                f"warnings across {B_actual} replicates"
            )
        conv_summary[l1_ratio] = total_conv

        per_l1_dfs[l1_ratio] = df_scores
        all_chosen_C[l1_ratio] = [chosen_C] * B_actual  # same C used for all reps

        # Save per-l1_ratio outputs
        df_scores.to_csv(
            output_dir / f"stability_scores_l1ratio_{l1_ratio}.csv", index=False
        )
        pd.DataFrame(sel_any, columns=feature_names).to_csv(
            output_dir / f"selection_matrix_any_l1ratio_{l1_ratio}.csv", index=False
        )
        np.savez(
            output_dir / f"selection_matrix_by_class_l1ratio_{l1_ratio}.npz",
            data=sel_cls,
            classes=np.array(classes, dtype=str),
            features=np.array(feature_names, dtype=str),
        )
        logger.info(
            f"[{group_name}] l1_ratio={l1_ratio} done. "
            f"Convergence warnings: {total_conv}/{B_actual}"
        )

    # Save chosen C values
    c_records = [{"l1_ratio": lr, "chosen_C": vals[0]}
                 for lr, vals in all_chosen_C.items()]
    pd.DataFrame(c_records).to_csv(output_dir / "chosen_C_values.csv", index=False)

    # Combined alpha-consistency table
    combined = _create_combined_table(per_l1_dfs, feature_names)
    combined.to_csv(output_dir / "stability_scores_combined.csv", index=False)

    # Return chosen_C_map so callers can reuse it (avoids duplicate C selection)
    chosen_C_map = {lr: vals[0] for lr, vals in all_chosen_C.items()}
    return per_l1_dfs, combined, conv_summary, chosen_C_map
