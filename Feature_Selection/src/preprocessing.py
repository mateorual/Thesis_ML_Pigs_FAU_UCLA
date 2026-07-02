"""
Unsupervised preprocessing: near-zero variance and high-correlation filtering.

Both filters use class-label-free criteria so they do not introduce label leakage
in the exploratory feature-selection stage.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("feature_selection")


def apply_nzv_filter(X: np.ndarray, feature_names: list,
                      variance_threshold: float, output_dir: Path) -> tuple:
    """
    Remove features with variance <= variance_threshold.
    Returns (X_filtered, kept_feature_names).
    """
    variances = X.var(axis=0)
    mask_keep = variances > variance_threshold
    removed = [f for f, keep in zip(feature_names, mask_keep) if not keep]
    kept = [f for f, keep in zip(feature_names, mask_keep) if keep]

    logger.info(f"NZV filter (threshold={variance_threshold}): "
                f"removed {len(removed)}, kept {len(kept)}")

    pd.DataFrame({"feature": removed, "variance": variances[~mask_keep]}).to_csv(
        output_dir / "nzv_removed_features.csv", index=False
    )
    pd.DataFrame({"feature": kept, "variance": variances[mask_keep]}).to_csv(
        output_dir / "features_after_nzv.csv", index=False
    )

    return X[:, mask_keep], kept


def apply_correlation_filter(X: np.ndarray, feature_names: list,
                               corr_threshold: float, output_dir: Path) -> tuple:
    """
    Remove one feature from each highly correlated pair (|r| > corr_threshold).
    Keeps the feature with larger variance; ties broken lexicographically.
    Returns (X_filtered, kept_feature_names).
    """
    n_features = X.shape[1]
    variances = X.var(axis=0)

    # Compute correlation matrix once
    logger.info(f"Computing correlation matrix for {n_features} features...")
    corr = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr, 0.0)  # ignore self-correlation

    to_drop = set()
    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > corr_threshold:
                # Keep the one with larger variance; tie → keep lexicographically earlier name
                if variances[i] >= variances[j]:
                    to_drop.add(j)
                elif variances[i] < variances[j]:
                    to_drop.add(i)
                    break  # i is dropped; skip remaining j for this i

    keep_mask = np.array([i not in to_drop for i in range(n_features)])
    removed = [feature_names[i] for i in sorted(to_drop)]
    kept = [feature_names[i] for i in range(n_features) if i not in to_drop]

    logger.info(f"Correlation filter (threshold={corr_threshold}): "
                f"removed {len(removed)}, kept {len(kept)}")

    pd.DataFrame({"feature": removed}).to_csv(
        output_dir / "correlation_removed_features.csv", index=False
    )
    pd.DataFrame({"feature": kept, "variance": variances[keep_mask]}).to_csv(
        output_dir / "features_after_correlation_filter.csv", index=False
    )

    return X[:, keep_mask], kept


def run_preprocessing(X: np.ndarray, feature_names: list,
                       variance_threshold: float, corr_threshold: float,
                       output_dir: Path) -> tuple:
    """Run NZV then correlation filter. Returns (X_clean, feature_names_clean)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X1, names1 = apply_nzv_filter(X, feature_names, variance_threshold, output_dir)
    X2, names2 = apply_correlation_filter(X1, names1, corr_threshold, output_dir)

    # Sanity check
    assert not np.isnan(X2).any(), "NaNs found after preprocessing"
    logger.info(f"Preprocessing complete: {len(feature_names)} -> {len(names2)} features")

    return X2, names2
