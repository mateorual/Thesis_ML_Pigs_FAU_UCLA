"""Load and validate the pig squeal acoustic dataset."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger("feature_selection")

METADATA_COLS = ["Squeal", "Subject", "Treatment", "Week", "Num_Cycles", "Stage"]
VALID_TREATMENTS = {"S", "U", "B"}
VALID_STAGES = {"Pre", "Early Post", "Mid Post", "Late Post"}
ANALYSIS_STAGES = {"Pre", "Early Post", "Late Post"}

STAGE_LABELS = {
    "S": {"Pre": "Pre_S", "Early Post": "Early_S", "Late Post": "Late_S"},
    "U": {"Pre": "Pre_U", "Early Post": "Early_U", "Late Post": "Late_U"},
    "B": {"Pre": "Pre_B", "Early Post": "Early_B", "Late Post": "Late_B"},
}


def load_and_validate(excel_path: str, sheet_name: str, summary_dir: Path) -> pd.DataFrame:
    """
    Load the Excel file, validate schema and content, save summary tables.
    Returns the full validated DataFrame.
    """
    logger.info(f"Loading data from: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # --- Validate metadata columns ---
    actual_meta = list(df.columns[:6])
    if actual_meta != METADATA_COLS:
        raise ValueError(
            f"Metadata columns mismatch.\n  Expected: {METADATA_COLS}\n  Got: {actual_meta}"
        )

    # --- Validate Treatment and Stage ---
    bad_treatment = set(df["Treatment"].unique()) - VALID_TREATMENTS
    if bad_treatment:
        raise ValueError(f"Unexpected Treatment values: {bad_treatment}")

    bad_stage = set(df["Stage"].unique()) - VALID_STAGES
    if bad_stage:
        raise ValueError(f"Unexpected Stage values: {bad_stage}")

    # --- Validate feature columns are numeric, no NaNs ---
    feature_cols = list(df.columns[6:])
    logger.info(f"Feature columns: {len(feature_cols)}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols):
        non_num = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(f"Non-numeric feature columns: {non_num[:5]} ...")

    # ── Handle Schlegel21 NaN values and zero-dominated columns ──────────────
    # NaN in Schlegel21 means the parameter was uncomputable (e.g. no voiced
    # frames for HNR, DC-dominated spectrum for Q50). These are treated as 0,
    # NOT median-imputed. Columns where >50 % of values are 0 are dropped as
    # they carry no discriminative information.
    ZERO_FRACTION_THRESHOLD = 0.50
    schlegel_cols = [c for c in feature_cols if c.startswith("Schlegel21_")]
    if schlegel_cols:
        # Fill NaN → 0 for Schlegel21 columns only
        nan_counts = df[schlegel_cols].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
            logger.warning(
                f"Schlegel21 NaN values filled with 0 (uncomputable parameter): "
                + ", ".join(f"{c}({v})" for c, v in nan_cols.items())
            )
            df[schlegel_cols] = df[schlegel_cols].fillna(0.0)

        # Drop Schlegel21 columns dominated by zeros
        zero_frac = (df[schlegel_cols] == 0.0).mean()
        drop_cols = zero_frac[zero_frac > ZERO_FRACTION_THRESHOLD].index.tolist()
        if drop_cols:
            logger.warning(
                f"Dropping {len(drop_cols)} Schlegel21 column(s) with "
                f">{ZERO_FRACTION_THRESHOLD*100:.0f}% zero values (likely noise): "
                + ", ".join(f"{c}({zero_frac[c]*100:.1f}%)" for c in drop_cols)
            )
            df = df.drop(columns=drop_cols)
            feature_cols = [c for c in feature_cols if c not in drop_cols]

    # ── Check no other NaN values remain ─────────────────────────────────────
    n_missing = df[feature_cols].isnull().sum().sum()
    if n_missing > 0:
        missing_by_col = df[feature_cols].isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        raise ValueError(
            f"Found {n_missing} missing values in non-Schlegel21 feature columns "
            f"(no imputation applied). Affected columns:\n"
            + "\n".join(f"  {c}: {v}" for c, v in missing_cols.items())
        )

    logger.info("Data validation passed.")

    # --- Save summary tables ---
    summary_dir.mkdir(parents=True, exist_ok=True)

    counts_treatment = df.groupby("Treatment").size().rename("n_squeals")
    counts_treatment.to_csv(summary_dir / "counts_by_treatment.csv")
    logger.info(f"\nCounts by Treatment:\n{counts_treatment.to_string()}")

    counts_ts = df.groupby(["Treatment", "Stage"]).size().rename("n_squeals")
    counts_ts.to_csv(summary_dir / "counts_by_treatment_stage.csv")
    logger.info(f"\nCounts by Treatment × Stage:\n{counts_ts.to_string()}")

    counts_sub = df.groupby(["Subject", "Treatment", "Stage"]).size().rename("n_squeals")
    counts_sub.to_csv(summary_dir / "counts_by_subject_treatment_stage.csv")
    logger.info(f"\nCounts by Subject × Treatment × Stage:\n{counts_sub.to_string()}")

    return df


def prepare_group(df: pd.DataFrame, treatment: str) -> dict:
    """
    Filter to one treatment group, remove Mid Post, relabel stages.
    Returns dict with keys: metadata, X, y, groups, feature_names, class_names.
    """
    mask = (df["Treatment"] == treatment) & (df["Stage"].isin(ANALYSIS_STAGES))
    sub = df[mask].copy()

    stage_map = STAGE_LABELS[treatment]
    sub["class_label"] = sub["Stage"].map(stage_map)

    feature_names = list(df.columns[6:])
    X = sub[feature_names].values.astype(np.float64)
    y = sub["class_label"].values
    groups = sub["Subject"].values
    metadata = sub[METADATA_COLS + ["class_label"]].reset_index(drop=True)

    class_names = sorted(sub["class_label"].unique().tolist())

    logger.info(
        f"Group {treatment}: {len(sub)} squeals, {len(feature_names)} features, "
        f"classes={class_names}"
    )
    _log_class_counts(y, class_names)

    return {
        "metadata": metadata,
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": feature_names,
        "class_names": class_names,
    }


def _log_class_counts(y, class_names):
    for cls in class_names:
        n = (y == cls).sum()
        logger.info(f"  {cls}: {n} squeals")
