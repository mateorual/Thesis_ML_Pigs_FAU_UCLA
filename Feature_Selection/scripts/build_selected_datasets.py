"""Build selected-feature datasets for downstream LMM analysis.

The script takes a consolidated feature matrix produced by the feature-extraction
pipeline and keeps:

1. the metadata columns, and
2. selected acoustic features from a union table.

Two datasets are produced by default:

- Path 1: S + U + B selected-feature union
- Path 2: S + U selected-feature union
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_COLS = ["Squeal", "Subject", "Treatment", "Week", "Num_Cycles", "Stage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create selected-feature Excel datasets for Path 1 and Path 2."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Consolidated feature matrix Excel file from feature extraction.",
    )
    parser.add_argument(
        "--sheet",
        default="Features_Data",
        help="Input Excel sheet name. Default: Features_Data.",
    )
    parser.add_argument(
        "--summary-dir",
        default="outputs/selected_feature_summaries",
        help="Folder containing union_S_U_B.csv and union_S_U.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/selected_feature_datasets",
        help="Output folder for the selected-feature Excel datasets.",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="+",
        default=DEFAULT_METADATA_COLS,
        help="Metadata columns to retain before acoustic features.",
    )
    return parser.parse_args()


def load_union_features(path: Path) -> list[str]:
    union_df = pd.read_csv(path)
    if "feature" not in union_df.columns:
        raise ValueError(f"{path} must contain a 'feature' column.")
    return union_df["feature"].dropna().astype(str).tolist()


def write_selected_dataset(
    full_df: pd.DataFrame,
    features: list[str],
    metadata_cols: list[str],
    output_path: Path,
) -> None:
    missing_meta = [col for col in metadata_cols if col not in full_df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns in input data: {missing_meta}")

    missing_features = [feature for feature in features if feature not in full_df.columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(
            f"{len(missing_features)} selected features were not found in the input data. "
            f"Examples: {preview}"
        )

    selected_df = full_df[metadata_cols + features].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_excel(output_path, sheet_name="Features_Data", index=False)
    print(
        f"Wrote {output_path.name}: {selected_df.shape[0]} rows, "
        f"{len(features)} selected features."
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    summary_dir = (ROOT / args.summary_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()

    full_df = pd.read_excel(input_path, sheet_name=args.sheet, engine="openpyxl")

    path1_features = load_union_features(summary_dir / "union_S_U_B.csv")
    path2_features = load_union_features(summary_dir / "union_S_U.csv")

    write_selected_dataset(
        full_df,
        path1_features,
        args.metadata_cols,
        output_dir / "Selected_Features_Groups_S_U_B_v1.xlsx",
    )
    write_selected_dataset(
        full_df,
        path2_features,
        args.metadata_cols,
        output_dir / "Selected_Features_Groups_S_U_v1.xlsx",
    )


if __name__ == "__main__":
    main()
