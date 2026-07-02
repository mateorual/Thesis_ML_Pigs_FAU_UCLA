"""
Optional Boruta feature selection using BorutaPy.

Boruta is run as a complementary nonlinear/all-relevant selector.
Agreement between Boruta and elastic-net stability selection provides
additional support; Boruta rejection is NOT used to veto elastic-net results.

Note: random-forest importance can be difficult to interpret when acoustic
predictors are strongly correlated, which is common in this dataset.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("feature_selection")


def run_boruta(X: np.ndarray, y: np.ndarray, feature_names: list,
               random_state: int, output_dir: Path, group_name: str,
               n_estimators: int = 2000, max_iter: int = 200,
               perc: int = 90) -> pd.DataFrame:
    """
    Run BorutaPy feature selection. Returns a DataFrame with boruta decisions.
    If BorutaPy is not installed, returns a DataFrame with 'Not run' for all features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    not_run_df = pd.DataFrame({
        "feature": feature_names,
        "boruta_decision": "Not run",
        "boruta_rank": np.nan,
    })

    try:
        from boruta import BorutaPy
    except ImportError:
        logger.info(f"[{group_name}] BorutaPy not installed — skipping Boruta.")
        not_run_df.to_csv(output_dir / "boruta_results.csv", index=False)
        return not_run_df

    try:
        from sklearn.ensemble import RandomForestClassifier

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        selector = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            max_iter=max_iter,
            perc=perc,
            random_state=random_state,
            verbose=0,
        )

        logger.info(f"[{group_name}] Running Boruta (max_iter={max_iter}, perc={perc})...")
        selector.fit(X, y_enc)

        decision = []
        for sup, tent in zip(selector.support_, selector.support_weak_):
            if sup:
                decision.append("Confirmed")
            elif tent:
                decision.append("Tentative")
            else:
                decision.append("Rejected")

        df = pd.DataFrame({
            "feature": feature_names,
            "boruta_decision": decision,
            "boruta_rank": selector.ranking_,
        })
        df.to_csv(output_dir / "boruta_results.csv", index=False)

        n_conf = (df["boruta_decision"] == "Confirmed").sum()
        n_tent = (df["boruta_decision"] == "Tentative").sum()
        logger.info(
            f"[{group_name}] Boruta complete: {n_conf} confirmed, {n_tent} tentative, "
            f"{len(feature_names) - n_conf - n_tent} rejected"
        )
        return df

    except Exception as e:
        logger.error(f"[{group_name}] Boruta failed: {e}")
        not_run_df.to_csv(output_dir / "boruta_results.csv", index=False)
        return not_run_df
