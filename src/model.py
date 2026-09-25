"""
src/model.py
============
LightGBM baseline model for Amazon ML Challenge.

Responsibilities
----------------
- Parse ground truth and generate binary labels for candidate pairs.
- Ensure strict separation of ID columns from feature columns.
- Train LightGBM classifier with early stopping.
- Dynamically compute `scale_pos_weight` based on candidate class distribution.
- Expose model predictions and feature importance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import cfg, PipelineConfig
from src.blocking import (
    parse_ground_truth_matches,
    COL_S1_ID, COL_CAND_ID, COL_CAND_SRC
)
from src.features import get_feature_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label Generation
# ---------------------------------------------------------------------------

def build_candidate_labels(
    candidates_df: pd.DataFrame,
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> pd.Series:
    """
    Generate binary training labels (1 or 0) for each candidate pair.

    A candidate pair is positive (1) if and only if the `matched_entity_id`
    is present in the parsed ground-truth set for that `source1_entity_id`.

    Important:
      - This does NOT fabricate missing candidate rows.
      - If a true match is absent from `candidates_df`, it simply won't
        have a row here. Candidate recall limits maximum achievable model recall.
      - Preserves S2/S3 identity inherently because IDs are matched exactly
        as strings (including prefix).

    Parameters
    ----------
    candidates_df:
        DataFrame containing at least `source1_entity_id` and `matched_entity_id`.
    gt:
        Preprocessed ground-truth DataFrame.
    config:
        Pipeline configuration.

    Returns
    -------
    pd.Series
        Binary integer labels (0 or 1) aligned with the `candidates_df` index.
    """
    # Use the unified unverified ground-truth parser from blocking
    gt_dict = parse_ground_truth_matches(gt, config)

    labels = np.zeros(len(candidates_df), dtype=np.int32)

    # Convert columns to lists for faster iteration than iterrows
    s1_ids = candidates_df[COL_S1_ID].values
    cand_ids = candidates_df[COL_CAND_ID].values

    for i in range(len(candidates_df)):
        s1 = s1_ids[i]
        cand = cand_ids[i]
        
        # Check if the candidate ID exists in the ground truth set for this S1 entity
        true_matches = gt_dict.get(s1, set())
        if cand in true_matches:
            labels[i] = 1

    return pd.Series(labels, index=candidates_df.index, name="label")


# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------

class EntityMatcher:
    """
    LightGBM binary classifier wrapper for entity resolution.
    
    Provides a clean API to train the model on generated features and
    predict match probabilities for validation/inference.
    """

    def __init__(self, config: PipelineConfig = cfg):
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_cols = get_feature_columns()

    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Verify that all expected features exist and are numeric.
        """
        missing_cols = set(self.feature_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing expected feature columns: {missing_cols}")

        # Check for forbidden columns
        forbidden = {COL_S1_ID, COL_CAND_ID, COL_CAND_SRC}
        overlap = forbidden.intersection(set(self.feature_cols))
        if overlap:
            raise ValueError(f"CRITICAL LEAKAGE: ID columns in feature list: {overlap}")

        # Ensure all features are numeric
        for col in self.feature_cols:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise TypeError(f"Feature column '{col}' is not numeric. Found: {X[col].dtype}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the LightGBM model.

        Calculates dynamic `scale_pos_weight` based on class imbalance.
        Uses early stopping if validation data is provided.

        Parameters
        ----------
        X_train, X_val:
            Candidate feature DataFrames. Must contain all columns returned
            by `get_feature_columns()`. Identity columns (if present) are ignored.
        y_train, y_val:
            Binary label series aligned with X_train/X_val.
        """
        self._validate_features(X_train)
        
        # Extract purely the feature matrix
        train_features = X_train[self.feature_cols]
        
        # Calculate class imbalance weighting dynamically
        num_pos = int(y_train.sum())
        num_neg = len(y_train) - num_pos
        
        logger.info(f"Training on {len(y_train)} candidates ({num_pos} pos, {num_neg} neg)")
        
        if num_pos == 0:
            logger.warning("Zero positive examples in training set!")
            scale_pos_weight = 1.0
        elif num_neg == 0:
            logger.warning("Zero negative examples in training set!")
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = float(num_neg) / float(num_pos)
            
        logger.info(f"Dynamic scale_pos_weight: {scale_pos_weight:.2f}")

        # Deep copy config params to avoid mutating the class default
        params = dict(self.config.LGBM_PARAMS)
        params["scale_pos_weight"] = scale_pos_weight

        # Prepare LightGBM Datasets
        dtrain = lgb.Dataset(train_features, label=y_train)
        valid_sets = [dtrain]
        valid_names = ["train"]

        callbacks = []

        if X_val is not None and y_val is not None:
            self._validate_features(X_val)
            val_features = X_val[self.feature_cols]
            dval = lgb.Dataset(val_features, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")
            
            # Use early stopping callback
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.LGBM_EARLY_STOPPING_ROUNDS,
                    first_metric_only=False,
                    verbose=False
                )
            )
            callbacks.append(lgb.log_evaluation(period=50))
        else:
            logger.warning("No validation data provided to fit(). Early stopping disabled.")

        # Train
        logger.info("Starting LightGBM training...")
        self.model = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        best_iter = self.model.best_iteration
        logger.info(f"Training completed. Best iteration: {best_iter}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict matching probabilities for candidates.

        Parameters
        ----------
        X:
            Candidate feature DataFrame.

        Returns
        -------
        np.ndarray
            1D array of probabilities (shape: [n_samples]),
            representing P(candidate is a true match).
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
            
        self._validate_features(X)
        features = X[self.feature_cols]
        
        # Predict positive class probabilities
        probs = self.model.predict(features)
        
        # Ensure probs are strictly bounded [0, 1]
        probs = np.clip(probs, 0.0, 1.0)
        return probs

    def feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from the fitted LightGBM model.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['feature', 'importance_gain', 'importance_split'],
            sorted descending by gain.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
            
        split = self.model.feature_importance(importance_type="split")
        gain = self.model.feature_importance(importance_type="gain")
        
        df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance_split": split,
            "importance_gain": gain,
        })
        
        return df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
