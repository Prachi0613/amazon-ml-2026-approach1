"""
src/evaluation.py
=================
Evaluation, threshold tuning, and validation splitting for the Amazon ML Challenge.

Responsibilities
----------------
- Calculate per-entity F0.5 score (macro-averaged over S1 entities).
- Handle zero-match (singleton) entities correctly.
- Perform deterministic entity-level validation splitting.
- Tune probability threshold to maximize macro F0.5.

Metrics
-------
The target metric is Macro F0.5.
For each S1 entity:
  TP = |True Matches ∩ Predicted Matches|
  FP = |Predicted Matches - True Matches|
  FN = |True Matches - Predicted Matches|

  If True = 0 and Predicted = 0: F0.5 = 1.0
  If True = 0 and Predicted > 0: F0.5 = 0.0
  If True > 0 and Predicted = 0: F0.5 = 0.0
  Otherwise:
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Macro F0.5 = Mean(F0.5 over all S1 entities)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import cfg, PipelineConfig
from src.blocking import parse_ground_truth_matches

logger = logging.getLogger(__name__)

# Constants for diagnostic DataFrame columns
COL_S1_ID = "source1_entity_id"
COL_TRUE_COUNT = "true_count"
COL_PRED_COUNT = "predicted_count"
COL_TP = "true_positive_count"
COL_FP = "false_positive_count"
COL_FN = "false_negative_count"
COL_PRECISION = "precision"
COL_RECALL = "recall"
COL_F05 = "f0_5"


# ---------------------------------------------------------------------------
# 1. Validation Splitting
# ---------------------------------------------------------------------------

def split_validation_data(
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Tuple[Set[str], Set[str]]:
    """
    Perform a deterministic entity-level split of Source 1 entities into
    training and validation sets.

    Candidates must NOT be randomly split across the train/val boundary,
    as that would leak pairwise information. An entire S1 entity and all
    its candidates belong purely to either train or val.

    Parameters
    ----------
    gt:
        Preprocessed ground-truth DataFrame containing all S1 entities.
    config:
        Pipeline configuration (VAL_FRACTION and VAL_RANDOM_SEED).

    Returns
    -------
    train_s1_ids : set of str
        S1 entity IDs for the training split.
    val_s1_ids : set of str
        S1 entity IDs for the validation split.
    """
    s1_ids = gt[config.COL_GT_SOURCE1_ID].unique()
    
    # Sort for deterministic behavior before splitting
    s1_ids_sorted = np.sort(s1_ids)

    # Scikit-learn train_test_split is used for deterministic robust splitting
    train_ids, val_ids = train_test_split(
        s1_ids_sorted,
        test_size=config.VAL_FRACTION,
        random_state=config.VAL_RANDOM_SEED,
        shuffle=True,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)

    logger.info(
        "Validation split: %d train entities, %d val entities (fraction=%.2f, seed=%d)",
        len(train_set), len(val_set), config.VAL_FRACTION, config.VAL_RANDOM_SEED,
    )

    # Sanity check for leakage
    intersection = train_set.intersection(val_set)
    if intersection:
        raise RuntimeError(f"Leakage detected! {len(intersection)} entities in both splits.")

    return train_set, val_set


# ---------------------------------------------------------------------------
# 2. Core Metrics
# ---------------------------------------------------------------------------

def calculate_entity_metrics(
    true_set: Set[str],
    pred_set: Set[str],
) -> Tuple[int, int, int, int, int, float, float, float]:
    """
    Calculate precision, recall, and F0.5 for a single S1 entity.

    Parameters
    ----------
    true_set:
        Set of true matched entity IDs (from S2/S3).
    pred_set:
        Set of predicted matched entity IDs.

    Returns
    -------
    true_count : int
    pred_count : int
    tp : int
    fp : int
    fn : int
    precision : float
    recall : float
    f0_5 : float
    """
    true_count = len(true_set)
    pred_count = len(pred_set)

    # Case 1: True empty (singleton)
    if true_count == 0:
        if pred_count == 0:
            return 0, 0, 0, 0, 0, 1.0, 1.0, 1.0
        else:
            return 0, pred_count, 0, pred_count, 0, 0.0, 0.0, 0.0

    # Case 2: True non-empty, Prediction empty
    if pred_count == 0:
        return true_count, 0, 0, 0, true_count, 0.0, 0.0, 0.0

    # Case 3: Standard overlap
    tp = len(true_set.intersection(pred_set))
    fp = pred_count - tp
    fn = true_count - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f0_5 = 0.0
    if precision > 0 and recall > 0:
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    return true_count, pred_count, tp, fp, fn, precision, recall, f0_5


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate predicted match pairs against ground truth.

    Parameters
    ----------
    predictions_df:
        DataFrame containing predicted matches. Must have columns:
        - `source1_entity_id`
        - `matched_entity_id`
    gt:
        Preprocessed ground-truth DataFrame.
    config:
        Pipeline configuration.

    Returns
    -------
    macro_f05 : float
        The mean F0.5 score across all S1 entities present in the ground truth.
    diagnostics_df : pd.DataFrame
        Per-entity evaluation results.
    """
    # 1. Parse ground truth correctly using the isolated logic in blocking.py
    # (Includes the unverified delimiter assumption documented there).
    gt_dict = parse_ground_truth_matches(gt, config)

    # 2. Build prediction lookup
    # Group by S1, converting matched_entity_id into a set per S1.
    # Duplicate predicted IDs are safely handled by set() conversion.
    pred_dict: Dict[str, Set[str]] = {}
    if not predictions_df.empty:
        # Import dynamically or use hardcoded column name if imported at top
        # We imported COL_S1_ID above, let's use it. We'll use "matched_entity_id" literal
        pred_dict = (
            predictions_df
            .groupby(COL_S1_ID)["matched_entity_id"]
            .apply(set)
            .to_dict()
        )

    # 3. Evaluate every S1 entity in ground truth
    results = []
    
    # Sort for deterministic output ordering
    for s1_id in sorted(gt_dict.keys()):
        true_set = gt_dict[s1_id]
        pred_set = pred_dict.get(s1_id, set())
        
        tc, pc, tp, fp, fn, prec, rec, f05 = calculate_entity_metrics(true_set, pred_set)
        
        results.append({
            COL_S1_ID: s1_id,
            COL_TRUE_COUNT: tc,
            COL_PRED_COUNT: pc,
            COL_TP: tp,
            COL_FP: fp,
            COL_FN: fn,
            COL_PRECISION: prec,
            COL_RECALL: rec,
            COL_F05: f05,
        })

    # 4. Construct diagnostics and macro F0.5
    diagnostics_df = pd.DataFrame(results)
    
    macro_f05 = 0.0
    if len(diagnostics_df) > 0:
        macro_f05 = float(diagnostics_df[COL_F05].mean())

    return macro_f05, diagnostics_df


# ---------------------------------------------------------------------------
# 3. Threshold Tuning
# ---------------------------------------------------------------------------

def _evaluate_at_threshold(
    probabilities_df: pd.DataFrame,
    gt: pd.DataFrame,
    threshold: float,
    config: PipelineConfig,
) -> float:
    """Helper to evaluate macro F0.5 at a specific probability threshold."""
    # Filter predictions above threshold
    # Assume probabilities_df has 'probability' column
    mask = probabilities_df["probability"] >= threshold
    preds_above = probabilities_df[mask]
    
    macro_f05, _ = evaluate_predictions(preds_above, gt, config)
    return macro_f05


def tune_threshold(
    probabilities_df: pd.DataFrame,
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Perform a deterministic grid search to find the probability threshold
    that maximizes macro F0.5 on the provided predictions.

    Optimization Objective: Macro F0.5.

    Process:
    1. Coarse search over config.THRESHOLD_MIN to config.THRESHOLD_MAX 
       with config.THRESHOLD_STEP.
    2. Fine search around the best coarse threshold using config.THRESHOLD_FINE_STEP.

    Parameters
    ----------
    probabilities_df:
        Candidate pairs with predicted probabilities. Must have columns:
        - `source1_entity_id`
        - `matched_entity_id`
        - `probability`
    gt:
        Preprocessed ground-truth DataFrame.
    config:
        Pipeline configuration.

    Returns
    -------
    best_threshold : float
        Threshold maximizing macro F0.5.
    best_macro_f05 : float
        The macro F0.5 score at the best threshold.
    history : list of dict
        List containing {'threshold': t, 'macro_f05': f} for all evaluated thresholds.
    """
    history: List[Dict[str, float]] = []

    if probabilities_df.empty:
        logger.warning("Empty probabilities_df passed to tune_threshold. Returning default.")
        return config.DEFAULT_MATCH_THRESHOLD, 0.0, []

    # 1. Coarse Grid Search
    coarse_grid = np.arange(
        config.THRESHOLD_MIN, 
        config.THRESHOLD_MAX + 1e-9, 
        config.THRESHOLD_STEP
    )
    
    best_coarse_th = config.DEFAULT_MATCH_THRESHOLD
    best_coarse_score = -1.0

    logger.info("Starting coarse threshold search...")
    for th in coarse_grid:
        th = float(np.round(th, 3))
        score = _evaluate_at_threshold(probabilities_df, gt, th, config)
        history.append({"threshold": th, "macro_f05": score})
        
        if score > best_coarse_score:
            best_coarse_score = score
            best_coarse_th = th

    logger.info(
        "Coarse search complete. Best: %.3f (F0.5 = %.4f)", 
        best_coarse_th, best_coarse_score
    )

    # 2. Fine Grid Search
    # Search ± 1 coarse step around the best coarse threshold
    fine_start = max(config.THRESHOLD_MIN, best_coarse_th - config.THRESHOLD_STEP)
    fine_end = min(config.THRESHOLD_MAX, best_coarse_th + config.THRESHOLD_STEP)
    
    fine_grid = np.arange(
        fine_start,
        fine_end + 1e-9,
        config.THRESHOLD_FINE_STEP
    )

    best_final_th = best_coarse_th
    best_final_score = best_coarse_score

    logger.info("Starting fine threshold search...")
    for th in fine_grid:
        th = float(np.round(th, 3))
        
        # Skip if already evaluated in coarse search
        if any(abs(h["threshold"] - th) < 1e-5 for h in history):
            continue
            
        score = _evaluate_at_threshold(probabilities_df, gt, th, config)
        history.append({"threshold": th, "macro_f05": score})
        
        if score > best_final_score:
            best_final_score = score
            best_final_th = th

    # Sort history by threshold
    history.sort(key=lambda x: x["threshold"])

    logger.info(
        "Threshold tuning complete. Best: %.3f (F0.5 = %.4f)", 
        best_final_th, best_final_score
    )

    return best_final_th, best_final_score, history
