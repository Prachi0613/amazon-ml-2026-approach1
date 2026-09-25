"""
src/run_pipeline.py
===================
Orchestrates the Amazon ML Challenge 2026 Approach #1 pipeline.

This entry point integrates preprocessing, blocking, features, modeling,
and submission modules. It expects the dataset path to be supplied via
the AMAZON_ML_DATA_DIR environment variable, falling back to the Kaggle
default if not provided.

Usage:
    python -m src.run_pipeline
"""

import logging
import sys
import time

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from src.config import cfg
from src.preprocessing import (
    load_and_validate_dataframe,
    normalize_dataframe,
    preprocess_ground_truth,
)
from src.blocking import (
    generate_candidates,
    evaluate_candidate_recall,
    summarize_candidates,
    parse_ground_truth_matches,
    COL_S1_ID,
)
from src.features import generate_features
from src.model import EntityMatcher, build_candidate_labels
from src.evaluation import split_validation_data, tune_threshold, evaluate_predictions
from src.submission import write_candidate_pairs, write_matching_results


def run():
    start_time = time.time()
    logger.info("Starting Amazon ML Challenge Pipeline - Approach #1")
    logger.info(f"Using DATA_DIR: {cfg.DATA_DIR}")
    
    # -----------------------------------------------------------------------
    # 1. Load and Preprocess Training Data
    # -----------------------------------------------------------------------
    logger.info("Loading training data...")
    raw_s1_train = load_and_validate_dataframe(cfg.TRAIN_SOURCE1, cfg)
    raw_s2_train = load_and_validate_dataframe(cfg.TRAIN_SOURCE2, cfg)
    raw_s3_train = load_and_validate_dataframe(cfg.TRAIN_SOURCE3, cfg)
    raw_gt = pd.read_csv(cfg.TRAIN_GROUND_TRUTH, sep=cfg.TSV_SEPARATOR)
    
    s1_train = normalize_dataframe(raw_s1_train, cfg)
    s2_train = normalize_dataframe(raw_s2_train, cfg)
    s3_train = normalize_dataframe(raw_s3_train, cfg)
    gt = preprocess_ground_truth(raw_gt, cfg)
    
    print("\n" + "="*40)
    print("TRAIN DATA")
    print("="*40)
    print(f"- S1 rows: {len(s1_train)}")
    print(f"- S2 rows: {len(s2_train)}")
    print(f"- S3 rows: {len(s3_train)}")
    
    # -----------------------------------------------------------------------
    # 2. Candidate Generation (Training)
    # -----------------------------------------------------------------------
    logger.info("Generating training candidates...")
    candidates_train = generate_candidates(s1_train, s2_train, s3_train, cfg)
    
    gt_dict = parse_ground_truth_matches(gt, cfg)
    c_recall_stats, _, _, _ = evaluate_candidate_recall(candidates_train, gt_dict, cfg)
    c_summary = summarize_candidates(candidates_train, len(s1_train))
    
    print("\n" + "="*40)
    print("CANDIDATE GENERATION (TRAIN)")
    print("="*40)
    print(f"- total candidate pairs: {c_summary['total_candidates']}")
    print(f"- average candidates/S1: {c_summary['avg_candidates_per_s1']:.2f}")
    
    # Calculate median candidates manually as it's not in summary by default
    cand_counts = candidates_train.groupby(COL_S1_ID).size()
    median_cands = cand_counts.median() if not cand_counts.empty else 0
    max_cands = cand_counts.max() if not cand_counts.empty else 0
    zero_cands = len(s1_train) - len(cand_counts)
    
    print(f"- median candidates/S1: {median_cands:.2f}")
    print(f"- max candidates/S1: {max_cands}")
    print(f"- zero-candidate S1 count: {zero_cands}")
    print(f"- candidate recall: {c_recall_stats['overall_recall']:.4f}")
    
    # -----------------------------------------------------------------------
    # 3. Validation Split
    # -----------------------------------------------------------------------
    logger.info("Splitting validation entities...")
    train_s1_ids, val_s1_ids = split_validation_data(gt, cfg)
    
    mask_train = candidates_train[COL_S1_ID].isin(train_s1_ids)
    mask_val = candidates_train[COL_S1_ID].isin(val_s1_ids)
    
    cands_t = candidates_train[mask_train].copy()
    cands_v = candidates_train[mask_val].copy()
    
    # -----------------------------------------------------------------------
    # 4. Label Generation
    # -----------------------------------------------------------------------
    logger.info("Building candidate labels...")
    y_train = build_candidate_labels(cands_t, gt, cfg)
    y_val = build_candidate_labels(cands_v, gt, cfg)
    
    # -----------------------------------------------------------------------
    # 5. Feature Generation
    # -----------------------------------------------------------------------
    logger.info("Generating training features...")
    X_train = generate_features(s1_train, s2_train, s3_train, cands_t, cfg)
    logger.info("Generating validation features...")
    X_val = generate_features(s1_train, s2_train, s3_train, cands_v, cfg)
    
    # -----------------------------------------------------------------------
    # 6. LightGBM Training
    # -----------------------------------------------------------------------
    logger.info("Training LightGBM Matcher...")
    matcher = EntityMatcher(cfg)
    matcher.fit(X_train, y_train, X_val, y_val)
    
    # -----------------------------------------------------------------------
    # 7. Validation Prediction & Threshold Tuning
    # -----------------------------------------------------------------------
    logger.info("Predicting validation probabilities...")
    val_probs = matcher.predict_proba(X_val)
    cands_v_scored = cands_v.copy()
    cands_v_scored["probability"] = val_probs
    
    # Filter ground truth to validation set only for accurate evaluation
    val_gt = gt[gt[cfg.COL_GT_SOURCE1_ID].isin(val_s1_ids)].copy()
    
    # Evaluate at default threshold
    logger.info(f"Evaluating at default threshold ({cfg.DEFAULT_MATCH_THRESHOLD})...")
    mask_default = cands_v_scored["probability"] >= cfg.DEFAULT_MATCH_THRESHOLD
    preds_default = cands_v_scored[mask_default]
    f05_default, diag_default = evaluate_predictions(preds_default, val_gt, cfg)
    
    logger.info("Tuning threshold...")
    best_th, best_f05, _ = tune_threshold(cands_v_scored, val_gt, cfg)
    
    # Get precision and recall at best threshold
    mask_best = cands_v_scored["probability"] >= best_th
    preds_best = cands_v_scored[mask_best]
    _, diag_best = evaluate_predictions(preds_best, val_gt, cfg)
    
    prec_mean = diag_best["precision"].mean() if len(diag_best) > 0 else 0.0
    rec_mean = diag_best["recall"].mean() if len(diag_best) > 0 else 0.0
    
    pred_matches = len(preds_best)
    true_matches = int(diag_best["true_count"].sum()) if len(diag_best) > 0 else 0
    
    print("\n" + "="*40)
    print("VALIDATION")
    print("="*40)
    print(f"- number of validation S1 entities: {len(val_s1_ids)}")
    print(f"- number of validation candidate pairs: {len(cands_v)}")
    print(f"- default threshold: {cfg.DEFAULT_MATCH_THRESHOLD}")
    print(f"- F0.5 at default threshold: {f05_default:.4f}")
    print(f"- best threshold: {best_th:.3f}")
    print(f"- best validation macro F0.5: {best_f05:.4f}")
    print(f"- precision: {prec_mean:.4f}")
    print(f"- recall: {rec_mean:.4f}")
    print(f"- predicted matches: {pred_matches}")
    print(f"- true matches: {true_matches}")

    # -----------------------------------------------------------------------
    # 8. Test Data Processing
    # -----------------------------------------------------------------------
    logger.info("Loading test data...")
    raw_s1_test = load_and_validate_dataframe(cfg.TEST_SOURCE1, cfg)
    raw_s2_test = load_and_validate_dataframe(cfg.TEST_SOURCE2, cfg)
    raw_s3_test = load_and_validate_dataframe(cfg.TEST_SOURCE3, cfg)
    
    s1_test = normalize_dataframe(raw_s1_test, cfg)
    s2_test = normalize_dataframe(raw_s2_test, cfg)
    s3_test = normalize_dataframe(raw_s3_test, cfg)
    
    logger.info("Generating test candidates...")
    candidates_test = generate_candidates(s1_test, s2_test, s3_test, cfg)
    
    logger.info("Generating test features...")
    X_test = generate_features(s1_test, s2_test, s3_test, candidates_test, cfg)
    
    logger.info("Predicting test probabilities...")
    test_probs = matcher.predict_proba(X_test)
    
    logger.info(f"Writing submissions using threshold {best_th:.3f}...")
    write_candidate_pairs(s1_test, candidates_test, cfg)
    write_matching_results(s1_test, candidates_test, test_probs, best_th, s2_test, s3_test, cfg)
    
    # Calculate test metrics
    test_cand_counts = candidates_test.groupby(COL_S1_ID).size()
    test_zero_cands = len(s1_test) - len(test_cand_counts)
    test_preds_mask = test_probs >= best_th
    test_preds = candidates_test[test_preds_mask]
    test_pred_matches = len(test_preds)
    
    test_pred_counts = test_preds.groupby(COL_S1_ID).size()
    test_zero_match_preds = len(s1_test) - len(test_pred_counts)
    
    print("\n" + "="*40)
    print("TEST")
    print("="*40)
    print(f"- number of test S1 entities: {len(s1_test)}")
    print(f"- number of test candidate pairs: {len(candidates_test)}")
    print(f"- zero-candidate S1 count: {test_zero_cands}")
    print(f"- predicted zero-match S1 count: {test_zero_match_preds}")
    print(f"- predicted match count: {test_pred_matches}")
    
    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run()
