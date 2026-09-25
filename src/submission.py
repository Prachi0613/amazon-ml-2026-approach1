"""
src/submission.py
=================
Submission generation and serialization for the Amazon ML Challenge.

Responsibilities
----------------
- Apply the tuned probability threshold to test candidates.
- Group accepted candidates into the final S1 matched lists.
- Serialize matching_results.tsv and candidate_pairs.tsv.
- Perform sanity checks on IDs and completeness to ensure all test S1
  entities (including zero-match) are present.
- Do NOT perform threshold optimization or model training.
"""

import logging
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd

from src.config import cfg, PipelineConfig
from src.blocking import COL_S1_ID, COL_CAND_ID, COL_CAND_SRC, SOURCE_S2, SOURCE_S3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation / Sanity Checks
# ---------------------------------------------------------------------------

def _validate_ids(
    test_s1_ids: Set[str],
    test_s2_ids: Set[str],
    test_s3_ids: Set[str],
    candidates_df: pd.DataFrame,
) -> None:
    """Internal check to ensure candidate IDs exist in the respective test sets."""
    # Check S1
    cand_s1_set = set(candidates_df[COL_S1_ID].unique())
    unknown_s1 = cand_s1_set - test_s1_ids
    if unknown_s1:
        raise ValueError(f"Candidate DataFrame contains unknown S1 IDs: {list(unknown_s1)[:5]}...")
        
    # Check S2
    s2_cands = candidates_df[candidates_df[COL_CAND_SRC] == SOURCE_S2]
    cand_s2_set = set(s2_cands[COL_CAND_ID].unique())
    unknown_s2 = cand_s2_set - test_s2_ids
    if unknown_s2:
        raise ValueError(f"Candidate DataFrame contains unknown S2 IDs: {list(unknown_s2)[:5]}...")
        
    # Check S3
    s3_cands = candidates_df[candidates_df[COL_CAND_SRC] == SOURCE_S3]
    cand_s3_set = set(s3_cands[COL_CAND_ID].unique())
    unknown_s3 = cand_s3_set - test_s3_ids
    if unknown_s3:
        raise ValueError(f"Candidate DataFrame contains unknown S3 IDs: {list(unknown_s3)[:5]}...")

    # Check valid sources
    invalid_src = set(candidates_df[COL_CAND_SRC].unique()) - {SOURCE_S2, SOURCE_S3}
    if invalid_src:
        raise ValueError(f"Candidate DataFrame contains invalid sources: {invalid_src}")
        
    # Check for duplicate candidates (S1, Cand, Src)
    duplicates = candidates_df.duplicated(subset=[COL_S1_ID, COL_CAND_ID, COL_CAND_SRC])
    if duplicates.any():
        raise ValueError(f"Candidate DataFrame contains {duplicates.sum()} duplicate pair(s).")


# ---------------------------------------------------------------------------
# Matching Results Serialization
# ---------------------------------------------------------------------------

def _serialize_match_list(matched_ids: set) -> str:
    """
    Serialize a set of matched IDs into the required output format.
    
    [UNVERIFIED ASSUMPTION]: Assumes comma-separated strings.
    This exact string encoding MUST be verified against the official Kaggle
    specifications and validator once available.
    """
    if not matched_ids:
        return ""
    # Deterministic sort for serialization
    return ",".join(sorted(list(matched_ids)))


def generate_matching_results_df(
    test_s1: pd.DataFrame,
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    test_s2: pd.DataFrame,
    test_s3: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> pd.DataFrame:
    """
    Apply threshold and generate the final matching results DataFrame.
    
    Guarantees exactly one row per test S1 entity.
    """
    if len(candidates_df) != len(probabilities):
        raise ValueError(
            f"Length mismatch: {len(candidates_df)} candidates vs "
            f"{len(probabilities)} probabilities."
        )

    # Convert test sets to ID sets for strict validation
    eid_col = config.COL_ENTITY_ID
    s1_ids = set(test_s1[eid_col])
    s2_ids = set(test_s2[eid_col])
    s3_ids = set(test_s3[eid_col])
    
    # 1. Sanity check incoming candidate pairs
    _validate_ids(s1_ids, s2_ids, s3_ids, candidates_df)
    
    # 2. Apply threshold
    # Note: probability array exactly aligns with candidates_df by row index
    mask = probabilities >= threshold
    accepted = candidates_df[mask].copy()
    
    # 3. Group by S1 and collect matched IDs
    # Use a set to inherently prevent duplicates within a match list
    match_lookup = {}
    if not accepted.empty:
        # Group purely on S1 ID, returning the matched_entity_id
        grouped = accepted.groupby(COL_S1_ID)[COL_CAND_ID].apply(set)
        match_lookup = grouped.to_dict()
        
    # 4. Construct final output array mapping exactly to all test S1 entities
    # Deterministic sorting of S1 IDs
    all_s1_sorted = sorted(list(s1_ids))
    
    results = []
    for s1_id in all_s1_sorted:
        matches = match_lookup.get(s1_id, set())
        serialized = _serialize_match_list(matches)
        
        results.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": serialized
        })
        
    out_df = pd.DataFrame(results)
    
    # 5. Final sanity checks on output
    if len(out_df) != len(s1_ids):
        raise RuntimeError("Output row count does not match test S1 count.")
    if out_df["source1_entity_id"].duplicated().any():
        raise RuntimeError("Duplicate S1 IDs found in generated output.")
        
    return out_df


def write_matching_results(
    test_s1: pd.DataFrame,
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    test_s2: pd.DataFrame,
    test_s3: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Path:
    """
    Generate and serialize matching_results.tsv to disk.
    
    Returns the Path to the written file.
    """
    out_df = generate_matching_results_df(
        test_s1, candidates_df, probabilities, threshold, test_s2, test_s3, config
    )
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "matching_results.tsv"
    
    logger.info(f"Writing matching_results.tsv with {len(out_df)} rows...")
    out_df.to_csv(out_path, sep='\t', index=False)
    
    return out_path


# ---------------------------------------------------------------------------
# Candidate Pairs Serialization
# ---------------------------------------------------------------------------

def generate_candidate_pairs_df(
    test_s1: pd.DataFrame,
    candidates_df: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> pd.DataFrame:
    """
    Format candidate pairs into the official submission format.
    Guarantees exactly one row per test S1 entity.
    """
    eid_col = config.COL_ENTITY_ID
    s1_ids = set(test_s1[eid_col])
    
    cand_lookup = {}
    if not candidates_df.empty:
        grouped = candidates_df.groupby(COL_S1_ID)[COL_CAND_ID].apply(set)
        cand_lookup = grouped.to_dict()
        
    all_s1_sorted = sorted(list(s1_ids))
    
    results = []
    for s1_id in all_s1_sorted:
        cands = cand_lookup.get(s1_id, set())
        serialized = _serialize_match_list(cands)
        
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": serialized
        })
        
    out_df = pd.DataFrame(results)
    
    if len(out_df) != len(s1_ids):
        raise RuntimeError("Output row count does not match test S1 count.")
    if out_df["source1_entity_id"].duplicated().any():
        raise RuntimeError("Duplicate S1 IDs found in generated output.")
        
    return out_df

def write_candidate_pairs(
    test_s1: pd.DataFrame,
    candidates_df: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Path:
    """
    Serialize the candidate pairs to disk in the official Kaggle format.
    Returns the Path to the written file.
    """
    if candidates_df.empty:
        logger.warning("write_candidate_pairs called with an empty candidates DataFrame.")
        
    out_df = generate_candidate_pairs_df(test_s1, candidates_df, config)
        
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
    
    logger.info(f"Writing candidate_pairs.tsv with {len(out_df)} candidate rows...")
    out_df.to_csv(out_path, sep='\t', index=False)
    
    return out_path
