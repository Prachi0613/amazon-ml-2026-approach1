"""
src/blocking.py
===============
Candidate generation (blocking / retrieval) for Amazon ML Challenge 2026.
Approach #1 -- Multi-pass Retrieval + LightGBM.

Responsibilities
----------------
- Run four retrieval passes and return their union as a candidate DataFrame.
- Compute candidate-recall diagnostics against training ground truth.
- Expose summary statistics (candidate counts, reduction ratio, etc.).
- Do NOT calculate pairwise features, train models, or write output files.

Four retrieval passes
---------------------
Pass 1  Exact normalized business-name match (inverted index)
Pass 2  Exact normalized business-address match (inverted index)
Pass 3  Character n-gram TF-IDF on business name  (sparse cosine, top-K)
Pass 4  Character n-gram TF-IDF on business address (sparse cosine, top-K)

Final candidate set = UNION of all four passes, deduplicated by
(source1_entity_id, matched_entity_id, matched_source).

Source identity (S2 vs S3) is preserved throughout -- entity IDs from
Source 2 and Source 3 are never conflated even if their IDs overlap.

Scalability guarantees
----------------------
- Exact passes use O(|S1| + |S2| + |S3|) inverted-index lookups.
- N-gram passes process S1 in chunks of NGRAM_CHUNK_SIZE to avoid
  materialising a dense (|S1| x |S2+S3|) similarity matrix.
- Only the top-K most similar candidates above MIN_NGRAM_SIMILARITY
  are returned per S1 record per pass.

Import safety
-------------
Safe to import without the dataset present.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import cfg, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Column-name constants (candidate DataFrame schema)
# ---------------------------------------------------------------------------

# These are the column names produced by generate_candidates().
# Downstream modules (features.py, model.py) should import them from here.
COL_S1_ID       = "source1_entity_id"
COL_CAND_ID     = "matched_entity_id"
COL_CAND_SRC    = "matched_source"      # "S2" or "S3"
COL_EX_NAME     = "from_exact_name"
COL_EX_ADDR     = "from_exact_address"
COL_NG_NAME     = "from_ngram_name"
COL_NG_ADDR     = "from_ngram_address"
COL_NG_NAME_SC  = "ngram_name_score"
COL_NG_ADDR_SC  = "ngram_address_score"
COL_PASS_COUNT  = "retrieval_pass_count"

# Source labels
SOURCE_S2 = "S2"
SOURCE_S3 = "S3"

# Type aliases
_ExactIndex = Dict[str, List[Tuple[str, str]]]   # norm_value -> [(eid, source), ...]
_CandKey    = Tuple[str, str, str]                # (s1_id, matched_id, source)


# ---------------------------------------------------------------------------
# 2.  Ground-truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth_matches(
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Dict[str, Set[str]]:
    """
    Parse the ground-truth DataFrame into a per-S1-entity lookup.

    Parameters
    ----------
    gt:
        Preprocessed ground-truth DataFrame (output of
        ``preprocessing.preprocess_ground_truth``).
    config:
        Pipeline configuration.

    Returns
    -------
    dict
        ``{source1_entity_id: set_of_matched_entity_ids}``

        S1 entities with no matches map to an empty set.
        The matched IDs are stripped of surrounding whitespace but are
        otherwise not modified.

    Notes
    -----
    [UNVERIFIED ASSUMPTION]: The ``matched_entity_ids`` column is currently 
    parsed as a comma-separated string (e.g. ``"S2-00047,S2-00193,S3-00812"``).
    This delimiter format is UNVERIFIED because the actual competition TSV 
    files are not available locally. 
    
    Before relying on this for candidate-recall measurements in Kaggle, 
    inspect the actual ``train_ground_truth.tsv`` file to confirm if it uses 
    commas, pipes (``|``), spaces, or a JSON array. If the official challenge 
    documentation specifies the exact encoding, update this parsing logic 
    accordingly.

    An empty string indicates a singleton (no true matches).
    No assumption is made about the prefix format of IDs.  The set
    contains the raw ID strings exactly as they appear in the TSV.
    """
    result: Dict[str, Set[str]] = {}
    s1_id_col = config.COL_GT_SOURCE1_ID
    matched_col = config.COL_GT_MATCHED_IDS

    for _, row in gt.iterrows():
        s1_id = row[s1_id_col]
        raw = str(row[matched_col]).strip()
        if not raw:
            result[s1_id] = set()
        else:
            result[s1_id] = {mid.strip() for mid in raw.split(",") if mid.strip()}
    return result


# ---------------------------------------------------------------------------
# 3.  Exact-match inverted index (Passes 1 and 2)
# ---------------------------------------------------------------------------

def _build_exact_index(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    field: str,
    config: PipelineConfig,
) -> _ExactIndex:
    """
    Build an inverted index mapping a normalised field value to all
    (entity_id, source_label) pairs that share that value.

    Parameters
    ----------
    s2, s3:
        Preprocessed source DataFrames (must contain ``entity_id`` and
        the requested ``field`` column).
    field:
        The normalised column to index (e.g. ``cfg.COL_NAME_NORM``).
    config:
        Pipeline configuration.

    Returns
    -------
    dict
        ``{normalised_value: [(entity_id, source_label), ...]}``

        Empty strings are never added to the index (they are not
        discriminative and would produce massive false-positive buckets).
    """
    index: _ExactIndex = defaultdict(list)
    eid_col = config.COL_ENTITY_ID

    for df, source_label in [(s2, SOURCE_S2), (s3, SOURCE_S3)]:
        entity_ids: List[str] = df[eid_col].tolist()
        field_vals: List[str] = df[field].fillna("").tolist()
        for eid, val in zip(entity_ids, field_vals):
            if val:  # skip empty -- they are not discriminative
                index[val].append((eid, source_label))

    return dict(index)


def _retrieve_exact_pass(
    s1: pd.DataFrame,
    index: _ExactIndex,
    field: str,
    pass_col: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Retrieve candidates for all S1 entities using the exact inverted index.
    """
    out_s1 = []
    out_cand = []
    out_src = []
    
    eid_col = config.COL_ENTITY_ID
    s1_ids: List[str] = s1[eid_col].tolist()
    field_vals: List[str] = s1[field].fillna("").tolist()

    for s1_id, val in zip(s1_ids, field_vals):
        if not val:
            continue  # empty normalised value -- skip
        matches = index.get(val, [])
        for (matched_id, matched_source) in matches:
            out_s1.append(s1_id)
            out_cand.append(matched_id)
            out_src.append(matched_source)

    return pd.DataFrame({
        COL_S1_ID: out_s1,
        COL_CAND_ID: out_cand,
        COL_CAND_SRC: pd.Series(out_src, dtype="category"),
        pass_col: True,
    })


# ---------------------------------------------------------------------------
# 4.  Character n-gram TF-IDF index (Passes 3 and 4)
# ---------------------------------------------------------------------------

def _build_ngram_index(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    field: str,
    config: PipelineConfig,
) -> Tuple[TfidfVectorizer, csr_matrix, List[str], List[str]]:
    """
    Fit a character-n-gram TF-IDF vectorizer on the S2+S3 corpus and
    transform the corpus into a sparse L2-normalised matrix.

    The TF-IDF vectorizer is fitted only once per field; the same
    vectorizer is then used to transform S1 query vectors at retrieval
    time.

    Parameters
    ----------
    s2, s3:
        Preprocessed source DataFrames.
    field:
        Normalised column (``COL_NAME_NORM`` or ``COL_ADDRESS_NORM``).
    config:
        Pipeline configuration.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer; use ``vectorizer.transform(texts)`` for queries.
    X_corpus : csr_matrix, shape (N_s2 + N_s3, n_features)
        L2-normalised TF-IDF matrix of the full S2+S3 corpus.
        Cosine similarity between a query vector and corpus row i equals
        their dot product because both are unit-normalised.
    corpus_ids : list of str
        Entity ID for each row in ``X_corpus``.
    corpus_sources : list of str
        Source label ("S2" or "S3") for each row in ``X_corpus``.

    Notes
    -----
    - ``norm='l2'`` (the sklearn default) ensures that dot product == cosine.
    - ``analyzer='char_wb'`` pads each token with word boundaries, giving
      more discriminative character n-grams than plain ``'char'``.
    - ``sublinear_tf=True`` dampens the weight of very frequent n-grams.
    - ``max_df=0.99`` drops n-grams that appear in almost every record
      (they add noise without discriminative value).
    - Empty strings in the corpus produce all-zero vectors and are harmless.
    """
    eid_col = config.COL_ENTITY_ID

    corpus_ids: List[str] = []
    corpus_sources: List[str] = []
    corpus_texts: List[str] = []

    for df, source_label in [(s2, SOURCE_S2), (s3, SOURCE_S3)]:
        eids = df[eid_col].tolist()
        texts = df[field].fillna("").tolist()
        corpus_ids.extend(eids)
        corpus_sources.extend([source_label] * len(eids))
        corpus_texts.extend(texts)

    # Guard: need at least one non-empty string to fit the vectorizer.
    if not any(corpus_texts):
        logger.warning(
            "All corpus texts are empty for field '%s'. "
            "N-gram pass will return no candidates.",
            field,
        )
        # Return a dummy unfitted vectorizer -- callers handle empty results.
        return (
            TfidfVectorizer(),   # unfitted
            csr_matrix((len(corpus_texts), 0)),
            corpus_ids,
            corpus_sources,
        )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(config.CHAR_NGRAM_MIN, config.CHAR_NGRAM_MAX),
        norm="l2",
        sublinear_tf=True,
        max_df=0.99,
        min_df=1,
        dtype=np.float32,
    )
    X_corpus: csr_matrix = vectorizer.fit_transform(corpus_texts)
    logger.debug(
        "N-gram index for '%s': corpus size=%d, vocab=%d, matrix=%s",
        field, len(corpus_texts), len(vectorizer.vocabulary_), X_corpus.shape,
    )
    return vectorizer, X_corpus, corpus_ids, corpus_sources


def _sparse_topk_rows(
    sim_csr: csr_matrix,
    top_k: int,
    min_sim: float,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Extract the top-K entries (above ``min_sim``) for each row of a
    sparse CSR similarity matrix.

    Parameters
    ----------
    sim_csr:
        Sparse CSR similarity matrix, shape (n_queries, n_corpus).
        Only stored non-zeros are examined; true zero entries are ignored.
    top_k:
        Maximum number of results per row.
    min_sim:
        Minimum similarity threshold; entries below this are discarded.

    Returns
    -------
    list of (row_idx, col_indices, scores)
        One tuple per row that has at least one entry >= ``min_sim``.
        ``col_indices`` and ``scores`` are 1-D NumPy arrays of equal length,
        sorted by score descending (with corpus index as deterministic
        tie-breaker).
    """
    results: List[Tuple[int, np.ndarray, np.ndarray]] = []
    indptr = sim_csr.indptr
    indices = sim_csr.indices
    data = sim_csr.data

    for row_idx in range(sim_csr.shape[0]):
        start, end = int(indptr[row_idx]), int(indptr[row_idx + 1])
        if start == end:
            continue  # no non-zero similarities for this row

        cols = indices[start:end]
        vals = data[start:end]

        # Filter by minimum similarity
        mask = vals >= min_sim
        if not mask.any():
            continue
        cols = cols[mask]
        vals = vals[mask]

        # Select top-K via partial sort (O(n) instead of O(n log n))
        if len(cols) > top_k:
            # argpartition returns indices of the top_k largest values
            partition_idx = np.argpartition(vals, -top_k)[-top_k:]
            cols = cols[partition_idx]
            vals = vals[partition_idx]

        # Deterministic sort: score DESC, then corpus-column index ASC
        # np.lexsort applies keys right-to-left: last key is primary.
        sort_order = np.lexsort((cols, -vals))
        cols = cols[sort_order]
        vals = vals[sort_order]

        results.append((row_idx, cols, vals))

    return results


def _retrieve_ngram_pass(
    s1: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X_corpus: csr_matrix,
    corpus_ids: List[str],
    corpus_sources: List[str],
    field: str,
    top_k: int,
    min_sim: float,
    score_col: str,
    pass_col: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Retrieve n-gram candidates for all S1 entities using chunked sparse
    cosine-similarity retrieval. Returns a DataFrame instead of dicts.
    """
    # Guard: if vectorizer has no vocabulary (empty corpus), return nothing.
    if not hasattr(vectorizer, "vocabulary_") or X_corpus.shape[1] == 0:
        return pd.DataFrame()

    eid_col = config.COL_ENTITY_ID
    chunk_size = config.NGRAM_CHUNK_SIZE

    s1_ids: List[str] = s1[eid_col].tolist()
    s1_texts: List[str] = s1[field].fillna("").tolist()
    n_s1 = len(s1_ids)
    
    out_s1 = []
    out_cand = []
    out_src = []
    out_scores = []

    for chunk_start in range(0, n_s1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_s1)
        chunk_texts = s1_texts[chunk_start:chunk_end]
        chunk_s1_ids = s1_ids[chunk_start:chunk_end]

        # Filter: skip entirely-empty chunks (avoids useless vectorizer calls)
        non_empty_mask = [bool(t) for t in chunk_texts]
        if not any(non_empty_mask):
            continue

        # Transform queries to TF-IDF vectors (sparse)
        X_query: csr_matrix = vectorizer.transform(chunk_texts)

        # Sparse cosine similarity
        sim_csr: csr_matrix = (X_query @ X_corpus.T).tocsr()

        # Extract top-K per row
        for row_idx, col_indices, scores in _sparse_topk_rows(
            sim_csr, top_k, min_sim
        ):
            if not non_empty_mask[row_idx]:
                continue  # S1 record had empty text -- skip
            s1_id = chunk_s1_ids[row_idx]
            for col_idx, score in zip(col_indices, scores):
                out_s1.append(s1_id)
                out_cand.append(corpus_ids[col_idx])
                out_src.append(corpus_sources[col_idx])
                out_scores.append(float(score))

    return pd.DataFrame({
        COL_S1_ID: out_s1,
        COL_CAND_ID: out_cand,
        COL_CAND_SRC: pd.Series(out_src, dtype="category"),
        pass_col: True,
        score_col: pd.Series(out_scores, dtype=np.float32),
    })


# ---------------------------------------------------------------------------
# 5.  Candidate merging (union + deduplication)
# ---------------------------------------------------------------------------

def _merge_candidate_records(
    record_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge candidate records from multiple retrieval passes into a single DataFrame.
    """
    dfs = [df for df in record_dfs if not df.empty]
    if not dfs:
        return pd.DataFrame(columns=[
            COL_S1_ID, COL_CAND_ID, COL_CAND_SRC,
            COL_EX_NAME, COL_EX_ADDR, COL_NG_NAME, COL_NG_ADDR,
            COL_NG_NAME_SC, COL_NG_ADDR_SC, COL_PASS_COUNT,
        ])

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure columns exist and fill NAs
    bool_cols = [COL_EX_NAME, COL_EX_ADDR, COL_NG_NAME, COL_NG_ADDR]
    for col in bool_cols:
        if col not in combined.columns:
            combined[col] = False
        else:
            combined[col] = combined[col].fillna(False)

    score_cols = [COL_NG_NAME_SC, COL_NG_ADDR_SC]
    for col in score_cols:
        if col not in combined.columns:
            combined[col] = 0.0
        else:
            combined[col] = combined[col].fillna(0.0).astype(np.float32)

    # Group and aggregate
    grouped = combined.groupby([COL_S1_ID, COL_CAND_ID, COL_CAND_SRC], observed=True, as_index=False).max()

    # Compute pass count
    grouped[COL_PASS_COUNT] = (
        grouped[COL_EX_NAME].astype(np.int8) + 
        grouped[COL_EX_ADDR].astype(np.int8) + 
        grouped[COL_NG_NAME].astype(np.int8) + 
        grouped[COL_NG_ADDR].astype(np.int8)
    )

    return grouped


# ---------------------------------------------------------------------------
# 6.  Candidate cap
# ---------------------------------------------------------------------------

def _apply_candidate_cap(
    df: pd.DataFrame,
    config: PipelineConfig,
    country_lookup: Optional[Dict[str, str]] = None,
    s1_country_lookup: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Apply MAX_CANDIDATES_PER_ENTITY to the DataFrame.
    """
    if df.empty:
        return df

    # --- Optional country constraint ---
    if (
        config.USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT
        and country_lookup is not None
        and s1_country_lookup is not None
    ):
        s1_countries = df[COL_S1_ID].map(s1_country_lookup).fillna("")
        cand_countries = df[COL_CAND_ID].map(country_lookup).fillna("")
        keep_mask = (s1_countries == "") | (cand_countries == "") | (
            s1_countries == cand_countries
        )
        n_before = len(df)
        df = df[keep_mask].reset_index(drop=True)
        logger.info(
            "Country constraint removed %d candidates (kept %d).",
            n_before - len(df), len(df),
        )

    max_cands = config.MAX_CANDIDATES_PER_ENTITY

    tier = np.where(
        df[COL_EX_NAME], 0,
        np.where(df[COL_EX_ADDR], 1, 2)
    )
    best_score = df[[COL_NG_NAME_SC, COL_NG_ADDR_SC]].max(axis=1)

    df = df.assign(_tier=tier, _best_score=best_score)
    df.sort_values(
        by=["_tier", "_best_score", COL_CAND_ID],
        ascending=[True, False, True],
        inplace=True
    )

    df = df.groupby(COL_S1_ID, sort=False).head(max_cands)
    df.drop(columns=["_tier", "_best_score"], inplace=True)

    df.sort_values([COL_S1_ID, COL_CAND_ID, COL_CAND_SRC], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# 7.  Country lookups for optional constraint
# ---------------------------------------------------------------------------

def _build_country_lookup(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    config: PipelineConfig,
) -> Dict[str, str]:
    """
    Build ``{entity_id: country_norm}`` for all S2+S3 records.
    Used only when ``USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT`` is True.
    """
    lookup: Dict[str, str] = {}
    for df in (s2, s3):
        eids = df[config.COL_ENTITY_ID].tolist()
        countries = df[config.COL_COUNTRY_NORM].fillna("").tolist()
        for eid, country in zip(eids, countries):
            lookup[eid] = country
    return lookup


def _build_s1_country_lookup(
    s1: pd.DataFrame,
    config: PipelineConfig,
) -> Dict[str, str]:
    """Build ``{entity_id: country_norm}`` for all S1 records."""
    eids = s1[config.COL_ENTITY_ID].tolist()
    countries = s1[config.COL_COUNTRY_NORM].fillna("").tolist()
    return dict(zip(eids, countries))


# ---------------------------------------------------------------------------
# 8.  Main public API
# ---------------------------------------------------------------------------

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> pd.DataFrame:
    """
    Run all four retrieval passes and return the unified candidate DataFrame.

    Parameters
    ----------
    s1, s2, s3:
        Preprocessed DataFrames (output of ``preprocessing.preprocess_source``).
        Must contain ``entity_id``, ``name_norm``, ``address_norm``,
        ``country_norm``.
    config:
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Candidate pairs with columns:

        source1_entity_id   str
        matched_entity_id   str
        matched_source      str   ("S2" or "S3")
        from_exact_name     bool
        from_exact_address  bool
        from_ngram_name     bool
        from_ngram_address  bool
        ngram_name_score    float32
        ngram_address_score float32
        retrieval_pass_count int

        One row per unique (source1_entity_id, matched_entity_id, matched_source)
        triple. Sorted deterministically by (source1_entity_id, matched_entity_id,
        matched_source).

    Notes
    -----
    - Country constraint is applied only if
      ``config.USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT`` is True.
    - Candidates per S1 entity are capped at
      ``config.MAX_CANDIDATES_PER_ENTITY``.
    """
    t0 = time.perf_counter()
    name_field = config.COL_NAME_NORM
    addr_field = config.COL_ADDRESS_NORM

    # ------------------------------------------------------------------
    # Pass 1: Exact name
    # ------------------------------------------------------------------
    logger.info("Pass 1: building exact name index ...")
    t1 = time.perf_counter()
    exact_name_idx = _build_exact_index(s2, s3, name_field, config)
    p1_records = _retrieve_exact_pass(s1, exact_name_idx, name_field, COL_EX_NAME, config)
    logger.info(
        "Pass 1 done in %.2fs: %d candidate pairs.", time.perf_counter() - t1, len(p1_records)
    )

    # ------------------------------------------------------------------
    # Pass 2: Exact address
    # ------------------------------------------------------------------
    logger.info("Pass 2: building exact address index ...")
    t2 = time.perf_counter()
    exact_addr_idx = _build_exact_index(s2, s3, addr_field, config)
    p2_records = _retrieve_exact_pass(s1, exact_addr_idx, addr_field, COL_EX_ADDR, config)
    logger.info(
        "Pass 2 done in %.2fs: %d candidate pairs.", time.perf_counter() - t2, len(p2_records)
    )

    # ------------------------------------------------------------------
    # Pass 3: Character n-gram name retrieval
    # ------------------------------------------------------------------
    logger.info(
        "Pass 3: fitting n-gram name index (ngram=%d-%d, top_k=%d, min_sim=%.2f) ...",
        config.CHAR_NGRAM_MIN, config.CHAR_NGRAM_MAX,
        config.TOP_K_NAME, config.MIN_NGRAM_SIMILARITY_NAME,
    )
    t3 = time.perf_counter()
    name_vect, X_name_corpus, name_ids, name_sources = _build_ngram_index(
        s2, s3, name_field, config
    )
    p3_records = _retrieve_ngram_pass(
        s1, name_vect, X_name_corpus, name_ids, name_sources,
        field=name_field,
        top_k=config.TOP_K_NAME,
        min_sim=config.MIN_NGRAM_SIMILARITY_NAME,
        score_col=COL_NG_NAME_SC,
        pass_col=COL_NG_NAME,
        config=config,
    )
    logger.info(
        "Pass 3 done in %.2fs: %d candidate pairs.", time.perf_counter() - t3, len(p3_records)
    )

    # ------------------------------------------------------------------
    # Pass 4: Character n-gram address retrieval
    # ------------------------------------------------------------------
    logger.info(
        "Pass 4: fitting n-gram address index (ngram=%d-%d, top_k=%d, min_sim=%.2f) ...",
        config.CHAR_NGRAM_MIN, config.CHAR_NGRAM_MAX,
        config.TOP_K_ADDRESS, config.MIN_NGRAM_SIMILARITY_ADDR,
    )
    t4 = time.perf_counter()
    addr_vect, X_addr_corpus, addr_ids, addr_sources = _build_ngram_index(
        s2, s3, addr_field, config
    )
    p4_records = _retrieve_ngram_pass(
        s1, addr_vect, X_addr_corpus, addr_ids, addr_sources,
        field=addr_field,
        top_k=config.TOP_K_ADDRESS,
        min_sim=config.MIN_NGRAM_SIMILARITY_ADDR,
        score_col=COL_NG_ADDR_SC,
        pass_col=COL_NG_ADDR,
        config=config,
    )
    logger.info(
        "Pass 4 done in %.2fs: %d candidate pairs.", time.perf_counter() - t4, len(p4_records)
    )

    # ------------------------------------------------------------------
    # Union + deduplication
    # ------------------------------------------------------------------
    logger.info("Merging candidate records from all passes ...")
    merged = _merge_candidate_records([p1_records, p2_records, p3_records, p4_records])
    logger.info("Unique candidate pairs before cap: %d", len(merged))

    # ------------------------------------------------------------------
    # Optional country constraint + cap
    # ------------------------------------------------------------------
    country_lookup = None
    s1_country_lookup = None
    if config.USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT:
        country_lookup = _build_country_lookup(s2, s3, config)
        s1_country_lookup = _build_s1_country_lookup(s1, config)

    candidates_df = _apply_candidate_cap(
        merged, config, country_lookup, s1_country_lookup
    )

    total_time = time.perf_counter() - t0
    logger.info(
        "Candidate generation complete: %d pairs, %.2fs total.",
        len(candidates_df), total_time,
    )
    return candidates_df


# ---------------------------------------------------------------------------
# 9.  Candidate recall evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate_recall(
    candidates_df: pd.DataFrame,
    gt: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Dict[str, Any]:
    """
    Compute candidate-recall metrics by comparing the candidate set against
    the training ground truth.

    Candidate recall is the theoretical ceiling on downstream model recall:
    a true match that is absent from the candidate set can never be found
    by the matcher.

    Parameters
    ----------
    candidates_df:
        Output of ``generate_candidates``.
    gt:
        Preprocessed ground-truth DataFrame.
    config:
        Pipeline configuration.

    Returns
    -------
    dict
        Keys:

        total_true_pairs       -- total positive pairs in ground truth
        recovered_pairs        -- positives found in candidate set
        missing_pairs          -- positives absent from candidate set
        candidate_recall       -- recovered / total (0.0 if no positives)
        total_candidates       -- total candidate rows
        n_s1_with_candidates   -- S1 entities that have >= 1 candidate
        n_s1_zero_candidates   -- S1 entities with no candidates
        avg_candidates_per_s1  -- mean candidates per S1 entity
        median_candidates      -- median candidates per S1 entity
        max_candidates         -- maximum candidates for any S1 entity
        recall_by_pass         -- per-pass recall dict
        missing_pair_examples  -- up to 10 (s1_id, matched_id) examples
        below_warning_threshold -- True if recall < config.MIN_CANDIDATE_RECALL_WARNING
    """
    # --- Parse ground truth ---
    gt_dict = parse_ground_truth_matches(gt, config)

    # --- Build candidate lookup: s1_id -> set[matched_entity_id] ---
    if len(candidates_df) > 0:
        cand_lookup: Dict[str, Set[str]] = (
            candidates_df
            .groupby(COL_S1_ID)[COL_CAND_ID]
            .apply(set)
            .to_dict()
        )
    else:
        cand_lookup = {}

    # --- Per-entity candidate counts ---
    per_entity_counts: Dict[str, int] = {
        s1_id: len(cands) for s1_id, cands in cand_lookup.items()
    }
    all_s1_ids = list(gt_dict.keys())
    counts_all = [per_entity_counts.get(sid, 0) for sid in all_s1_ids]

    # --- Overall recall ---
    total_true_pairs = 0
    recovered_pairs  = 0
    missing_examples: List[Tuple[str, str]] = []

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue  # singleton -- nothing to recover
        cands = cand_lookup.get(s1_id, set())
        for match_id in true_matches:
            total_true_pairs += 1
            if match_id in cands:
                recovered_pairs += 1
            else:
                if len(missing_examples) < 10:
                    missing_examples.append((s1_id, match_id))

    candidate_recall = (
        recovered_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    )

    # --- Per-pass recall ---
    pass_recall: Dict[str, float] = {}
    for pass_col, label in [
        (COL_EX_NAME, "exact_name"),
        (COL_EX_ADDR, "exact_address"),
        (COL_NG_NAME, "ngram_name"),
        (COL_NG_ADDR, "ngram_address"),
    ]:
        if len(candidates_df) == 0 or pass_col not in candidates_df.columns:
            pass_recall[label] = 0.0
            continue
        pass_df = candidates_df[candidates_df[pass_col] == True]
        if len(pass_df) == 0:
            pass_recall[label] = 0.0
            continue
        pass_lookup = (
            pass_df.groupby(COL_S1_ID)[COL_CAND_ID].apply(set).to_dict()
        )
        p_recovered = sum(
            1
            for s1_id, true_matches in gt_dict.items()
            for mid in true_matches
            if mid in pass_lookup.get(s1_id, set())
        )
        pass_recall[label] = (
            p_recovered / total_true_pairs if total_true_pairs > 0 else 0.0
        )

    # --- Summary stats ---
    n_s1_with_candidates = sum(1 for c in counts_all if c > 0)
    n_s1_zero_candidates = len(counts_all) - n_s1_with_candidates
    avg_cands = float(np.mean(counts_all)) if counts_all else 0.0
    med_cands = float(np.median(counts_all)) if counts_all else 0.0
    max_cands = int(max(counts_all)) if counts_all else 0

    below_warning = candidate_recall < config.MIN_CANDIDATE_RECALL_WARNING

    if below_warning and total_true_pairs > 0:
        logger.warning(
            "CANDIDATE RECALL WARNING: %.4f is below the configured "
            "warning threshold %.4f. Low blocking recall is the ceiling "
            "for model recall -- investigate the retrieval configuration.",
            candidate_recall, config.MIN_CANDIDATE_RECALL_WARNING,
        )

    return {
        "total_true_pairs":       total_true_pairs,
        "recovered_pairs":        recovered_pairs,
        "missing_pairs":          total_true_pairs - recovered_pairs,
        "candidate_recall":       candidate_recall,
        "total_candidates":       len(candidates_df),
        "n_s1_with_candidates":   n_s1_with_candidates,
        "n_s1_zero_candidates":   n_s1_zero_candidates,
        "avg_candidates_per_s1":  avg_cands,
        "median_candidates":      med_cands,
        "max_candidates":         max_cands,
        "recall_by_pass":         pass_recall,
        "missing_pair_examples":  missing_examples,
        "below_warning_threshold": below_warning,
    }


# ---------------------------------------------------------------------------
# 10.  Candidate summary statistics
# ---------------------------------------------------------------------------

def summarize_candidates(
    candidates_df: pd.DataFrame,
    s1: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> Dict[str, Any]:
    """
    Return a human-readable statistics dictionary about the candidate set.

    Parameters
    ----------
    candidates_df:
        Output of ``generate_candidates``.
    s1:
        Preprocessed Source-1 DataFrame (used to determine total S1 count).
    config:
        Pipeline configuration.

    Returns
    -------
    dict
        Contains candidate counts, per-pass counts, and reduction-ratio
        statistics.  Suitable for logging and experiment tracking.
    """
    n_s1_total = len(s1)
    n_s2_plus_s3_hypothetical = None  # only computable if we have s2/s3 sizes

    if len(candidates_df) == 0:
        return {
            "n_s1_total": n_s1_total,
            "total_candidates": 0,
            "avg_per_s1": 0.0,
            "median_per_s1": 0.0,
            "max_per_s1": 0,
            "n_s1_with_any_candidate": 0,
            "n_s1_zero_candidates": n_s1_total,
            "n_from_exact_name": 0,
            "n_from_exact_address": 0,
            "n_from_ngram_name": 0,
            "n_from_ngram_address": 0,
            "n_from_multiple_passes": 0,
        }

    per_s1 = (
        candidates_df.groupby(COL_S1_ID)
        .size()
        .reindex(s1[config.COL_ENTITY_ID], fill_value=0)
    )

    n_from_ex_name  = int(candidates_df[COL_EX_NAME].sum())
    n_from_ex_addr  = int(candidates_df[COL_EX_ADDR].sum())
    n_from_ng_name  = int(candidates_df[COL_NG_NAME].sum())
    n_from_ng_addr  = int(candidates_df[COL_NG_ADDR].sum())
    n_multi_pass    = int((candidates_df[COL_PASS_COUNT] > 1).sum())

    return {
        "n_s1_total":              n_s1_total,
        "total_candidates":        len(candidates_df),
        "avg_per_s1":              float(per_s1.mean()),
        "median_per_s1":           float(per_s1.median()),
        "max_per_s1":              int(per_s1.max()),
        "n_s1_with_any_candidate": int((per_s1 > 0).sum()),
        "n_s1_zero_candidates":    int((per_s1 == 0).sum()),
        "n_from_exact_name":       n_from_ex_name,
        "n_from_exact_address":    n_from_ex_addr,
        "n_from_ngram_name":       n_from_ng_name,
        "n_from_ngram_address":    n_from_ng_addr,
        "n_from_multiple_passes":  n_multi_pass,
    }
