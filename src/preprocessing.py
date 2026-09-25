"""
src/preprocessing.py
====================
Text normalization and data-loading utilities for Amazon ML Challenge 2026.
Approach #1 — Multi-pass Retrieval + LightGBM.

Responsibilities
----------------
- Normalize ``business_name``, ``business_address``, and ``country`` fields
  for downstream blocking and feature engineering.
- Load and validate every TSV file using ``cfg`` paths.
- Preserve all original columns; add new ``*_norm`` columns alongside them.
- Fail loudly if required columns are missing.
- Never filter rows based on country or missing values.
- Never query external APIs or databases.

Normalization contract (safe for callers to rely on)
----------------------------------------------------
- Input ``None`` or empty string → returns ``""`` (empty string).
- Output is always ``str``, never ``None``.
- Same input always produces the same output (deterministic).
- Original raw columns are never mutated.
- Numbers inside names/addresses are preserved.
- Transliterated text is NOT modified (no language-specific rewrites).

Import safety
-------------
This module is safe to import even when the dataset files are absent.
``FileNotFoundError`` is only raised when a loader function is called.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from src.config import cfg, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Low-level text normalization helpers
# ---------------------------------------------------------------------------

# Compile patterns once at module load — cheap and thread-safe.
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_AMPERSAND   = re.compile(r"\s*&\s*")
_RE_PUNCT_STRIP = re.compile(
    r"[^\w\s]",   # keep word chars (\w = letters/digits/underscore) and spaces
    flags=re.UNICODE,
)
# Matches characters that are not ASCII after NFKD decomposition.
# Used to strip combining diacritical marks (accents) from folded text.
_RE_NON_ASCII   = re.compile(r"[^\x00-\x7F]")


def normalize_text(
    text: Optional[str],
    *,
    replace_ampersand: bool = True,
    strip_punctuation: bool = True,
    fold_accents: bool = False,
) -> str:
    """
    Core text normalization used by both name and address normalizers.

    Steps applied in order
    ----------------------
    1. Coerce to string; handle ``None`` / ``NaN`` → return ``""``.
    2. Unicode NFC normalization (collapses composed / decomposed forms).
    3. Lowercase.
    4. Optionally fold accented characters to ASCII equivalents via NFKD
       decomposition (e.g., ``é`` → ``e``, ``ö`` → ``o``).
       Enabled for business name and address; disabled for country so that
       values like "Côte d'Ivoire" are preserved exactly.
    5. Optionally replace ``&`` (with surrounding spaces) with `` and ``.
    6. Optionally strip non-word, non-space characters.
       Numbers, letters (any script), and underscore are kept.
    7. Collapse internal whitespace to a single space.
    8. Strip leading / trailing whitespace.

    Parameters
    ----------
    text:
        Raw input string.  ``None`` and pandas ``NaN`` are treated as empty.
    replace_ampersand:
        When ``True``, ``&`` is replaced with ``and`` before stripping.
        This preserves the semantic token ("AT&T" → "at and t") instead of
        silently deleting ``&``.  Defaults to ``True``.
    strip_punctuation:
        When ``True``, non-word/non-space characters are removed after the
        ampersand step.  Set to ``False`` only for fields where punctuation
        carries structural meaning (currently unused).  Defaults to ``True``.
    fold_accents:
        When ``True``, apply NFKD decomposition and strip non-ASCII combining
        marks so that accented characters are replaced by their ASCII base
        (e.g., ``café`` → ``cafe``).  Defaults to ``False``.

    Returns
    -------
    str
        Normalized string, always a ``str``, never ``None``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        # Handles float('nan') from pandas and other non-string inputs.
        s = str(text)
        if s.lower() in ("nan", "none", "nat", ""):
            return ""
    else:
        s = text

    # Step 1 handled above; proceed with s as a genuine string.
    # Step 2 — Unicode NFC
    s = unicodedata.normalize("NFC", s)
    # Step 3 — lowercase
    s = s.lower()
    # Step 4 — accent folding (NFKD → drop combining marks → ASCII base chars)
    if fold_accents:
        s = unicodedata.normalize("NFKD", s)
        s = _RE_NON_ASCII.sub("", s)
    # Step 5 — ampersand
    if replace_ampersand:
        s = _RE_AMPERSAND.sub(" and ", s)
    # Step 6 — strip punctuation (keep word chars + whitespace)
    if strip_punctuation:
        s = _RE_PUNCT_STRIP.sub(" ", s)
    # Step 7 — collapse whitespace
    s = _RE_MULTI_SPACE.sub(" ", s)
    # Step 8 — strip edges
    s = s.strip()

    return s


def normalize_business_name(text: Optional[str]) -> str:
    """
    Normalize a ``business_name`` field for blocking and feature engineering.

    Applies the core ``normalize_text`` pipeline.  Ampersand replacement and
    punctuation stripping are both enabled because business names often use
    ``&`` interchangeably with ``and`` and carry heavy punctuation noise.

    Meaningful tokens (numbers, abbreviations, transliterated text) are
    preserved.  No external business-knowledge dictionary is used.

    Accent folding is applied so that encoding variations of the same name
    are mapped to the same normalized form (e.g., ``café`` = ``cafe``).

    Apostrophes and other punctuation are stripped and replaced with a space,
    so ``McDonald's`` becomes ``mcdonald s`` (not ``mcdonalds``).  This is
    intentional: the space-delimited token ``mcdonald`` is still highly
    discriminative, and aggressive character deletion risks merging distinct
    tokens that happen to share a prefix.

    Examples
    --------
    >>> normalize_business_name("  McDonald's   ")
    'mcdonald s'
    >>> normalize_business_name("AT&T Inc.")
    'at and t inc'
    >>> normalize_business_name("Café Rösti GmbH")
    'cafe rosti gmbh'
    >>> normalize_business_name(None)
    ''
    """
    return normalize_text(
        text,
        replace_ampersand=True,
        strip_punctuation=True,
        fold_accents=True,
    )


def normalize_business_address(text: Optional[str]) -> str:
    """
    Normalize a ``business_address`` field for blocking and feature engineering.

    Applies the same core pipeline as ``normalize_business_name``.

    Design notes
    ------------
    - Street numbers, PIN codes, and postal codes are kept because they are
      critical for address matching.
    - Punctuation (commas, periods, hyphens in addresses) is removed so that
      "123, Main St." and "123 Main St" produce the same token sequence.
    - No geocoding or address-database lookup is performed.
    - Ampersand replacement is enabled (rare in addresses but harmless).

    Examples
    --------
    >>> normalize_business_address("123, Main Street, New York, NY 10001")
    '123 main street new york ny 10001'
    >>> normalize_business_address("Plot No. 45, Sector-5, Gurugram")
    'plot no 45 sector 5 gurugram'
    >>> normalize_business_address(None)
    ''
    """
    return normalize_text(
        text,
        replace_ampersand=True,
        strip_punctuation=True,
        fold_accents=True,
    )


def normalize_country(text: Optional[str]) -> str:
    """
    Normalize a ``country`` field conservatively.

    Only lowercasing and whitespace stripping are applied.  No country-code
    expansion, no mapping to ISO codes, no filtering of unseen values.
    This ensures France (or any other new country in the test set) is
    passed through unchanged.

    Examples
    --------
    >>> normalize_country("United States")
    'united states'
    >>> normalize_country("  India  ")
    'india'
    >>> normalize_country("France")
    'france'
    >>> normalize_country(None)
    ''
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        s = str(text)
        if s.lower() in ("nan", "none", "nat", ""):
            return ""
    else:
        s = text
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = _RE_MULTI_SPACE.sub(" ", s)
    return s


# ---------------------------------------------------------------------------
# 2.  DataFrame-level normalization
# ---------------------------------------------------------------------------

def preprocess_source(
    df: pd.DataFrame,
    source_label: str,
    config: PipelineConfig = cfg,
) -> pd.DataFrame:
    """
    Validate and normalize a source DataFrame (S1, S2, or S3).

    Validation
    ----------
    - Checks that every required column defined in ``config.SOURCE_COLUMNS``
      is present.  Raises ``ValueError`` with a clear message if not.
    - Reports null counts per column to the logger.
    - Reports duplicate ``entity_id`` count (informational, not an error).

    Normalization
    -------------
    Adds three new columns alongside the originals:

    - ``name_norm``    — from ``normalize_business_name``
    - ``address_norm`` — from ``normalize_business_address``
    - ``country_norm`` — from ``normalize_country``

    The raw columns (``business_name``, ``business_address``, ``country``)
    are never modified.  No rows are dropped.

    Parameters
    ----------
    df:
        Raw DataFrame as read from a competition TSV file.
    source_label:
        Human-readable label used in log messages (e.g., ``"S1"``).
    config:
        Pipeline configuration (defaults to the module singleton).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with three additional normalized columns.
        The index is reset to a clean integer range.

    Raises
    ------
    ValueError
        If any required column is missing from ``df``.
    """
    _validate_source_columns(df, source_label, config)

    n_rows = len(df)
    logger.info("[%s] Loaded %d rows.", source_label, n_rows)

    # --- null / empty diagnostics (informational only) ---
    for col in config.SOURCE_COLUMNS:
        n_null = df[col].isna().sum()
        n_empty = (df[col].fillna("").astype(str).str.strip() == "").sum()
        if n_null > 0 or n_empty > 0:
            logger.info(
                "[%s] Column '%s': %d null, %d empty/whitespace-only.",
                source_label, col, n_null, n_empty,
            )

    # --- duplicate entity_id diagnostics ---
    n_dup_ids = df[config.COL_ENTITY_ID].duplicated().sum()
    if n_dup_ids > 0:
        logger.warning(
            "[%s] %d duplicate entity_id values found.",
            source_label, n_dup_ids,
        )

    # --- normalization (vectorized via pandas .map) ---
    df = df.copy()  # do not mutate the caller's DataFrame
    df[config.COL_NAME_NORM] = (
        df[config.COL_BUSINESS_NAME]
        .map(normalize_business_name, na_action="ignore")
        .fillna("")
    )
    df[config.COL_ADDRESS_NORM] = (
        df[config.COL_BUSINESS_ADDRESS]
        .map(normalize_business_address, na_action="ignore")
        .fillna("")
    )
    df[config.COL_COUNTRY_NORM] = (
        df[config.COL_COUNTRY]
        .map(normalize_country, na_action="ignore")
        .fillna("")
    )

    # Reset index for clean positional access downstream.
    df = df.reset_index(drop=True)

    logger.info(
        "[%s] Normalization complete. Columns: %s",
        source_label, list(df.columns),
    )
    return df


def preprocess_ground_truth(
    df: pd.DataFrame,
    config: PipelineConfig = cfg,
) -> pd.DataFrame:
    """
    Validate and lightly clean the ground-truth DataFrame.

    Validation
    ----------
    Checks that ``source1_entity_id`` and ``matched_entity_ids`` columns
    are present.

    Normalization
    -------------
    - ``source1_entity_id`` values are preserved exactly (no modification).
    - ``matched_entity_ids`` values are preserved as raw strings.
      Parsing into lists is performed later by evaluation logic.
    - Rows with empty ``matched_entity_ids`` are kept (they represent S1
      entities with no true matches — singleton entities).
    - ``matched_entity_ids`` NaN is coerced to ``""`` (empty string) so
      callers can safely test ``row == ""``.

    Parameters
    ----------
    df:
        Raw ground-truth DataFrame.
    config:
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Validated and cleaned ground-truth DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    _validate_ground_truth_columns(df, config)

    n_rows = len(df)
    logger.info("[GroundTruth] Loaded %d rows.", n_rows)

    n_null_ids = df[config.COL_GT_SOURCE1_ID].isna().sum()
    if n_null_ids > 0:
        logger.warning(
            "[GroundTruth] %d null source1_entity_id values — these rows "
            "cannot be evaluated and will be skipped downstream.",
            n_null_ids,
        )

    df = df.copy()
    # Coerce NaN matched_entity_ids to empty string.
    df[config.COL_GT_MATCHED_IDS] = (
        df[config.COL_GT_MATCHED_IDS].fillna("").astype(str).str.strip()
    )

    # How many S1 entities have no matches (singletons)?
    n_singletons = (df[config.COL_GT_MATCHED_IDS] == "").sum()
    logger.info(
        "[GroundTruth] %d S1 entities with no matches (singletons = %.1f%%).",
        n_singletons, 100.0 * n_singletons / max(n_rows, 1),
    )

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3.  TSV loaders
# ---------------------------------------------------------------------------

def _read_tsv(path: Path, config: PipelineConfig = cfg) -> pd.DataFrame:
    """
    Read a TSV file into a pandas DataFrame.

    Uses ``config.TSV_SEPARATOR`` and ``config.FILE_ENCODING``.
    ``dtype=str`` ensures that entity IDs and other fields are never
    silently coerced to numeric types (e.g., leading zeros preserved).
    """
    logger.info("Reading: %s", path)
    return pd.read_csv(
        path,
        sep=config.TSV_SEPARATOR,
        encoding=config.FILE_ENCODING,
        dtype=str,          # preserve entity IDs / postal codes as strings
        keep_default_na=False,  # treat empty cells as "" not NaN initially
        na_values=[""],     # then re-map "" → NaN for null-count reporting
    )


def load_training_data(
    config: PipelineConfig = cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all four training TSV files.

    Returns
    -------
    s1, s2, s3, gt : tuple of four DataFrames
        - ``s1``  — preprocessed Source 1 (normalized columns added)
        - ``s2``  — preprocessed Source 2 (normalized columns added)
        - ``s3``  — preprocessed Source 3 (normalized columns added)
        - ``gt``  — validated ground-truth (matched_entity_ids preserved as str)

    Raises
    ------
    FileNotFoundError
        If any required TSV file does not exist at the configured path.
    ValueError
        If required columns are missing from any file.
    """
    logger.info("=== Loading training data ===")
    _assert_file_exists(config.TRAIN_SOURCE1)
    _assert_file_exists(config.TRAIN_SOURCE2)
    _assert_file_exists(config.TRAIN_SOURCE3)
    _assert_file_exists(config.TRAIN_GROUND_TRUTH)

    s1 = preprocess_source(_read_tsv(config.TRAIN_SOURCE1, config), "S1-train", config)
    s2 = preprocess_source(_read_tsv(config.TRAIN_SOURCE2, config), "S2-train", config)
    s3 = preprocess_source(_read_tsv(config.TRAIN_SOURCE3, config), "S3-train", config)
    gt = preprocess_ground_truth(_read_tsv(config.TRAIN_GROUND_TRUTH, config), config)

    _log_loading_summary("Training", s1, s2, s3, gt)
    return s1, s2, s3, gt


def load_test_data(
    config: PipelineConfig = cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all three test TSV files.

    No ground truth is available for test data.

    Returns
    -------
    s1, s2, s3 : tuple of three DataFrames
        Preprocessed with normalized columns added.

    Raises
    ------
    FileNotFoundError
        If any required TSV file does not exist at the configured path.
    ValueError
        If required columns are missing from any file.
    """
    logger.info("=== Loading test data ===")
    _assert_file_exists(config.TEST_SOURCE1)
    _assert_file_exists(config.TEST_SOURCE2)
    _assert_file_exists(config.TEST_SOURCE3)

    s1 = preprocess_source(_read_tsv(config.TEST_SOURCE1, config), "S1-test", config)
    s2 = preprocess_source(_read_tsv(config.TEST_SOURCE2, config), "S2-test", config)
    s3 = preprocess_source(_read_tsv(config.TEST_SOURCE3, config), "S3-test", config)

    _log_loading_summary("Test", s1, s2, s3)
    return s1, s2, s3


# ---------------------------------------------------------------------------
# 4.  Private helpers
# ---------------------------------------------------------------------------

def _assert_file_exists(path: Path) -> None:
    """Raise ``FileNotFoundError`` with a clear message if path is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {path}\n"
            "Set the AMAZON_ML_DATA_DIR environment variable to your local "
            "dataset directory, or run this code on Kaggle where the dataset "
            "is mounted at the configured path."
        )


def _validate_source_columns(
    df: pd.DataFrame,
    source_label: str,
    config: PipelineConfig,
) -> None:
    """
    Check that all required source columns are present.

    Raises ``ValueError`` immediately listing ALL missing columns (not just
    the first one) so the user can fix them in one shot.
    """
    required: Sequence[str] = config.SOURCE_COLUMNS
    missing: List[str] = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source_label}] Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _validate_ground_truth_columns(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Check that ground-truth required columns are present."""
    required: Sequence[str] = config.GROUND_TRUTH_COLUMNS
    missing: List[str] = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[GroundTruth] Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _log_loading_summary(
    split: str,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Optional[pd.DataFrame] = None,
) -> None:
    """Log a concise row-count summary after loading a split."""
    lines = [
        f"=== {split} data loaded ===",
        f"  S1 : {len(s1):>8,} rows",
        f"  S2 : {len(s2):>8,} rows",
        f"  S3 : {len(s3):>8,} rows",
    ]
    if gt is not None:
        lines.append(f"  GT : {len(gt):>8,} rows")
    logger.info("\n".join(lines))
