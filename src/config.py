"""
src/config.py
=============
Centralized configuration for Amazon ML Challenge 2026 — Approach #1.
Multi-pass Retrieval + LightGBM.

Design principles
-----------------
- Single source of truth for every tuneable knob.
- No heavy ML library imports (stays lightweight and importable anywhere).
- DATA_DIR is resolved at import time from an environment variable,
  falling back to the canonical Kaggle dataset path.
- All other paths are derived from DATA_DIR and OUTPUT_DIR, so changing
  one variable is enough to redirect the whole pipeline.
- Country is treated as an OPEN SET throughout (never hard-coded to US/India).
- The matching threshold is NOT fixed at 0.5; it must be tuned on
  validation data using macro-averaged F0.5.

Environment variables
---------------------
AMAZON_ML_DATA_DIR   Override DATA_DIR (local development, CI, etc.)

Usage example
-------------
    from src.config import cfg

    df = pd.read_csv(cfg.TRAIN_SOURCE1, sep=cfg.TSV_SEPARATOR)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# 1. Path resolution helpers
# ---------------------------------------------------------------------------

#: Canonical Kaggle dataset root for this competition's student resource.
_KAGGLE_DATA_DIR: str = (
    "/kaggle/input/datasets/lokeshgile/"
    "student-resource-amazonml/student_resource/dataset/"
)

#: Environment-variable name used to override DATA_DIR locally.
_DATA_DIR_ENV_VAR: str = "AMAZON_ML_DATA_DIR"


def _resolve_data_dir() -> Path:
    """
    Resolve DATA_DIR with the following priority:

    1. ``AMAZON_ML_DATA_DIR`` environment variable (if set and non-empty).
    2. Kaggle canonical path (``_KAGGLE_DATA_DIR``).

    The resolved value is **not** validated for existence here so that
    ``config.py`` can be imported successfully even when the dataset is
    absent (e.g., during unit tests or CI that only tests helper logic).
    Path existence is checked lazily by the modules that actually read data.
    """
    env_override = os.environ.get(_DATA_DIR_ENV_VAR, "").strip()
    if env_override:
        return Path(env_override)
    return Path(_KAGGLE_DATA_DIR)


# ---------------------------------------------------------------------------
# 2. Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration for the full Approach-#1 pipeline.

    Every attribute has a concise docstring explaining its role.
    Values that require non-trivial justification carry additional comments.
    """

    # ------------------------------------------------------------------
    # 2a. Reproducibility
    # ------------------------------------------------------------------

    RANDOM_SEED: int = 42
    """Global random seed.  Pass to NumPy, scikit-learn, and LightGBM."""

    # ------------------------------------------------------------------
    # 2b. File-format constants
    # ------------------------------------------------------------------

    TSV_SEPARATOR: str = "\t"
    """Column separator for all TSV files in this competition."""

    FILE_ENCODING: str = "utf-8"
    """Character encoding for all competition TSV files."""

    # ------------------------------------------------------------------
    # 2c. Data paths
    # ------------------------------------------------------------------

    DATA_DIR: Path = field(default_factory=_resolve_data_dir)
    """
    Root directory that contains all competition TSV files.

    Resolution order:
      1. ``AMAZON_ML_DATA_DIR`` environment variable.
      2. Kaggle default: /kaggle/input/datasets/lokeshgile/
                         student-resource-amazonml/student_resource/dataset/

    To run locally, export::

        set AMAZON_ML_DATA_DIR=C:\\Projects\\Amazon_ML\\dataset
    """

    OUTPUT_DIR: Path = Path("output")
    """Directory where the pipeline writes its output files."""

    # ------------------------------------------------------------------
    # 2d. Dataset column names
    # ------------------------------------------------------------------

    #: Columns expected in train_source1/2/3.tsv and test_source1/2/3.tsv
    COL_ENTITY_ID: str = "entity_id"
    COL_BUSINESS_NAME: str = "business_name"
    COL_BUSINESS_ADDRESS: str = "business_address"
    COL_COUNTRY: str = "country"

    #: Columns expected in train_ground_truth.tsv
    COL_GT_SOURCE1_ID: str = "source1_entity_id"
    COL_GT_MATCHED_IDS: str = "matched_entity_ids"
    """
    ``matched_entity_ids`` is a comma-separated list of S2/S3 entity IDs.
    An empty string means no match exists for this S1 entity.
    """

    # Derived normalized column names (created by preprocessing.py)
    COL_NAME_NORM: str = "name_norm"
    COL_ADDRESS_NORM: str = "address_norm"
    COL_COUNTRY_NORM: str = "country_norm"

    # ------------------------------------------------------------------
    # 2e. Submission / output filenames
    # ------------------------------------------------------------------

    MATCHING_RESULTS_FILENAME: str = "matching_results.tsv"
    """Final submission file: one row per test S1 entity."""

    CANDIDATE_PAIRS_FILENAME: str = "candidate_pairs.tsv"
    """
    Intermediate output: the exact candidate set passed into LightGBM.
    Every ID that appears in matching_results.tsv must also appear here.
    """

    # ------------------------------------------------------------------
    # 2f. Validation split
    # ------------------------------------------------------------------

    VAL_FRACTION: float = 0.20
    """
    Fraction of training S1 entities held out for validation.

    Split is performed at the S1-entity level to avoid leakage:
    all candidate pairs for a given S1 entity land entirely in either
    train or validation, never split across both.
    """

    VAL_RANDOM_SEED: int = 42
    """Separate seed for the validation split (kept equal to RANDOM_SEED
    for full reproducibility; change only if ablation requires it)."""

    # ------------------------------------------------------------------
    # 2g. Retrieval / blocking configuration
    # ------------------------------------------------------------------

    # Pass 3 & 4 — character n-gram TF-IDF
    CHAR_NGRAM_MIN: int = 3
    """Minimum character n-gram size for TF-IDF vectorizer."""

    CHAR_NGRAM_MAX: int = 5
    """Maximum character n-gram size for TF-IDF vectorizer."""

    TOP_K_NAME: int = 20
    """
    Maximum candidates retrieved per S1 entity from the char-n-gram
    name similarity pass.  Increasing raises candidate recall but also
    pairwise computation cost.
    """

    TOP_K_ADDRESS: int = 10
    """Maximum candidates from the char-n-gram address similarity pass."""

    NGRAM_CHUNK_SIZE: int = 512
    """
    Number of S1 query vectors processed in a single batch during
    sparse cosine-similarity retrieval.  Avoids materializing the full
    dense S1×(S2+S3) similarity matrix in memory.
    """

    #: Minimum cosine similarity to accept a character-n-gram candidate.
    #: Lowering increases recall at the cost of more negatives for LightGBM.
    MIN_NGRAM_SIMILARITY_NAME: float = 0.20
    MIN_NGRAM_SIMILARITY_ADDR: float = 0.15

    USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT: bool = False
    """
    When True, retrieval passes may restrict candidates to the same
    country as the S1 query.

    IMPORTANT: This is False by default because:
      - Country is an open set (France appears only in test, not train).
      - Country labels may be noisy or missing.
      - Activating this could silently drop valid cross-country matches.

    Only enable after validation proves it improves candidate recall.
    """

    MAX_CANDIDATES_PER_ENTITY: int = 100
    """
    Hard ceiling on candidates per S1 entity after union of all passes.
    Protects against pathological cases (e.g., very common business names)
    that would generate an enormous feature matrix.
    When a ceiling is hit, candidates are ranked by combined retrieval
    signal and the top-MAX_CANDIDATES_PER_ENTITY are retained.
    """

    # ------------------------------------------------------------------
    # 2h. Feature engineering configuration
    # ------------------------------------------------------------------

    MAX_PAIRWISE_BATCH_SIZE: int = 50_000
    """
    Maximum number of candidate pairs processed in one feature-computation
    batch.  Prevents large intermediate DataFrames from exhausting RAM.
    """

    MAX_FEATURE_BATCH_SIZE: int = 50_000
    """
    Maximum number of pairs scored by RapidFuzz in one vectorized call.
    Mirrors MAX_PAIRWISE_BATCH_SIZE but can be tuned independently.
    """

    # ------------------------------------------------------------------
    # 2i. LightGBM configuration
    # ------------------------------------------------------------------

    LGBM_PARAMS: Dict[str, object] = field(default_factory=lambda: {
        # Task
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,

        # Reproducibility
        "seed": 42,
        "deterministic": True,

        # Tree structure — conservative baseline; tune later
        "num_leaves": 63,
        "max_depth": -1,          # -1 = unlimited (controlled via num_leaves)
        "min_child_samples": 20,

        # Learning
        "learning_rate": 0.05,
        "n_estimators": 1000,     # early stopping will select the best

        # Subsampling (reduces overfitting, speeds training)
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,

        # Regularisation
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,

        # Class imbalance
        # scale_pos_weight is set dynamically in model.py (n_neg / n_pos).
        # Do NOT hard-code it here because the ratio depends on the
        # actual candidate set generated at run time.
        "is_unbalance": False,    # we use scale_pos_weight instead

        # Threading — 0 = use all available cores
        "n_jobs": -1,
    })
    """
    LightGBM binary-classification parameters.

    These are a conservative baseline.  Do NOT present them as tuned
    without measured validation evidence.

    Key design decisions:
    - ``n_estimators=1000`` with early stopping: the model stops before
      1000 rounds if validation loss stops improving.
    - ``scale_pos_weight`` is computed dynamically from the training pair
      distribution and passed at training time (not stored here).
    - ``deterministic=True`` ensures fully reproducible results across runs
      on the same hardware.
    """

    LGBM_EARLY_STOPPING_ROUNDS: int = 50
    """Stop training if validation logloss does not improve for this many
    consecutive rounds."""

    LGBM_EVAL_METRIC: str = "binary_logloss"
    """Metric monitored during early stopping."""

    # ------------------------------------------------------------------
    # 2j. Threshold / decision configuration
    # ------------------------------------------------------------------

    DEFAULT_MATCH_THRESHOLD: float = 0.50
    """
    *** NOT the final threshold. ***

    This is a placeholder starting point only.  The actual threshold
    is selected by ``evaluation.tune_threshold()`` using macro F0.5
    on the held-out validation set.

    The competition metric is precision-heavy (F0.5), so the optimal
    threshold is typically higher than 0.5.  This value must never be
    used for final submission without prior validation-set tuning.
    """

    THRESHOLD_MIN: float = 0.30
    """Lower bound of threshold search grid."""

    THRESHOLD_MAX: float = 0.98
    """Upper bound of threshold search grid."""

    THRESHOLD_STEP: float = 0.01
    """Grid step for coarse threshold search."""

    THRESHOLD_FINE_STEP: float = 0.002
    """Grid step for fine-grained search around the best coarse region."""

    # ------------------------------------------------------------------
    # 2k. Candidate-recall diagnostic target (informational only)
    # ------------------------------------------------------------------

    MIN_CANDIDATE_RECALL_WARNING: float = 0.90
    """
    If blocking candidate recall (on the validation set) drops below
    this value, the pipeline will emit a loud warning.

    This is NOT a hard stop.  It is an operational signal to investigate
    the retrieval configuration before trusting downstream F0.5 scores.

    Candidate recall is the theoretical ceiling for the matcher recall,
    so low blocking recall cannot be recovered downstream.
    """

    # ------------------------------------------------------------------
    # 2l. Derived path properties
    # ------------------------------------------------------------------

    @property
    def TRAIN_SOURCE1(self) -> Path:
        return self.DATA_DIR / "train_source1.tsv"

    @property
    def TRAIN_SOURCE2(self) -> Path:
        return self.DATA_DIR / "train_source2.tsv"

    @property
    def TRAIN_SOURCE3(self) -> Path:
        return self.DATA_DIR / "train_source3.tsv"

    @property
    def TRAIN_GROUND_TRUTH(self) -> Path:
        return self.DATA_DIR / "train_ground_truth.tsv"

    @property
    def TEST_SOURCE1(self) -> Path:
        return self.DATA_DIR / "test_source1.tsv"

    @property
    def TEST_SOURCE2(self) -> Path:
        return self.DATA_DIR / "test_source2.tsv"

    @property
    def TEST_SOURCE3(self) -> Path:
        return self.DATA_DIR / "test_source3.tsv"

    @property
    def MATCHING_RESULTS_PATH(self) -> Path:
        return self.OUTPUT_DIR / self.MATCHING_RESULTS_FILENAME

    @property
    def CANDIDATE_PAIRS_PATH(self) -> Path:
        return self.OUTPUT_DIR / self.CANDIDATE_PAIRS_FILENAME

    @property
    def SOURCE_COLUMNS(self) -> Tuple[str, ...]:
        """Expected columns in every source TSV (S1, S2, S3)."""
        return (
            self.COL_ENTITY_ID,
            self.COL_BUSINESS_NAME,
            self.COL_BUSINESS_ADDRESS,
            self.COL_COUNTRY,
        )

    @property
    def GROUND_TRUTH_COLUMNS(self) -> Tuple[str, ...]:
        """Expected columns in train_ground_truth.tsv."""
        return (
            self.COL_GT_SOURCE1_ID,
            self.COL_GT_MATCHED_IDS,
        )

    def describe(self) -> str:
        """Return a human-readable summary of resolved paths and key knobs."""
        lines = [
            "=" * 60,
            "PipelineConfig — Approach #1 (Multi-pass + LightGBM)",
            "=" * 60,
            f"  DATA_DIR            : {self.DATA_DIR}",
            f"  OUTPUT_DIR          : {self.OUTPUT_DIR}",
            "",
            "  Train files",
            f"    TRAIN_SOURCE1     : {self.TRAIN_SOURCE1}",
            f"    TRAIN_SOURCE2     : {self.TRAIN_SOURCE2}",
            f"    TRAIN_SOURCE3     : {self.TRAIN_SOURCE3}",
            f"    TRAIN_GROUND_TRUTH: {self.TRAIN_GROUND_TRUTH}",
            "",
            "  Test files",
            f"    TEST_SOURCE1      : {self.TEST_SOURCE1}",
            f"    TEST_SOURCE2      : {self.TEST_SOURCE2}",
            f"    TEST_SOURCE3      : {self.TEST_SOURCE3}",
            "",
            "  Output files",
            f"    MATCHING_RESULTS  : {self.MATCHING_RESULTS_PATH}",
            f"    CANDIDATE_PAIRS   : {self.CANDIDATE_PAIRS_PATH}",
            "",
            f"  RANDOM_SEED         : {self.RANDOM_SEED}",
            f"  VAL_FRACTION        : {self.VAL_FRACTION}",
            "",
            "  Blocking",
            f"    CHAR_NGRAM_MIN    : {self.CHAR_NGRAM_MIN}",
            f"    CHAR_NGRAM_MAX    : {self.CHAR_NGRAM_MAX}",
            f"    TOP_K_NAME        : {self.TOP_K_NAME}",
            f"    TOP_K_ADDRESS     : {self.TOP_K_ADDRESS}",
            f"    NGRAM_CHUNK_SIZE  : {self.NGRAM_CHUNK_SIZE}",
            f"    MAX_CANDS/ENTITY  : {self.MAX_CANDIDATES_PER_ENTITY}",
            f"    COUNTRY_CONSTRAINT: {self.USE_COUNTRY_AS_RETRIEVAL_CONSTRAINT}",
            "",
            "  LightGBM",
            f"    n_estimators      : {self.LGBM_PARAMS['n_estimators']}",
            f"    learning_rate     : {self.LGBM_PARAMS['learning_rate']}",
            f"    num_leaves        : {self.LGBM_PARAMS['num_leaves']}",
            f"    early_stop_rounds : {self.LGBM_EARLY_STOPPING_ROUNDS}",
            "",
            "  Threshold",
            f"    DEFAULT (UNTUNED) : {self.DEFAULT_MATCH_THRESHOLD}",
            f"    Search range      : [{self.THRESHOLD_MIN}, {self.THRESHOLD_MAX}]"
            f" step={self.THRESHOLD_STEP}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Module-level singleton
# ---------------------------------------------------------------------------

#: The single importable config instance.
#: Import as:  ``from src.config import cfg``
cfg: PipelineConfig = PipelineConfig()
