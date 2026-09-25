import os
import gc
import sys
import logging
import psutil
import duckdb
from src.storage import DiskStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAGGLE_PATH = "/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/"
DB_PATH = "output/entity_resolution.duckdb"
TEMP_DIR = "output/intermediate/tmp"

def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)

def run_ingestion():
    files_to_check = {
        "s1": os.path.join(KAGGLE_PATH, "train", "train_source1.tsv"),
        "s2": os.path.join(KAGGLE_PATH, "train", "train_source2.tsv"),
        "s3": os.path.join(KAGGLE_PATH, "train", "train_source3.tsv"),
        "gt": os.path.join(KAGGLE_PATH, "train", "train_ground_truth.tsv")
    }
    
    for name, path in files_to_check.items():
        if not os.path.exists(path):
            logger.error(f"Missing Kaggle data file for {name}: {path}")
            sys.exit(1)
            
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Strictly remove the old database so we don't end up with partial state
    if os.path.exists(DB_PATH):
        logger.warning(f"Existing database found at {DB_PATH}. Removing to ensure clean ingestion.")
        try:
            os.remove(DB_PATH)
        except OSError as e:
            logger.error(f"Failed to remove {DB_PATH}: {e}")
            sys.exit(1)
            
    logger.info("Initializing DuckDB for clean ingestion...")
    storage = DiskStorage(DB_PATH, TEMP_DIR, memory_limit="20GB", threads=4)
    batch_size = 50_000
    
    logger.info(f"[MEMORY] stage=Before Ingestion | rss_gb={get_rss_gb():.3f}")
    
    storage.ingest_source(files_to_check["s1"], "s1", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S1 | rss_gb={get_rss_gb():.3f}")
    
    storage.ingest_source(files_to_check["s2"], "s2", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S2 | rss_gb={get_rss_gb():.3f}")
    
    storage.ingest_source(files_to_check["s3"], "s3", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S3 | rss_gb={get_rss_gb():.3f}")
    
    storage.ingest_ground_truth(files_to_check["gt"], batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After GT | rss_gb={get_rss_gb():.3f}")
    
    storage.close()

def verify_counts():
    conn = duckdb.connect(DB_PATH)
    logger.info("--- Strict Verification Results ---")
    
    expected_counts = {
        "s1": 2206821,
        "s2": 5034616,
        "s3": 5285603
    }
    
    mismatch = False
    for src, expected in expected_counts.items():
        cnt = conn.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        if cnt != expected:
            logger.error(f"Table {src} count mismatch: Expected {expected}, got {cnt}")
            mismatch = True
        else:
            logger.info(f"Table {src}: {cnt} rows (VERIFIED)")
            
    gt_cnt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    gt_distinct = conn.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM ground_truth").fetchone()[0]
    
    if gt_cnt != 7638365:
        logger.error(f"Table ground_truth edge count mismatch: Expected 7638365, got {gt_cnt}")
        mismatch = True
    else:
        logger.info(f"Table ground_truth: {gt_cnt} edge pairs (VERIFIED)")
        
    if gt_distinct != 2083574:
        logger.error(f"Table ground_truth distinct S1 IDs mismatch: Expected 2083574, got {gt_distinct}")
        mismatch = True
    else:
        logger.info(f"Table ground_truth distinct S1 IDs: {gt_distinct} (VERIFIED)")
    
    if mismatch:
        logger.error("Database ingestion is incomplete or corrupted! Exiting with non-zero status.")
        conn.close()
        sys.exit(1)
        
    logger.info(f"Peak RSS at completion: {get_rss_gb():.3f} GB")
    logger.info(f"Database Path: {os.path.abspath(DB_PATH)}")
    
    conn.close()

if __name__ == "__main__":
    logger.info("=== KAGGLE INGESTION STAGE ===")
    
    try:
        run_ingestion()
        verify_counts()
    except SystemExit:
        logger.error("System exit triggered due to missing files. Ensure running in correct Kaggle environment.")
