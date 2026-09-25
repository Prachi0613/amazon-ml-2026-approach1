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
    
    # Check if DB already populated (resumable)
    if os.path.exists(DB_PATH):
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
            cnt = conn.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
            conn.close()
            if cnt > 1000000:
                logger.info("Database appears to already be populated. Skipping ingestion.")
                return
        except Exception:
            pass
    
    logger.info("Initializing DuckDB for ingestion...")
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
    logger.info("--- Verification Results ---")
    
    for src in ["s1", "s2", "s3"]:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        logger.info(f"Table {src}: {cnt} rows")
        
    gt_cnt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    gt_distinct = conn.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM ground_truth").fetchone()[0]
    
    logger.info(f"Table ground_truth: {gt_cnt} edge pairs | {gt_distinct} distinct S1 IDs")
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
