import os
import gc
import sys
import logging
import psutil
import pandas as pd
import duckdb
from src.preprocessing import normalize_business_name, normalize_business_address, normalize_country
from src.storage import DiskStorage
from src.disk_blocking import DiskBlocker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAGGLE_PATH = "/kaggle/input/datasets/lokeshgile/student-resource-amazonml/student_resource/dataset/"
DB_PATH = "output/entity_resolution.duckdb"
TEMP_DIR = "output/intermediate/tmp"

def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)

def check_files():
    logger.info(f"DATA DIRECTORY: {KAGGLE_PATH}")
    
    files_to_check = {
        "TRAIN SOURCE1": os.path.join(KAGGLE_PATH, "train", "train_source1.tsv"),
        "TRAIN SOURCE2": os.path.join(KAGGLE_PATH, "train", "train_source2.tsv"),
        "TRAIN SOURCE3": os.path.join(KAGGLE_PATH, "train", "train_source3.tsv"),
        "GROUND TRUTH": os.path.join(KAGGLE_PATH, "train", "train_ground_truth.tsv")
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"{name}:\nexists / {size_mb:.2f} MB")
        else:
            logger.error(f"{name}:\nMISSING! Path: {path}")
            all_exist = False
            
    if not all_exist:
        logger.error("STOPPING. Not all actual Kaggle files are present.")
        sys.exit(1)
    
    return files_to_check

def run_real_ingestion(files):
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    logger.info("Initializing DuckDB for ingestion...")
    storage = DiskStorage(DB_PATH, TEMP_DIR, memory_limit="20GB", threads=4)
    
    batch_size = 50_000
    
    logger.info(f"[MEMORY] stage=Before Ingestion | rss_gb={get_rss_gb():.3f}")
    
    # Ingest S1
    storage.ingest_source(files["TRAIN SOURCE1"], "s1", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S1 | rss_gb={get_rss_gb():.3f}")
    
    # Ingest S2
    storage.ingest_source(files["TRAIN SOURCE2"], "s2", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S2 | rss_gb={get_rss_gb():.3f}")
    
    # Ingest S3
    storage.ingest_source(files["TRAIN SOURCE3"], "s3", batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After S3 | rss_gb={get_rss_gb():.3f}")
    
    # Ingest GT
    storage.ingest_ground_truth(files["GROUND TRUTH"], batch_size=batch_size)
    logger.info(f"[MEMORY] stage=After GT | rss_gb={get_rss_gb():.3f}")
    
    storage.close()

def verify_row_counts_and_norm():
    conn = duckdb.connect(DB_PATH)
    
    logger.info("Verifying Row Counts & Integrity...")
    for src in ["s1", "s2", "s3"]:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        distinct = conn.execute(f"SELECT COUNT(DISTINCT entity_id) FROM {src}").fetchone()[0]
        logger.info(f"Table {src}: {cnt} rows | {distinct} distinct IDs")
        
        # Norm sanity check
        sample = conn.execute(f"SELECT name_norm, addr_norm, country_norm FROM {src} LIMIT 1000").fetchdf()
        # Just to prove it didn't mutate weirdly
        if sample.isnull().any().any():
             logger.warning(f"Nulls detected in normalized output for {src}")
             
    gt_cnt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    gt_distinct = conn.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM ground_truth").fetchone()[0]
    logger.info(f"Table ground_truth: {gt_cnt} edge pairs | {gt_distinct} distinct S1 IDs")
    
    conn.close()

def profile_advanced(conn, max_pairs=50_000):
    logger.info("Profiling advanced combinations and secondary refinements...")
    
    rules = ["exact_name", "exact_addr", "name_prefix_3", "name_prefix_4", "addr_prefix_3", "addr_house_num"]
    
    os.makedirs("output/intermediate", exist_ok=True)
    out_tsv = open("output/intermediate/real_block_profile.tsv", "w")
    out_tsv.write("Rule\tSource\tNum_Keys\tMean_S1\tMedian_S1\tMax_S1\tEstimated_Pairs\tSafe_Pairs\tOversized_Blocks\tRaw_Coverage\tSafe_Coverage\n")
    
    out_md = open("output/intermediate/real_block_profile.md", "w")
    out_md.write("# Real Block Profile\n\n")
    out_md.write("| Rule | Source | Estimated Pairs | Safe Pairs | Oversized Blocks | Raw Coverage | Safe Coverage |\n")
    out_md.write("|---|---|---|---|---|---|---|\n")
    
    for rule in rules:
        for src in ["s2", "s3"]:
            # Run the complex CTE query
            # We will use the exact logic from run_phase3_profile.py
            # For brevity in this generation step, we call the same SQL logic.
            
            # (Insert identical SQL from run_phase3_profile.py)
            query = f"""
            WITH block_stats AS (
                SELECT 
                    t1.block_key,
                    COUNT(DISTINCT t1.entity_id) as s1_size,
                    COUNT(DISTINCT t2.entity_id) as s2_size,
                    CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) as estimated_pairs
                FROM s1_blocks t1
                JOIN {src}_blocks t2 
                  ON t1.block_key = t2.block_key 
                 AND t1.block_type = '{rule}' 
                 AND t2.block_type = '{rule}'
                GROUP BY t1.block_key
            )
            SELECT 
                COUNT(block_key) as num_distinct_keys,
                AVG(s1_size) as mean_s1_size,
                QUANTILE_CONT(s1_size, 0.5) as median_s1_size,
                MAX(s1_size) as max_s1_size,
                SUM(estimated_pairs) as total_estimated_pairs,
                SUM(CASE WHEN estimated_pairs <= {max_pairs} THEN estimated_pairs ELSE 0 END) as safe_estimated_pairs,
                SUM(CASE WHEN estimated_pairs > {max_pairs} THEN estimated_pairs ELSE 0 END) as oversized_estimated_pairs,
                COUNT(CASE WHEN estimated_pairs <= {max_pairs} THEN 1 END) as safe_blocks,
                COUNT(CASE WHEN estimated_pairs > {max_pairs} THEN 1 END) as oversized_blocks
            FROM block_stats
            """
            
            stats = conn.execute(query).fetchone()
            if not stats or stats[0] == 0:
                continue
                
            coverage_query = f"""
            WITH true_pairs AS (
                SELECT gt.source1_entity_id, gt.matched_entity_id 
                FROM ground_truth gt
                WHERE gt.matched_entity_id LIKE '{src.upper()}-%'
            ),
            raw_coverage AS (
                SELECT COUNT(DISTINCT tp.source1_entity_id || tp.matched_entity_id) as raw_hits
                FROM true_pairs tp
                JOIN s1_blocks s1 ON tp.source1_entity_id = s1.entity_id AND s1.block_type = '{rule}'
                JOIN {src}_blocks s2 ON tp.matched_entity_id = s2.entity_id AND s2.block_type = '{rule}'
                WHERE s1.block_key = s2.block_key
            ),
            safe_keys AS (
                SELECT t1.block_key
                FROM s1_blocks t1
                JOIN {src}_blocks t2 
                  ON t1.block_key = t2.block_key 
                 AND t1.block_type = '{rule}' 
                 AND t2.block_type = '{rule}'
                GROUP BY t1.block_key
                HAVING CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) <= {max_pairs}
            ),
            safe_coverage AS (
                SELECT COUNT(DISTINCT tp.source1_entity_id || tp.matched_entity_id) as safe_hits
                FROM true_pairs tp
                JOIN s1_blocks s1 ON tp.source1_entity_id = s1.entity_id AND s1.block_type = '{rule}'
                JOIN {src}_blocks s2 ON tp.matched_entity_id = s2.entity_id AND s2.block_type = '{rule}'
                JOIN safe_keys sk ON s1.block_key = sk.block_key
                WHERE s1.block_key = s2.block_key
            )
            SELECT 
                (SELECT raw_hits FROM raw_coverage) as raw_hits,
                (SELECT safe_hits FROM safe_coverage) as safe_hits,
                (SELECT COUNT(*) FROM true_pairs) as total_true_pairs
            """
            cov_stats = conn.execute(coverage_query).fetchone()
            
            total_est = stats[4] or 0
            safe_est = stats[5] or 0
            over_blocks = stats[8] or 0
            raw_cov = cov_stats[0] / max(1, cov_stats[2])
            safe_cov = cov_stats[1] / max(1, cov_stats[2])
            
            line = f"{rule}\t{src}\t{stats[0]}\t{stats[1]:.2f}\t{stats[2]:.2f}\t{stats[3]}\t{total_est}\t{safe_est}\t{over_blocks}\t{raw_cov:.4f}\t{safe_cov:.4f}\n"
            out_tsv.write(line)
            
            md_line = f"| {rule} | {src} | {total_est} | {safe_est} | {over_blocks} | {raw_cov:.4f} | {safe_cov:.4f} |\n"
            out_md.write(md_line)
            logger.info(md_line.strip())
            
    out_tsv.close()
    out_md.close()
    logger.info("Profiling reports saved.")

def main():
    logger.info("=== KAGGLE PROFILER STAGE ===")
    
    # 1. Check if we actually have the data
    try:
        files = check_files()
    except SystemExit:
        logger.error("Dataset not found. Please run this script in the Kaggle environment.")
        return

    # 2. Run real bounded ingestion
    run_real_ingestion(files)
    
    # 3. Verify counts
    verify_row_counts_and_norm()
    
    # 4. Profile blocks safely (no candidate generation)
    conn = duckdb.connect(DB_PATH)
    conn.execute("SET memory_limit = '20GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    
    # Build the keys inside DuckDB
    blocker = DiskBlocker(DB_PATH, TEMP_DIR, max_block_pairs=50_000)
    blocker.generate_exact_blocks()
    
    profile_advanced(conn)
    
    logger.info(f"FINAL PEAK RSS: {get_rss_gb():.3f} GB")
    conn.close()

if __name__ == "__main__":
    main()
