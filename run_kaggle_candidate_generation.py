import os
import gc
import sys
import time
import logging
import psutil
import duckdb
import pandas as pd

from src.config import cfg
from src.disk_blocking import DiskBlocker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "output/entity_resolution.duckdb"

def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)

def check_db():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database {DB_PATH} not found.")
        sys.exit(1)
    conn = duckdb.connect(DB_PATH)
    return conn

def evaluate_candidates(conn, src: str):
    logger.info(f"========== EVALUATING CANDIDATES {src.upper()} ==========")
    
    # Total unique candidates
    total_cands = conn.execute(f"SELECT COUNT(*) FROM candidates WHERE matched_source = '{src}'").fetchone()[0]
    
    # S1 entities represented
    s1_rep = conn.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM candidates WHERE matched_source = '{src}'").fetchone()[0]
    
    # Ground truth coverage
    total_true = conn.execute(f"SELECT COUNT(*) FROM ground_truth WHERE matched_entity_id LIKE '{src.upper()}-%'").fetchone()[0]
    
    if total_true > 0:
        retrieved_true = conn.execute(f"""
        SELECT COUNT(*)
        FROM ground_truth gt
        JOIN candidates c ON gt.source1_entity_id = c.source1_entity_id 
                         AND gt.matched_entity_id = c.matched_entity_id
        WHERE c.matched_source = '{src}'
        """).fetchone()[0]
        recall = retrieved_true / total_true
    else:
        retrieved_true = 0
        recall = 0.0
        
    total_s1 = conn.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    
    # Candidates per S1
    cands_per_s1 = total_cands / total_s1 if total_s1 > 0 else 0
    
    # S1 with zero candidates
    zero_cands = total_s1 - s1_rep
    
    # S1 with more than cap (if cap were applied)
    max_cap = cfg.MAX_CANDIDATES_PER_S1
    over_cap_query = f"""
    SELECT COUNT(*) FROM (
        SELECT source1_entity_id, COUNT(*) as cnt
        FROM candidates
        WHERE matched_source = '{src}'
        GROUP BY source1_entity_id
        HAVING cnt > {max_cap}
    )
    """
    over_cap = conn.execute(over_cap_query).fetchone()[0]
    
    logger.info(f"[1] Total unique candidate pairs: {total_cands}")
    logger.info(f"[2] Number of S1 entities represented: {s1_rep} / {total_s1}")
    logger.info(f"[3] Number of GT edges covered: {retrieved_true}")
    logger.info(f"[4] Candidate recall: {retrieved_true}/{total_true} ({recall*100:.4f}%)")
    logger.info(f"[5] Candidate pairs per S1 entity: {cands_per_s1:.2f}")
    logger.info(f"[6] Number of S1 entities with ZERO candidates: {zero_cands}")
    logger.info(f"[7] Number of S1 entities > cap ({max_cap}): {over_cap}")
    
    # Per-rule candidate counts
    r1_cnt = conn.execute(f"SELECT COUNT(*) FROM candidates WHERE matched_source = '{src}' AND from_exact_name = TRUE").fetchone()[0]
    r2_cnt = conn.execute(f"SELECT COUNT(*) FROM candidates WHERE matched_source = '{src}' AND from_exact_addr = TRUE").fetchone()[0]
    r3_cnt = conn.execute(f"SELECT COUNT(*) FROM candidates WHERE matched_source = '{src}' AND from_name_block = TRUE").fetchone()[0]
    logger.info(f"[10] Per-rule candidates: exact_name={r1_cnt}, exact_addr={r2_cnt}, prefix4_house={r3_cnt}")
    
    # Per-rule coverage
    if total_true > 0:
        r1_cov = conn.execute(f"""
        SELECT COUNT(*) FROM ground_truth gt JOIN candidates c ON gt.source1_entity_id = c.source1_entity_id AND gt.matched_entity_id = c.matched_entity_id
        WHERE c.matched_source = '{src}' AND c.from_exact_name = TRUE
        """).fetchone()[0]
        r2_cov = conn.execute(f"""
        SELECT COUNT(*) FROM ground_truth gt JOIN candidates c ON gt.source1_entity_id = c.source1_entity_id AND gt.matched_entity_id = c.matched_entity_id
        WHERE c.matched_source = '{src}' AND c.from_exact_addr = TRUE
        """).fetchone()[0]
        r3_cov = conn.execute(f"""
        SELECT COUNT(*) FROM ground_truth gt JOIN candidates c ON gt.source1_entity_id = c.source1_entity_id AND gt.matched_entity_id = c.matched_entity_id
        WHERE c.matched_source = '{src}' AND c.from_name_block = TRUE
        """).fetchone()[0]
        logger.info(f"[11] Per-rule coverage: exact_name={r1_cov} ({r1_cov/total_true*100:.2f}%), exact_addr={r2_cov} ({r2_cov/total_true*100:.2f}%), prefix4_house={r3_cov} ({r3_cov/total_true*100:.2f}%)")
        logger.info(f"[12] Union coverage: {retrieved_true} ({recall*100:.2f}%)")

def main():
    logger.info("=== Phase 3B Candidate Generation ===")
    start_time = time.time()
    
    conn = check_db()
    conn.execute("SET memory_limit = '10GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    os.makedirs("output/intermediate/tmp", exist_ok=True)
    
    # Clear any old candidates to ensure a fresh run
    conn.execute("DELETE FROM candidates")
    conn.execute("DROP TABLE IF EXISTS checkpoints")
    
    # Initialize disk blocker
    logger.info(f"Initializing DiskBlocker... (RSS: {get_rss_gb():.3f} GB)")
    blocker = DiskBlocker(DB_PATH, "output/intermediate/tmp", max_block_pairs=50_000)
    
    # 1. Generate blocks (extract keys for S1, S2, S3)
    blocker.generate_exact_blocks()
    
    rules = [
        ('exact_name', 'from_exact_name'),
        ('exact_addr', 'from_exact_addr'),
        ('prefix4_house', 'from_name_block')
    ]
    
    # 2. Candidate Generation for S2
    for r, flag in rules:
        blocker.generate_candidates(r, 's2', flag)
        gc.collect()
        logger.info(f"RSS after {r} -> s2: {get_rss_gb():.3f} GB")
        
    evaluate_candidates(conn, 's2')
    
    # 3. Candidate Generation for S3
    for r, flag in rules:
        blocker.generate_candidates(r, 's3', flag)
        gc.collect()
        logger.info(f"RSS after {r} -> s3: {get_rss_gb():.3f} GB")
        
    evaluate_candidates(conn, 's3')
    
    # Note: We are not explicitly capping candidates here, as this script is meant to evaluate
    # the un-capped performance of Phase 3B. However, the stats report how many S1s are over cap.
    # The actual full pipeline will cap candidates in the next stages.
    
    elapsed = time.time() - start_time
    logger.info(f"[8] Total Runtime: {elapsed:.2f}s")
    logger.info(f"[9] Peak RSS tracking via OS (Current RSS: {get_rss_gb():.3f} GB)")
    
    blocker.close()

if __name__ == "__main__":
    main()
