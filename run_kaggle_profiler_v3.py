import os
import gc
import sys
import time
import logging
import psutil
import duckdb

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

def profile_v3_poc(conn, max_pairs: int = 50_000):
    rule = "name_prefix_3 + name_length_bucket"
    src = "s2"
    
    # This is a profiling-only experiment to determine whether adding 
    # a coarse name length bucket (LENGTH / 5) improves the selectivity 
    # of name_prefix_3 while retaining acceptable ground-truth coverage.
    
    logger.info(f"--- V3 Bounded Profiling POC: {rule} against {src} ---")
    start_time = time.time()
    
    # Bucket definition: length divided by 5 (e.g., len 1-4 -> 0, 5-9 -> 1, etc.)
    key_expr = "SUBSTRING(name_norm, 1, 3) || '_' || CAST(LENGTH(name_norm)/5 AS VARCHAR)"
    
    # 1. Bounded Group By using 64-bit integer hashes
    logger.info("Computing 64-bit integer hashes for s1 counts...")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s1_cnt AS 
    SELECT hash({key_expr}) as block_key, COUNT(*) as s1_size 
    FROM s1 
    WHERE name_norm IS NOT NULL AND name_norm != ''
    GROUP BY 1
    """)
    
    logger.info(f"Computing 64-bit integer hashes for {src} counts...")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s2_cnt AS 
    SELECT hash({key_expr}) as block_key, COUNT(*) as s2_size 
    FROM {src} 
    WHERE name_norm IS NOT NULL AND name_norm != ''
    GROUP BY 1
    """)
    
    # 2. Join the tiny compact count tables
    logger.info("Estimating pair volumes...")
    stats_q = f"""
    SELECT 
        SUM(CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) as total_estimated_pairs,
        SUM(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) <= {max_pairs} 
                 THEN CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT) ELSE 0 END) as safe_estimated_pairs,
        COUNT(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) > {max_pairs} THEN 1 END) as oversized_blocks
    FROM tmp_s1_cnt t1
    JOIN tmp_s2_cnt t2 ON t1.block_key = t2.block_key
    """
    stats = conn.execute(stats_q).fetchone()
    total_est = stats[0] or 0
    safe_est = stats[1] or 0
    oversized = stats[2] or 0
    logger.info(f"Estimated pairs: {total_est} | Oversized blocks: {oversized}")
    
    # 3. Ground Truth Coverage via Bounded Chunks
    src_upper = src.upper()
    total_true = conn.execute(f"SELECT COUNT(*) FROM ground_truth WHERE matched_entity_id LIKE '{src_upper}-%'").fetchone()[0]
    
    logger.info(f"Computing GT coverage over {total_true} edges in chunks...")
    
    chunk_size = 250_000
    offset = 0
    raw_hits = 0
    safe_hits = 0
    
    while offset < total_true:
        chunk_start = time.time()
        conn.execute(f"""
        CREATE TEMP TABLE tmp_gt_chunk AS 
        SELECT source1_entity_id, matched_entity_id
        FROM ground_truth
        WHERE matched_entity_id LIKE '{src_upper}-%'
        ORDER BY source1_entity_id, matched_entity_id
        LIMIT {chunk_size} OFFSET {offset}
        """)
        
        # Calculate raw hits in chunk
        hits = conn.execute(f"""
        SELECT COUNT(DISTINCT gt.source1_entity_id || gt.matched_entity_id)
        FROM tmp_gt_chunk gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN {src} s2 ON gt.matched_entity_id = s2.entity_id
        WHERE hash({key_expr.replace('name_norm', 's1.name_norm')}) = hash({key_expr.replace('name_norm', 's2.name_norm')})
        """).fetchone()[0]
        raw_hits += hits
        
        # Calculate safe hits in chunk
        s_hits = conn.execute(f"""
        SELECT COUNT(DISTINCT gt.source1_entity_id || gt.matched_entity_id)
        FROM tmp_gt_chunk gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN {src} s2 ON gt.matched_entity_id = s2.entity_id
        JOIN tmp_s1_cnt c1 ON hash({key_expr.replace('name_norm', 's1.name_norm')}) = c1.block_key
        JOIN tmp_s2_cnt c2 ON hash({key_expr.replace('name_norm', 's2.name_norm')}) = c2.block_key
        WHERE hash({key_expr.replace('name_norm', 's1.name_norm')}) = hash({key_expr.replace('name_norm', 's2.name_norm')})
          AND (CAST(c1.s1_size AS BIGINT) * CAST(c2.s2_size AS BIGINT)) <= {max_pairs}
        """).fetchone()[0]
        safe_hits += s_hits
        
        conn.execute("DROP TABLE tmp_gt_chunk")
        
        chunk_elapsed = time.time() - chunk_start
        offset += chunk_size
        logger.info(f"Chunk [{offset}/{total_true}] | Hits: {hits} | Peak RSS: {get_rss_gb():.3f} GB | Time: {chunk_elapsed:.2f}s")
        gc.collect()

    conn.execute("DROP TABLE tmp_s1_cnt")
    conn.execute("DROP TABLE tmp_s2_cnt")
    
    elapsed = time.time() - start_time
    logger.info("--- POC COMPLETION ---")
    logger.info(f"Estimated pairs: {total_est}")
    logger.info(f"Raw True Coverage: {raw_hits / max(1, total_true):.4f} ({raw_hits}/{total_true})")
    logger.info(f"Safe True Coverage: {safe_hits / max(1, total_true):.4f} ({safe_hits}/{total_true})")
    logger.info(f"Elapsed Time: {elapsed:.2f}s")
    logger.info(f"Final RSS: {get_rss_gb():.3f} GB")

def main():
    conn = check_db()
    # Safest limits possible, trusting the integer chunking architecture
    conn.execute("SET memory_limit = '10GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    os.makedirs("output/intermediate/tmp", exist_ok=True)
    
    try:
        profile_v3_poc(conn)
    except Exception as e:
        logger.error(f"V3 POC Failed: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
