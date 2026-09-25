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

def compute_rule_counts(conn, rule_name: str, key_expr: str, src: str, max_pairs: int = 50_000):
    logger.info(f"--- Computing Bounded Hashes & Pair Volumes for {rule_name} ---")
    conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
    conn.execute("DROP TABLE IF EXISTS tmp_src_cnt")
    gc.collect()
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s1_cnt AS 
    SELECT hash({key_expr}) as block_key, COUNT(*) as s1_size 
    FROM s1 
    WHERE {key_expr} IS NOT NULL 
    GROUP BY 1
    """)
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_src_cnt AS 
    SELECT hash({key_expr}) as block_key, COUNT(*) as s2_size 
    FROM {src} 
    WHERE {key_expr} IS NOT NULL
    GROUP BY 1
    """)
    
    stats_q = f"""
    SELECT 
        SUM(CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) as total_estimated_pairs,
        SUM(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) <= {max_pairs} 
                 THEN CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT) ELSE 0 END) as safe_estimated_pairs,
        COUNT(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) > {max_pairs} THEN 1 END) as oversized_blocks
    FROM tmp_s1_cnt t1
    JOIN tmp_src_cnt t2 ON t1.block_key = t2.block_key
    """
    stats = conn.execute(stats_q).fetchone()
    
    return {
        "rule": rule_name,
        "total_est": stats[0] or 0,
        "safe_est": stats[1] or 0,
        "oversized": stats[2] or 0
    }

def update_gt_flags(conn, rule_idx: int, key_expr: str, src: str, max_pairs: int = 50_000):
    logger.info(f"Updating GT coverage flags for Rule {rule_idx} in chunks...")
    
    src_upper = src.upper()
    total_true = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags").fetchone()[0]
    
    chunk_size = 250_000
    offset = 0
    
    while offset < total_true:
        chunk_start = time.time()
        
        # We join the current chunk of GT with S1 and S2, evaluate the hash, and update the flags.
        # This requires matching rowids or just using the unique pair (source1_entity_id, matched_entity_id)
        
        s1_expr = key_expr.replace('name_norm', 's1.name_norm').replace('addr_norm', 's1.addr_norm')
        s2_expr = key_expr.replace('name_norm', f'{src}.name_norm').replace('addr_norm', f'{src}.addr_norm')
        
        update_q = f"""
        UPDATE tmp_gt_flags
        SET 
            r{rule_idx}_raw = TRUE,
            r{rule_idx}_safe = (CAST(c1.s1_size AS BIGINT) * CAST(c2.s2_size AS BIGINT) <= {max_pairs})
        FROM (
            SELECT gt.source1_entity_id, gt.matched_entity_id, s1.entity_id as s1_id, {src}.entity_id as s2_id,
                   c1.s1_size, c2.s2_size
            FROM (SELECT * FROM tmp_gt_flags ORDER BY source1_entity_id, matched_entity_id LIMIT {chunk_size} OFFSET {offset}) gt
            JOIN s1 ON gt.source1_entity_id = s1.entity_id
            JOIN {src} ON gt.matched_entity_id = {src}.entity_id
            JOIN tmp_s1_cnt c1 ON hash({s1_expr}) = c1.block_key
            JOIN tmp_src_cnt c2 ON hash({s2_expr}) = c2.block_key
            WHERE hash({s1_expr}) = hash({s2_expr})
        ) sub
        WHERE tmp_gt_flags.source1_entity_id = sub.source1_entity_id
          AND tmp_gt_flags.matched_entity_id = sub.matched_entity_id
        """
        conn.execute(update_q)
        
        chunk_elapsed = time.time() - chunk_start
        offset += chunk_size
        logger.info(f"Chunk [{min(offset, total_true)}/{total_true}] | RSS: {get_rss_gb():.3f} GB | Time: {chunk_elapsed:.2f}s")
        gc.collect()

def profile_union_poc(conn, max_pairs: int = 50_000):
    src = "s2"
    logger.info(f"=== V3 Bounded Union Profiling POC against {src} ===")
    start_time = time.time()
    
    rules = [
        ("exact_name", "name_norm"),
        ("exact_addr", "addr_norm"),
        ("name_prefix_4 + addr_house_num", "SUBSTRING(name_norm, 1, 4) || '_' || split_part(addr_norm, ' ', 1)")
    ]
    
    # Initialize the GT flags table
    logger.info("Initializing GT coverage tracking table...")
    conn.execute("DROP TABLE IF EXISTS tmp_gt_flags")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_gt_flags AS
    SELECT source1_entity_id, matched_entity_id, 
           FALSE as r1_raw, FALSE as r1_safe,
           FALSE as r2_raw, FALSE as r2_safe,
           FALSE as r3_raw, FALSE as r3_safe
    FROM ground_truth
    WHERE matched_entity_id LIKE '{src.upper()}-%'
    """)
    
    pair_volumes = []
    
    # Process each rule completely isolated to bounded memory
    for i, (r_name, r_expr) in enumerate(rules, 1):
        stats = compute_rule_counts(conn, r_name, r_expr, src, max_pairs)
        pair_volumes.append(stats)
        
        update_gt_flags(conn, i, r_expr, src, max_pairs)
        
        conn.execute("DROP TABLE tmp_s1_cnt")
        conn.execute("DROP TABLE tmp_src_cnt")
        gc.collect()
        
    logger.info("=== Coverage & Overlap Analysis ===")
    
    total_edges = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags").fetchone()[0]
    
    # Raw individual coverages
    r1_safe = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe").fetchone()[0]
    r2_safe = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r2_safe").fetchone()[0]
    r3_safe = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r3_safe").fetchone()[0]
    
    logger.info(f"[1] exact_name safe coverage: {r1_safe} / {total_edges} ({r1_safe/total_edges:.4f})")
    logger.info(f"[2] exact_addr safe coverage: {r2_safe} / {total_edges} ({r2_safe/total_edges:.4f})")
    logger.info(f"[3] prefix4+house safe coverage: {r3_safe} / {total_edges} ({r3_safe/total_edges:.4f})")
    
    # Unions
    u12 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe OR r2_safe").fetchone()[0]
    u123 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe OR r2_safe OR r3_safe").fetchone()[0]
    
    logger.info(f"Union (1 U 2) safe coverage: {u12} / {total_edges} ({u12/total_edges:.4f})")
    logger.info(f"Union (1 U 2 U 3) safe coverage: {u123} / {total_edges} ({u123/total_edges:.4f})")
    
    incremental = u123 - u12
    logger.info(f"Incremental edges from Rule 3: {incremental} ({(incremental/total_edges)*100:.2f} percentage points)")
    
    # Overlaps
    o12 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe AND r2_safe").fetchone()[0]
    o13 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe AND r3_safe").fetchone()[0]
    o23 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r2_safe AND r3_safe").fetchone()[0]
    
    logger.info(f"Overlap (1 AND 2): {o12} edges")
    logger.info(f"Overlap (1 AND 3): {o13} edges")
    logger.info(f"Overlap (2 AND 3): {o23} edges")
    
    # Candidate Pair Volumes
    logger.info("=== Candidate Volume Estimates ===")
    sum_pairs = sum(s["safe_est"] for s in pair_volumes)
    logger.info("Individual Safe Pair Estimates:")
    for s in pair_volumes:
        logger.info(f"  {s['rule']}: {s['safe_est']} pairs (Oversized: {s['oversized']})")
        
    logger.info(f"Upper Bound Union Pair Volume (Sum): {sum_pairs}")
    logger.info("Exact pair overlap cannot be computed without materializing the Cartesian product pairs, "
                "which is prohibited by safety constraints. The True Union Volume <= Upper Bound.")
                
    elapsed = time.time() - start_time
    logger.info(f"Elapsed Time: {elapsed:.2f}s")
    logger.info(f"Final RSS: {get_rss_gb():.3f} GB")
    
    conn.execute("DROP TABLE tmp_gt_flags")

def main():
    conn = check_db()
    conn.execute("SET memory_limit = '10GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    os.makedirs("output/intermediate/tmp", exist_ok=True)
    
    try:
        profile_union_poc(conn)
    except Exception as e:
        logger.error(f"V3 POC Failed: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
