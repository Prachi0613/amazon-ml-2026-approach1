import os
import gc
import sys
import time
import logging
import psutil
import duckdb
import pandas as pd

from src.config import cfg
from src.blocking import _build_ngram_index, _retrieve_ngram_pass

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

def evaluate_source(conn, src: str, max_pairs: int = 50_000):
    logger.info(f"========== EVALUATING {src.upper()} ==========")
    
    # 1. Initialize GT Flags Table
    logger.info("Initializing GT coverage tracking table...")
    conn.execute("DROP TABLE IF EXISTS tmp_gt_flags")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_gt_flags AS
    SELECT source1_entity_id, matched_entity_id, 
           FALSE as r1_safe,
           FALSE as r2_safe,
           FALSE as r3_safe,
           FALSE as r4_safe
    FROM ground_truth
    WHERE matched_entity_id LIKE '{src.upper()}-%'
    """)
    
    rules = [
        ("exact_name", "name_norm"),
        ("exact_addr", "addr_norm"),
        ("name_prefix_4 + addr_house_num", "SUBSTRING(name_norm, 1, 4) || '_' || split_part(addr_norm, ' ', 1)")
    ]
    
    total_edges = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags").fetchone()[0]
    
    # 2. Compute Existing Three-Pass Rules
    for i, (r_name, key_expr) in enumerate(rules, 1):
        logger.info(f"Computing rule {i}: {r_name}")
        conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
        conn.execute("DROP TABLE IF EXISTS tmp_src_cnt")
        
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
        
        # Update GT flags
        s1_expr = key_expr.replace('name_norm', 's1.name_norm').replace('addr_norm', 's1.addr_norm')
        s2_expr = key_expr.replace('name_norm', f'{src}.name_norm').replace('addr_norm', f'{src}.addr_norm')
        
        chunk_size = 250_000
        offset = 0
        while offset < total_edges:
            update_q = f"""
            UPDATE tmp_gt_flags
            SET r{i}_safe = (CAST(sub.s1_size AS BIGINT) * CAST(sub.s2_size AS BIGINT) <= {max_pairs})
            FROM (
                SELECT gt.source1_entity_id, gt.matched_entity_id,
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
            offset += chunk_size
            gc.collect()
            
    conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
    conn.execute("DROP TABLE IF EXISTS tmp_src_cnt")
    gc.collect()

    # 3. N-Gram Profiling
    logger.info(f"--- N-Gram Profiling ({src}) ---")
    start_time = time.time()
    
    # Load required data into memory
    logger.info("Loading entity IDs and name_norm for N-Gram...")
    s1_df = conn.execute("SELECT entity_id, name_norm FROM s1").fetchdf()
    src_df = conn.execute(f"SELECT entity_id, name_norm FROM {src}").fetchdf()
    empty_df = pd.DataFrame(columns=['entity_id', 'name_norm'])
    
    if src == 's2':
        s2_arg = src_df
        s3_arg = empty_df
    else:
        s2_arg = empty_df
        s3_arg = src_df
        
    logger.info("Building N-gram index...")
    vect, X_corp, corp_ids, corp_srcs = _build_ngram_index(s2_arg, s3_arg, cfg.COL_NAME_NORM, cfg)
    
    n_s1 = len(s1_df)
    logger.info(f"Total S1 rows evaluated: {n_s1}")
    logger.info(f"TOP_K used: {cfg.TOP_K_NAME}")
    
    chunk_size = cfg.NGRAM_CHUNK_SIZE
    total_retrieved = 0
    
    for chunk_start in range(0, n_s1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_s1)
        s1_chunk = s1_df.iloc[chunk_start:chunk_end].copy()
        
        # Retrieve candidates for this chunk
        cands = _retrieve_ngram_pass(
            s1_chunk, vect, X_corp, corp_ids, corp_srcs,
            field=cfg.COL_NAME_NORM, top_k=cfg.TOP_K_NAME, min_sim=cfg.MIN_NGRAM_SIMILARITY_NAME,
            score_col="score", pass_col="flag", config=cfg
        )
        
        if not cands.empty:
            # Register them to DuckDB and update coverage
            conn.register("tmp_cands", cands)
            
            if chunk_start == 0:
                logger.info(f"--- DIAGNOSTICS (First Chunk) ---")
                n_cands = len(cands)
                logger.info(f"Retrieved candidate pairs: {n_cands}")
                u_target = cands['matched_entity_id'].nunique()
                logger.info(f"Unique target entity IDs: {u_target}")
                
                # Check if target ID exists in target table
                t_exist = conn.execute(f"SELECT COUNT(*) FROM tmp_cands c JOIN {src} s ON c.matched_entity_id = s.entity_id").fetchone()[0]
                logger.info(f"Retrieved pairs whose target ID exists in {src}: {t_exist}")
                
                # Check if pair exists in ground truth
                gt_match = conn.execute(f"SELECT COUNT(*) FROM tmp_cands c JOIN tmp_gt_flags gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id").fetchone()[0]
                logger.info(f"Retrieved pairs existing in ground truth: {gt_match}")
                
                if gt_match > 0:
                    sample = conn.execute(f"SELECT c.source1_entity_id, c.matched_entity_id FROM tmp_cands c JOIN tmp_gt_flags gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id LIMIT 1").fetchone()
                    logger.info(f"Sample matching pair: {sample}")
                else:
                    logger.info("No matching pairs found in first chunk.")
                logger.info(f"--------------------------------")
            
            conn.execute("""
            UPDATE tmp_gt_flags
            SET r4_safe = TRUE
            FROM tmp_cands c
            WHERE tmp_gt_flags.source1_entity_id = c.source1_entity_id
              AND tmp_gt_flags.matched_entity_id = c.matched_entity_id
            """)
            conn.unregister("tmp_cands")
            total_retrieved += len(cands)
            
        del s1_chunk, cands
        if chunk_start > 0 and chunk_start % 500_000 < chunk_size:
            logger.info(f"N-Gram Chunk [{chunk_start}/{n_s1}] | RSS: {get_rss_gb():.3f} GB")
            gc.collect()
            
    elapsed = time.time() - start_time
    logger.info(f"Total N-Gram pairs retrieved: {total_retrieved}")
    
    # 4. Coverage Analysis
    logger.info("=== Coverage Analysis ===")
    
    # Baseline 3-Pass
    r123 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe OR r2_safe OR r3_safe").fetchone()[0]
    
    # N-Gram only
    r4 = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r4_safe").fetchone()[0]
    
    # 3-Pass + N-Gram
    union_all = conn.execute("SELECT COUNT(*) FROM tmp_gt_flags WHERE r1_safe OR r2_safe OR r3_safe OR r4_safe").fetchone()[0]
    
    incremental = union_all - r123
    
    logger.info(f"[A] Existing 3-Pass coverage: {r123} / {total_edges} ({r123/total_edges*100:.2f}%)")
    logger.info(f"[B] N-Gram only coverage: {r4} / {total_edges} ({r4/total_edges*100:.2f}%)")
    logger.info(f"    - N-Gram GT Edges Retrieved: {r4}")
    logger.info(f"    - N-Gram GT Edges Missed: {total_edges - r4}")
    logger.info(f"[C] 3-Pass UNION N-Gram coverage: {union_all} / {total_edges} ({union_all/total_edges*100:.2f}%)")
    logger.info(f"[D] Incremental edges from N-Gram: {incremental}")
    logger.info(f"[E] Incremental recall points: {(incremental/total_edges)*100:.2f}")
    
    logger.info(f"Runtime: {elapsed:.2f}s | Peak RSS tracking via OS")
    
    conn.execute("DROP TABLE tmp_gt_flags")
    del s1_df, src_df, empty_df, s2_arg, s3_arg, vect, X_corp, corp_ids, corp_srcs
    gc.collect()

def main():
    conn = check_db()
    conn.execute("SET memory_limit = '10GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    os.makedirs("output/intermediate/tmp", exist_ok=True)
    
    try:
        evaluate_source(conn, "s2")
        evaluate_source(conn, "s3")
    except Exception as e:
        logger.error(f"N-Gram Profiler Failed: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
