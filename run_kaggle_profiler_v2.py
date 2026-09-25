import os
import gc
import sys
import logging
import psutil
import pandas as pd
import duckdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "output/entity_resolution.duckdb"

def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)

def check_db():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database {DB_PATH} not found. Ensure ingestion was run successfully.")
        sys.exit(1)
        
    conn = duckdb.connect(DB_PATH)
    logger.info("Verifying Database Row Counts...")
    for src in ["s1", "s2", "s3", "ground_truth"]:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
        logger.info(f"Table {src}: {cnt} rows")
    return conn

def compute_rule_stats(conn, rule: str, src: str, max_pairs: int = 50_000):
    logger.info(f"--- Profiling Rule: {rule} against {src} ---")
    logger.info(f"Memory RSS Before: {get_rss_gb():.2f} GB")
    
    conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
    conn.execute("DROP TABLE IF EXISTS tmp_s2_cnt")
    conn.execute("DROP TABLE IF EXISTS tmp_s1_keys")
    conn.execute("DROP TABLE IF EXISTS tmp_s2_keys")
    conn.execute("DROP TABLE IF EXISTS tmp_true_hits")
    
    # 1. Compact count tables
    logger.info("Building compact aggregate counts...")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s1_cnt AS 
    SELECT block_key, COUNT(entity_id) as s1_size 
    FROM s1_blocks WHERE block_type = '{rule}' GROUP BY block_key
    """)
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s2_cnt AS 
    SELECT block_key, COUNT(entity_id) as s2_size 
    FROM {src}_blocks WHERE block_type = '{rule}' GROUP BY block_key
    """)
    
    # 2. Estimate pairs
    logger.info("Estimating pairs and oversized blocks...")
    stats_q = f"""
    SELECT 
        COUNT(t1.block_key) as num_distinct_keys,
        SUM(CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) as total_estimated_pairs,
        SUM(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) <= {max_pairs} 
                 THEN CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT) ELSE 0 END) as safe_estimated_pairs,
        COUNT(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) > {max_pairs} THEN 1 END) as oversized_blocks
    FROM tmp_s1_cnt t1
    JOIN tmp_s2_cnt t2 ON t1.block_key = t2.block_key
    """
    stats = conn.execute(stats_q).fetchone()
    
    # 3. Ground Truth Coverage
    logger.info("Computing Ground Truth Coverage...")
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s1_keys AS 
    SELECT entity_id, block_key FROM s1_blocks WHERE block_type = '{rule}'
    """)
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s2_keys AS 
    SELECT entity_id, block_key FROM {src}_blocks WHERE block_type = '{rule}'
    """)
    
    src_upper = src.upper()
    conn.execute(f"""
    CREATE TEMP TABLE tmp_true_hits AS
    SELECT gt.source1_entity_id, gt.matched_entity_id, t1.block_key
    FROM ground_truth gt
    JOIN tmp_s1_keys t1 ON gt.source1_entity_id = t1.entity_id
    JOIN tmp_s2_keys t2 ON gt.matched_entity_id = t2.entity_id 
    WHERE t1.block_key = t2.block_key AND gt.matched_entity_id LIKE '{src_upper}-%'
    """)
    
    raw_hits = conn.execute("SELECT COUNT(DISTINCT source1_entity_id || matched_entity_id) FROM tmp_true_hits").fetchone()[0]
    
    safe_hits_q = f"""
    SELECT COUNT(DISTINCT h.source1_entity_id || h.matched_entity_id)
    FROM tmp_true_hits h
    JOIN tmp_s1_cnt c1 ON h.block_key = c1.block_key
    JOIN tmp_s2_cnt c2 ON h.block_key = c2.block_key
    WHERE (CAST(c1.s1_size AS BIGINT) * CAST(c2.s2_size AS BIGINT)) <= {max_pairs}
    """
    safe_hits = conn.execute(safe_hits_q).fetchone()[0]
    
    total_true = conn.execute(f"SELECT COUNT(*) FROM ground_truth WHERE matched_entity_id LIKE '{src_upper}-%'").fetchone()[0]
    
    # Cleanup
    conn.execute("DROP TABLE tmp_s1_cnt")
    conn.execute("DROP TABLE tmp_s2_cnt")
    conn.execute("DROP TABLE tmp_s1_keys")
    conn.execute("DROP TABLE tmp_s2_keys")
    conn.execute("DROP TABLE tmp_true_hits")
    gc.collect()
    
    return {
        "rule": rule,
        "source": src,
        "estimated_pairs": stats[1] or 0,
        "safe_estimated_pairs": stats[2] or 0,
        "oversized_blocks": stats[3] or 0,
        "raw_hits": raw_hits,
        "safe_hits": safe_hits,
        "total_true": total_true,
        "raw_cov": raw_hits / max(1, total_true),
        "safe_cov": safe_hits / max(1, total_true)
    }

def compute_combined_stats(conn, combo_name: str, src: str, s1_key_expr: str, src_key_expr: str, max_pairs: int = 50_000):
    logger.info(f"--- Profiling Combination: {combo_name} against {src} ---")
    
    conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
    conn.execute("DROP TABLE IF EXISTS tmp_s2_cnt")
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s1_cnt AS 
    SELECT ({s1_key_expr}) as block_key, COUNT(entity_id) as s1_size 
    FROM s1 GROUP BY 1
    """)
    
    conn.execute(f"""
    CREATE TEMP TABLE tmp_s2_cnt AS 
    SELECT ({src_key_expr}) as block_key, COUNT(entity_id) as s2_size 
    FROM {src} GROUP BY 1
    """)
    
    stats_q = f"""
    SELECT 
        SUM(CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) as total_estimated_pairs,
        SUM(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) <= {max_pairs} 
                 THEN CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT) ELSE 0 END) as safe_estimated_pairs,
        COUNT(CASE WHEN (CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT)) > {max_pairs} THEN 1 END) as oversized_blocks
    FROM tmp_s1_cnt t1
    JOIN tmp_s2_cnt t2 ON t1.block_key = t2.block_key
    WHERE t1.block_key IS NOT NULL AND t1.block_key != ''
    """
    stats = conn.execute(stats_q).fetchone()
    
    conn.execute("DROP TABLE IF EXISTS tmp_true_hits")
    
    src_upper = src.upper()
    conn.execute(f"""
    CREATE TEMP TABLE tmp_true_hits AS
    SELECT gt.source1_entity_id, gt.matched_entity_id, ({s1_key_expr}) as block_key
    FROM ground_truth gt
    JOIN s1 ON gt.source1_entity_id = s1.entity_id
    JOIN {src} s2 ON gt.matched_entity_id = s2.entity_id 
    WHERE ({s1_key_expr}) = ({src_key_expr}) 
      AND ({s1_key_expr}) IS NOT NULL 
      AND ({s1_key_expr}) != ''
      AND gt.matched_entity_id LIKE '{src_upper}-%'
    """)
    
    raw_hits = conn.execute("SELECT COUNT(DISTINCT source1_entity_id || matched_entity_id) FROM tmp_true_hits").fetchone()[0]
    
    safe_hits_q = f"""
    SELECT COUNT(DISTINCT h.source1_entity_id || h.matched_entity_id)
    FROM tmp_true_hits h
    JOIN tmp_s1_cnt c1 ON h.block_key = c1.block_key
    JOIN tmp_s2_cnt c2 ON h.block_key = c2.block_key
    WHERE (CAST(c1.s1_size AS BIGINT) * CAST(c2.s2_size AS BIGINT)) <= {max_pairs}
    """
    safe_hits = conn.execute(safe_hits_q).fetchone()[0]
    
    total_true = conn.execute(f"SELECT COUNT(*) FROM ground_truth WHERE matched_entity_id LIKE '{src_upper}-%'").fetchone()[0]
    
    conn.execute("DROP TABLE tmp_s1_cnt")
    conn.execute("DROP TABLE tmp_s2_cnt")
    conn.execute("DROP TABLE tmp_true_hits")
    gc.collect()
    
    return {
        "rule": combo_name,
        "source": src,
        "estimated_pairs": stats[0] or 0,
        "safe_estimated_pairs": stats[1] or 0,
        "oversized_blocks": stats[2] or 0,
        "raw_cov": raw_hits / max(1, total_true),
        "safe_cov": safe_hits / max(1, total_true)
    }

def main():
    conn = check_db()
    conn.execute("SET memory_limit = '20GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    
    os.makedirs("output/intermediate", exist_ok=True)
    out_tsv = "output/intermediate/real_block_profile.tsv"
    
    # Hardcoded exact results that were already successful
    already_computed = [
        {"rule": "exact_name", "source": "s2", "estimated_pairs": 11597631, "safe_estimated_pairs": 11597631, "oversized_blocks": 0, "raw_cov": 0.2513, "safe_cov": 0.2513},
        {"rule": "exact_name", "source": "s3", "estimated_pairs": 12892967, "safe_estimated_pairs": 12892967, "oversized_blocks": 0, "raw_cov": 0.2612, "safe_cov": 0.2612},
        {"rule": "exact_addr", "source": "s2", "estimated_pairs": 564631, "safe_estimated_pairs": 564631, "oversized_blocks": 0, "raw_cov": 0.1249, "safe_cov": 0.1249},
        {"rule": "exact_addr", "source": "s3", "estimated_pairs": 206260, "safe_estimated_pairs": 206260, "oversized_blocks": 0, "raw_cov": 0.0434, "safe_cov": 0.0434}
    ]
    
    with open(out_tsv, "w") as f:
        f.write("Rule\tSource\tEstimated_Pairs\tSafe_Pairs\tOversized_Blocks\tRaw_Coverage\tSafe_Coverage\n")
        for res in already_computed:
            f.write(f"{res['rule']}\t{res['source']}\t{res['estimated_pairs']}\t{res['safe_estimated_pairs']}\t{res['oversized_blocks']}\t{res['raw_cov']:.4f}\t{res['safe_cov']:.4f}\n")

    rules_to_run = ["name_prefix_3", "name_prefix_4", "addr_prefix_3", "addr_house_num"]
    
    for rule in rules_to_run:
        for src in ["s2", "s3"]:
            try:
                res = compute_rule_stats(conn, rule, src)
                with open(out_tsv, "a") as f:
                    f.write(f"{rule}\t{src}\t{res['estimated_pairs']}\t{res['safe_estimated_pairs']}\t{res['oversized_blocks']}\t{res['raw_cov']:.4f}\t{res['safe_cov']:.4f}\n")
            except Exception as e:
                logger.error(f"Failed to profile {rule} on {src}: {e}")
                
    combinations = [
        ("exact_name + name_length_bucket", "s1.name_norm || '_' || CAST(LENGTH(s1.name_norm)/5 AS VARCHAR)", "s2.name_norm || '_' || CAST(LENGTH(s2.name_norm)/5 AS VARCHAR)"),
        ("exact_name + addr_prefix_3", "s1.name_norm || '_' || SUBSTRING(s1.addr_norm, 1, 3)", "s2.name_norm || '_' || SUBSTRING(s2.addr_norm, 1, 3)"),
        ("exact_name + address_length_bucket", "s1.name_norm || '_' || CAST(LENGTH(s1.addr_norm)/10 AS VARCHAR)", "s2.name_norm || '_' || CAST(LENGTH(s2.addr_norm)/10 AS VARCHAR)"),
        ("exact_addr + name_prefix_3", "s1.addr_norm || '_' || SUBSTRING(s1.name_norm, 1, 3)", "s2.addr_norm || '_' || SUBSTRING(s2.name_norm, 1, 3)"),
        ("exact_addr + name_length_bucket", "s1.addr_norm || '_' || CAST(LENGTH(s1.name_norm)/5 AS VARCHAR)", "s2.addr_norm || '_' || CAST(LENGTH(s2.name_norm)/5 AS VARCHAR)")
    ]
    
    for c_name, s1_expr, src_expr in combinations:
        for src in ["s2", "s3"]:
            try:
                actual_src_expr = src_expr.replace("s2.", f"{src}.")
                res = compute_combined_stats(conn, c_name, src, s1_expr, actual_src_expr)
                with open(out_tsv, "a") as f:
                    f.write(f"{res['rule']}\t{res['source']}\t{res['estimated_pairs']}\t{res['safe_estimated_pairs']}\t{res['oversized_blocks']}\t{res['raw_cov']:.4f}\t{res['safe_cov']:.4f}\n")
            except Exception as e:
                logger.error(f"Failed to profile combo {c_name} on {src}: {e}")
                
    logger.info("Profiling Complete. Saved to TSV.")
    conn.close()

if __name__ == "__main__":
    main()
