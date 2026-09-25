import os
import argparse
import logging
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

def prepare_profiler():
    logger.info("Initializing DuckDB Profiler...")
    conn = duckdb.connect("output/entity_resolution.duckdb")
    
    # Enable safe memory
    conn.execute("SET memory_limit = '20GB'")
    conn.execute("PRAGMA temp_directory='output/intermediate/tmp'")
    
    return conn

def profile_blocks(conn, block_type: str, matched_source: str, max_block_pairs: int = 50_000):
    logger.info(f"Profiling block '{block_type}' on '{matched_source}'")
    
    query = f"""
    WITH block_stats AS (
        SELECT 
            t1.block_key,
            COUNT(DISTINCT t1.entity_id) as s1_size,
            COUNT(DISTINCT t2.entity_id) as s2_size,
            CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) as estimated_pairs
        FROM s1_blocks t1
        JOIN {matched_source}_blocks t2 
          ON t1.block_key = t2.block_key 
         AND t1.block_type = '{block_type}' 
         AND t2.block_type = '{block_type}'
        GROUP BY t1.block_key
    )
    SELECT 
        COUNT(block_key) as num_distinct_keys,
        AVG(s1_size) as mean_s1_size,
        QUANTILE_CONT(s1_size, 0.5) as median_s1_size,
        QUANTILE_CONT(s1_size, 0.95) as p95_s1_size,
        QUANTILE_CONT(s1_size, 0.99) as p99_s1_size,
        MAX(s1_size) as max_s1_size,
        SUM(estimated_pairs) as total_estimated_pairs,
        SUM(CASE WHEN estimated_pairs <= {max_block_pairs} THEN estimated_pairs ELSE 0 END) as safe_estimated_pairs,
        SUM(CASE WHEN estimated_pairs > {max_block_pairs} THEN estimated_pairs ELSE 0 END) as oversized_estimated_pairs,
        COUNT(CASE WHEN estimated_pairs <= {max_block_pairs} THEN 1 END) as safe_blocks,
        COUNT(CASE WHEN estimated_pairs > {max_block_pairs} THEN 1 END) as oversized_blocks
    FROM block_stats
    """
    stats = conn.execute(query).fetchone()
    
    # True match coverage query
    coverage_query = f"""
    WITH true_pairs AS (
        SELECT gt.source1_entity_id, gt.matched_entity_id 
        FROM ground_truth gt
        WHERE gt.matched_entity_id LIKE '{matched_source.upper()}-%'
    ),
    raw_coverage AS (
        SELECT COUNT(DISTINCT tp.source1_entity_id || tp.matched_entity_id) as raw_hits
        FROM true_pairs tp
        JOIN s1_blocks s1 ON tp.source1_entity_id = s1.entity_id AND s1.block_type = '{block_type}'
        JOIN {matched_source}_blocks s2 ON tp.matched_entity_id = s2.entity_id AND s2.block_type = '{block_type}'
        WHERE s1.block_key = s2.block_key
    ),
    safe_keys AS (
        SELECT t1.block_key
        FROM s1_blocks t1
        JOIN {matched_source}_blocks t2 
          ON t1.block_key = t2.block_key 
         AND t1.block_type = '{block_type}' 
         AND t2.block_type = '{block_type}'
        GROUP BY t1.block_key
        HAVING CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) <= {max_block_pairs}
    ),
    safe_coverage AS (
        SELECT COUNT(DISTINCT tp.source1_entity_id || tp.matched_entity_id) as safe_hits
        FROM true_pairs tp
        JOIN s1_blocks s1 ON tp.source1_entity_id = s1.entity_id AND s1.block_type = '{block_type}'
        JOIN {matched_source}_blocks s2 ON tp.matched_entity_id = s2.entity_id AND s2.block_type = '{block_type}'
        JOIN safe_keys sk ON s1.block_key = sk.block_key
        WHERE s1.block_key = s2.block_key
    )
    SELECT 
        (SELECT raw_hits FROM raw_coverage) as raw_hits,
        (SELECT safe_hits FROM safe_coverage) as safe_hits,
        (SELECT COUNT(*) FROM true_pairs) as total_true_pairs
    """
    cov_stats = conn.execute(coverage_query).fetchone()
    
    return {
        "num_distinct_keys": stats[0],
        "mean_s1_size": stats[1],
        "median_s1_size": stats[2],
        "p95_s1_size": stats[3],
        "p99_s1_size": stats[4],
        "max_s1_size": stats[5],
        "total_estimated_pairs": stats[6] or 0,
        "safe_estimated_pairs": stats[7] or 0,
        "oversized_estimated_pairs": stats[8] or 0,
        "safe_blocks": stats[9] or 0,
        "oversized_blocks": stats[10] or 0,
        "raw_true_coverage": cov_stats[0] / max(1, cov_stats[2]),
        "safe_true_coverage": cov_stats[1] / max(1, cov_stats[2]),
        "raw_hits": cov_stats[0],
        "safe_hits": cov_stats[1],
        "total_true_pairs": cov_stats[2]
    }

def run_profiling():
    conn = prepare_profiler()
    
    rules = [
        "exact_name",
        "exact_addr",
        "name_prefix_3",
        "name_prefix_4",
        "addr_prefix_3",
        "addr_house_num"
    ]
    
    print("| Blocking Rule | Source | Estimated Pairs | Safe Pairs | Oversized Blocks | Raw Coverage | Safe Coverage |")
    print("|---|---|---|---|---|---|---|")
    
    for rule in rules:
        for src in ["s2", "s3"]:
            stats = profile_blocks(conn, rule, src)
            print(f"| {rule} | {src} | {stats['total_estimated_pairs']} | {stats['safe_estimated_pairs']} | {stats['oversized_blocks']} | {stats['raw_true_coverage']:.4f} | {stats['safe_true_coverage']:.4f} |")

    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_profiling()
