import os
import argparse
import logging
from src.config import cfg
from src.storage import DiskStorage
from src.disk_blocking import DiskBlocker
from src.memory import log_memory_state
from run_phase2 import generate_synthetic_tsvs

logger = logging.getLogger(__name__)

def run_blocking_stage(s1_max=10_000, s2_max=50_000, s3_max=50_000, batch_size=50_000):
    p1, p2, p3, pg = generate_synthetic_tsvs(s1_max, s2_max, s3_max)
    
    db_path = "output/entity_resolution.duckdb"
    temp_dir = "output/intermediate/tmp"
    
    # 1. Ingestion Phase
    storage = DiskStorage(db_path, temp_dir, memory_limit="8GB", threads=4)
    storage.ingest_source(p1, "s1", batch_size=batch_size, max_rows=s1_max)
    storage.ingest_source(p2, "s2", batch_size=batch_size, max_rows=s2_max)
    storage.ingest_source(p3, "s3", batch_size=batch_size, max_rows=s3_max)
    storage.ingest_ground_truth(pg, batch_size=batch_size, max_rows=s1_max)
    storage.close()
    
    # 2. Blocking Phase
    log_memory_state("Before Blocking", db_path, temp_dir)
    blocker = DiskBlocker(db_path, temp_dir, max_block_pairs=50_000)
    
    blocker.generate_exact_blocks()
    log_memory_state("After Block Generation", db_path, temp_dir)
    
    # Profile S1 x S2
    blocker.profile_blocks("exact_name", "s2")
    blocker.profile_blocks("exact_addr", "s2")
    blocker.profile_blocks("name_prefix_4", "s2")
    blocker.profile_blocks("addr_house_num", "s2")
    
    # Profile S1 x S3
    blocker.profile_blocks("exact_name", "s3")
    blocker.profile_blocks("exact_addr", "s3")
    blocker.profile_blocks("name_prefix_4", "s3")
    blocker.profile_blocks("addr_house_num", "s3")
    log_memory_state("After Block Profiling", db_path, temp_dir)
    
    # Generate S1 x S2 Candidates
    blocker.generate_candidates("exact_name", "s2", "from_exact_name")
    blocker.generate_candidates("exact_addr", "s2", "from_exact_addr")
    blocker.generate_candidates("name_prefix_4", "s2", "from_name_block")
    blocker.generate_candidates("addr_house_num", "s2", "from_addr_block")
    
    # Generate S1 x S3 Candidates
    blocker.generate_candidates("exact_name", "s3", "from_exact_name")
    blocker.generate_candidates("exact_addr", "s3", "from_exact_addr")
    blocker.generate_candidates("name_prefix_4", "s3", "from_name_block")
    blocker.generate_candidates("addr_house_num", "s3", "from_addr_block")
    log_memory_state("After Candidate Generation", db_path, temp_dir)
    
    # Cap Candidates
    blocker.cap_candidates(100)
    log_memory_state("After Candidate Capping", db_path, temp_dir)
    
    # 3. Evaluate Recall
    recall_stats = blocker.evaluate_recall()
    print("\n--- Recall Statistics ---")
    for k, v in recall_stats.items():
        print(f"{k}: {v}")
    
    # Print candidates per S1 statistics
    cands_per_s1_query = """
    SELECT 
        AVG(c_count) as avg_cands, 
        MAX(c_count) as max_cands,
        QUANTILE_CONT(c_count, 0.95) as p95_cands,
        QUANTILE_CONT(c_count, 0.99) as p99_cands
    FROM (
        SELECT source1_entity_id, COUNT(*) as c_count
        FROM candidates
        GROUP BY source1_entity_id
    )
    """
    s1_stats = blocker.conn.execute(cands_per_s1_query).fetchone()
    if s1_stats and s1_stats[0] is not None:
        print("\n--- Candidates per S1 ---")
        print(f"AVG: {s1_stats[0]:.2f}")
        print(f"MAX: {s1_stats[1]}")
        print(f"P95: {s1_stats[2]:.2f}")
        print(f"P99: {s1_stats[3]:.2f}")
    
    blocker.close()
    log_memory_state("After DB Close", db_path, temp_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-num", type=int, default=1, help="Test to run (1, 2, 3)")
    args = parser.parse_args()
    
    if args.test_num == 1:
        print("=== TEST A: 10k/50k/50k ===")
        run_blocking_stage(10_000, 50_000, 50_000, 10_000)
    elif args.test_num == 2:
        print("=== TEST B: 50k/250k/250k ===")
        run_blocking_stage(50_000, 250_000, 250_000, 50_000)
    elif args.test_num == 3:
        print("=== TEST C: 250k/1M/1M ===")
        run_blocking_stage(250_000, 1_000_000, 1_000_000, 50_000)
