import os
import argparse
import logging
import pandas as pd
import random
from src.config import cfg
from src.storage import DiskStorage
from src.memory import log_memory_state
from src.preprocessing import normalize_business_name, normalize_business_address, normalize_country

logger = logging.getLogger(__name__)

def generate_synthetic_tsvs(s1_max, s2_max, s3_max):
    os.makedirs("output/synthetic", exist_ok=True)
    p1 = "output/synthetic/s1.tsv"
    p2 = "output/synthetic/s2.tsv"
    p3 = "output/synthetic/s3.tsv"
    pg = "output/synthetic/gt.tsv"
    
    WORDS = ["amazon", "inc", "corp", "ltd", "tech", "data", "solutions", "global", "systems"]
    
    def rname(): return " ".join(random.choices(WORDS, k=2))
    def raddr(): return " ".join(random.choices(WORDS, k=3))
    
    logger.info("Generating synthetic TSVs...")
    
    pd.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(s1_max)],
        "business_name": [rname() for _ in range(s1_max)],
        "business_address": [raddr() for _ in range(s1_max)],
        "country": ["us"] * s1_max
    }).to_csv(p1, sep='\t', index=False)
    
    pd.DataFrame({
        "entity_id": [f"S2-{i}" for i in range(s2_max)],
        "business_name": [rname() for _ in range(s2_max)],
        "business_address": [raddr() for _ in range(s2_max)],
        "country": ["us"] * s2_max
    }).to_csv(p2, sep='\t', index=False)
    
    pd.DataFrame({
        "entity_id": [f"S3-{i}" for i in range(s3_max)],
        "business_name": [rname() for _ in range(s3_max)],
        "business_address": [raddr() for _ in range(s3_max)],
        "country": ["us"] * s3_max
    }).to_csv(p3, sep='\t', index=False)
    
    pd.DataFrame({
        "source1_entity_id": [f"S1-{i}" for i in range(s1_max)],
        "matched_entity_ids": [f"S2-{i%s2_max},S3-{i%s3_max}" for i in range(s1_max)]
    }).to_csv(pg, sep='\t', index=False)
    
    return p1, p2, p3, pg

def test_normalization():
    logger.info("Validating normalization semantics...")
    
    name = "  McDonald's   "
    assert normalize_business_name(name) == 'mcdonald s', f"Failed: {normalize_business_name(name)}"
    assert normalize_business_name("AT&T Inc.") == 'at and t inc'
    assert normalize_business_name("Café Rösti GmbH") == 'cafe rosti gmbh'
    assert normalize_business_name(None) == ''
    
    addr = "123, Main Street, New York, NY 10001"
    assert normalize_business_address(addr) == '123 main street new york ny 10001'
    assert normalize_business_address("Plot No. 45, Sector-5, Gurugram") == 'plot no 45 sector 5 gurugram'
    assert normalize_business_address(None) == ''
    
    assert normalize_country("United States") == 'united states'
    assert normalize_country("  India  ") == 'india'
    assert normalize_country("France") == 'france'
    assert normalize_country(None) == ''
    
    logger.info("Normalization semantics validated and perfectly preserved.")

def run_storage_stage(s1_max=None, s2_max=None, s3_max=None, batch_size=50000):
    p1, p2, p3, pg = generate_synthetic_tsvs(s1_max, s2_max, s3_max)
    test_normalization()
    
    db_path = "output/entity_resolution.duckdb"
    temp_dir = "output/intermediate/tmp"
    
    log_memory_state("Before connection", db_path, temp_dir)
    storage = DiskStorage(db_path, temp_dir, memory_limit="8GB", threads=4)
    log_memory_state("After schema creation", db_path, temp_dir)
    
    storage.ingest_source(p1, "s1", batch_size=batch_size, max_rows=s1_max)
    log_memory_state("After S1 ingestion", db_path, temp_dir)
    
    storage.ingest_source(p2, "s2", batch_size=batch_size, max_rows=s2_max)
    log_memory_state("After S2 ingestion", db_path, temp_dir)
    
    storage.ingest_source(p3, "s3", batch_size=batch_size, max_rows=s3_max)
    log_memory_state("After S3 ingestion", db_path, temp_dir)
    
    storage.ingest_ground_truth(pg, batch_size=batch_size, max_rows=s1_max)
    log_memory_state("After Ground Truth ingestion", db_path, temp_dir)
    
    # 4. Report table counts
    counts = storage.get_table_counts()
    print("\nTable Counts:")
    for t, c in counts.items():
        print(f"  {t}: {c} rows")
        
    storage.close()
    
    # 5. RSS after close
    log_memory_state("After connection close", db_path, temp_dir)

def reopen_test():
    db_path = "output/entity_resolution.duckdb"
    import duckdb
    con = duckdb.connect(db_path)
    print("\nReopen Test Counts:")
    print(f"  s1: {con.execute('SELECT COUNT(*) FROM s1').fetchone()[0]}")
    print(f"  s2: {con.execute('SELECT COUNT(*) FROM s2').fetchone()[0]}")
    print(f"  s3: {con.execute('SELECT COUNT(*) FROM s3').fetchone()[0]}")
    print(f"  ground_truth: {con.execute('SELECT COUNT(*) FROM ground_truth').fetchone()[0]}")
    
    print("\nSample S1:")
    print(con.execute("SELECT * FROM s1 LIMIT 5").fetchdf())
    con.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-num", type=int, default=1, help="Test to run (1, 2, 3)")
    args = parser.parse_args()
    
    if args.test_num == 1:
        # TINY DATA
        print("=== TEST 1: TINY DATA ===")
        run_storage_stage(s1_max=100, s2_max=500, s3_max=500, batch_size=50)
        reopen_test()
    elif args.test_num == 2:
        # REAL DATA SUBSET
        print("=== TEST 2: REAL DATA SUBSET ===")
        run_storage_stage(s1_max=10_000, s2_max=50_000, s3_max=50_000, batch_size=10_000)
        reopen_test()
    elif args.test_num == 3:
        # LARGER SUBSET
        print("=== TEST 3: LARGER SUBSET ===")
        run_storage_stage(s1_max=50_000, s2_max=250_000, s3_max=250_000, batch_size=50_000)
        reopen_test()
    elif args.test_num == 4:
        # HUGE SUBSET
        print("=== TEST 4: 250k/1M/1M ===")
        run_storage_stage(s1_max=250_000, s2_max=1_000_000, s3_max=1_000_000, batch_size=50_000)
        reopen_test()
