import os
import gc
import duckdb
import logging
import pandas as pd
from typing import Optional
from src.preprocessing import normalize_business_name, normalize_business_address, normalize_country
from src.memory import log_memory_state

logger = logging.getLogger(__name__)

class DiskStorage:
    def __init__(self, db_path: str, temp_dir: str, memory_limit: str = "8GB", threads: int = 4):
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.memory_limit = memory_limit
        self.threads = threads
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.conn = duckdb.connect(self.db_path)
        self._configure()
        self._create_schema()
        
    def _configure(self):
        self.conn.execute(f"SET threads = {self.threads}")
        self.conn.execute(f"SET memory_limit = '{self.memory_limit}'")
        self.conn.execute("SET preserve_insertion_order = false")
        self.conn.execute(f"PRAGMA temp_directory='{self.temp_dir}'")
        logger.info(f"DuckDB configured: limit={self.memory_limit}, threads={self.threads}, temp={self.temp_dir}")
        
    def _create_schema(self):
        table_definitions = [
            """
            CREATE TABLE IF NOT EXISTS s1 (
                entity_id VARCHAR PRIMARY KEY,
                name_norm VARCHAR,
                addr_norm VARCHAR,
                country_norm VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS s2 (
                entity_id VARCHAR PRIMARY KEY,
                name_norm VARCHAR,
                addr_norm VARCHAR,
                country_norm VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS s3 (
                entity_id VARCHAR PRIMARY KEY,
                name_norm VARCHAR,
                addr_norm VARCHAR,
                country_norm VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ground_truth (
                source1_entity_id VARCHAR,
                matched_entity_id VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS candidates (
                source1_entity_id VARCHAR,
                matched_entity_id VARCHAR,
                matched_source VARCHAR,
                from_exact_name BOOLEAN DEFAULT FALSE,
                from_exact_addr BOOLEAN DEFAULT FALSE,
                from_name_block BOOLEAN DEFAULT FALSE,
                from_addr_block BOOLEAN DEFAULT FALSE,
                from_sorted_neighborhood BOOLEAN DEFAULT FALSE,
                score FLOAT DEFAULT 0.0,
                PRIMARY KEY (source1_entity_id, matched_entity_id)
            )
            """
        ]
        for stmt in table_definitions:
            self.conn.execute(stmt)
        logger.info("DuckDB schema initialized.")

    def ingest_source(self, path: str, table_name: str, batch_size: int = 50_000, max_rows: int = None):
        logger.info(f"Ingesting source {table_name} from {path} in batches of {batch_size}")
        self.conn.execute(f"DELETE FROM {table_name}")
        
        iterator = pd.read_csv(
            path, sep='\t', chunksize=batch_size, dtype=str, 
            keep_default_na=False, na_values=[""]
        )
        
        total_inserted = 0
        batch_num = 1
        
        import psutil
        process = psutil.Process(os.getpid())
        def get_rss(): return process.memory_info().rss / (1024 * 1024 * 1024)
        
        for chunk in iterator:
            rss_before = get_rss()
            rows_in_batch = len(chunk)
            rss_after_read = get_rss()
            
            chunk['name_norm'] = chunk['business_name'].apply(normalize_business_name)
            chunk['addr_norm'] = chunk['business_address'].apply(normalize_business_address)
            chunk['country_norm'] = chunk['country'].apply(normalize_country)
            rss_after_norm = get_rss()
            
            batch_df = chunk[['entity_id', 'name_norm', 'addr_norm', 'country_norm']]
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM batch_df")
            rss_after_insert = get_rss()
            
            total_inserted += len(batch_df)
            
            del batch_df
            del chunk
            gc.collect()
            rss_after_gc = get_rss()
            
            if batch_num <= 3 or (max_rows and total_inserted >= max_rows):
                logger.info(f"[{table_name} Batch {batch_num}] Rows: {rows_in_batch} | "
                            f"RSS Before: {rss_before:.3f} | After Read: {rss_after_read:.3f} | "
                            f"After Norm: {rss_after_norm:.3f} | After Insert: {rss_after_insert:.3f} | "
                            f"After GC: {rss_after_gc:.3f}")
            
            batch_num += 1
            if max_rows and total_inserted >= max_rows:
                break
                
        logger.info(f"Completed ingestion of {table_name}: {total_inserted} rows.")
        
    def ingest_ground_truth(self, path: str, batch_size: int = 50_000, max_rows: int = None):
        """
        Streams ground truth TSV, un-nests comma-separated matched_entity_ids, and inserts.
        """
        logger.info(f"Ingesting ground_truth from {path}")
        self.conn.execute("DELETE FROM ground_truth")
        
        iterator = pd.read_csv(
            path, sep='\t', chunksize=batch_size, dtype=str, 
            keep_default_na=False, na_values=[""]
        )
        
        total_inserted = 0
        total_s1 = 0
        for chunk in iterator:
            total_s1 += len(chunk)
            
            chunk['matched_entity_ids'] = chunk['matched_entity_ids'].fillna("").astype(str)
            chunk = chunk[chunk['matched_entity_ids'] != ""]
            
            if not chunk.empty:
                # Unnest
                chunk = chunk.assign(
                    matched_entity_id=chunk['matched_entity_ids'].str.split(',')
                ).explode('matched_entity_id')
                
                chunk['matched_entity_id'] = chunk['matched_entity_id'].str.strip()
                chunk = chunk[chunk['matched_entity_id'] != ""]
                
                if not chunk.empty:
                    batch_df = chunk[['source1_entity_id', 'matched_entity_id']]
                    self.conn.execute("INSERT INTO ground_truth SELECT * FROM batch_df")
                    total_inserted += len(batch_df)
                    
                    del batch_df
                    
            del chunk
            gc.collect()
            
            if max_rows and total_s1 >= max_rows:
                break
                
        logger.info(f"Completed ingestion of ground_truth: {total_inserted} edges from {total_s1} S1 entities.")

    def get_table_counts(self):
        counts = {}
        for table in ['s1', 's2', 's3', 'ground_truth', 'candidates']:
            res = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = res[0]
        return counts

    def close(self):
        self.conn.close()
        logger.info("DuckDB connection closed.")
