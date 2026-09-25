import os
import gc
import logging
import pandas as pd
from typing import Dict, List, Optional
import duckdb

logger = logging.getLogger(__name__)

class DiskBlocker:
    def __init__(self, db_path: str, temp_dir: str, max_block_pairs: int = 50_000):
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.max_block_pairs = max_block_pairs
        self.conn = duckdb.connect(self.db_path)
        self._init_block_tables()
        
    def _init_block_tables(self):
        # Create block tables if not exist
        # block_type: e.g. 'exact_name', 'exact_addr', 'name_prefix_4'
        for src in ['s1', 's2', 's3']:
            self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {src}_blocks (
                entity_id VARCHAR,
                block_type VARCHAR,
                block_key VARCHAR
            )
            """)
        logger.info("Block tables initialized.")

    def close(self):
        self.conn.close()

    def _get_rss_gb(self) -> float:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)

    def generate_exact_blocks(self):
        """Populate exact name, exact address, and prefix4+house blocks (Phase 3B)."""
        import time
        self.conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id VARCHAR PRIMARY KEY)")
        exists = self.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE checkpoint_id = 'PHASE3B_BLOCKING_KEYS_COMPLETE'").fetchone()[0]
        if exists > 0:
            logger.info("PHASE3B_BLOCKING_KEYS_COMPLETE: Skipped block generation.")
            return
            
        logger.info("Generating exact match and prefix4+house blocking keys...")
        
        try:
            for src in ['s1', 's2', 's3']:
                # explicit clean/reset of temporary blocking tables
                logger.info(f"START {src}_blocks reset")
                t0 = time.time()
                self.conn.execute(f"DROP TABLE IF EXISTS {src}_blocks")
                self.conn.execute(f"""
                CREATE TABLE {src}_blocks (
                    entity_id VARCHAR,
                    block_type VARCHAR,
                    block_key VARCHAR
                )
                """)
                logger.info(f"END {src}_blocks reset | Time: {time.time()-t0:.2f}s | RSS: {self._get_rss_gb():.3f} GB")
                
                # 1. Exact Name
                logger.info(f"START {src} exact_name key population")
                t0 = time.time()
                self.conn.execute(f"""
                INSERT INTO {src}_blocks
                SELECT entity_id, 'exact_name', name_norm 
                FROM {src} 
                WHERE name_norm IS NOT NULL AND name_norm != ''
                """)
                logger.info(f"END {src} exact_name key population | Time: {time.time()-t0:.2f}s | RSS: {self._get_rss_gb():.3f} GB")
                
                # 2. Exact Address
                logger.info(f"START {src} exact_addr key population")
                t0 = time.time()
                self.conn.execute(f"""
                INSERT INTO {src}_blocks
                SELECT entity_id, 'exact_addr', addr_norm 
                FROM {src} 
                WHERE addr_norm IS NOT NULL AND addr_norm != ''
                """)
                logger.info(f"END {src} exact_addr key population | Time: {time.time()-t0:.2f}s | RSS: {self._get_rss_gb():.3f} GB")
                
                # 3. name_prefix_4 + addr_house_num
                logger.info(f"START {src} prefix4_house key population")
                t0 = time.time()
                self.conn.execute(f"""
                INSERT INTO {src}_blocks
                SELECT entity_id, 'prefix4_house', SUBSTRING(name_norm, 1, 4) || '_' || split_part(addr_norm, ' ', 1) 
                FROM {src} 
                WHERE name_norm IS NOT NULL AND name_norm != ''
                  AND addr_norm IS NOT NULL AND addr_norm != ''
                """)
                logger.info(f"END {src} prefix4_house key population | Time: {time.time()-t0:.2f}s | RSS: {self._get_rss_gb():.3f} GB")
                
            self.conn.execute("INSERT INTO checkpoints VALUES ('PHASE3B_BLOCKING_KEYS_COMPLETE')")
            logger.info("PHASE3B_BLOCKING_KEYS_COMPLETE")
        except Exception as e:
            import traceback
            logger.error("PHASE3B_BLOCKING_KEYS_FAILED")
            logger.error(f"Failed during block population. RSS: {self._get_rss_gb():.3f} GB")
            traceback.print_exc()
            raise

    def profile_blocks(self, block_type: str, matched_source: str) -> pd.DataFrame:
        """
        Calculates S1 x matched_source block sizes.
        Returns a DataFrame of block statistics.
        """
        logger.info(f"Profiling block_type='{block_type}' against '{matched_source}'")
        
        query = f"""
        SELECT 
            t1.block_key,
            COUNT(DISTINCT t1.entity_id) as s1_size,
            COUNT(DISTINCT t2.entity_id) as s2_size,
            CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) as estimated_pairs
        FROM s1_blocks t1
        JOIN {matched_source}_blocks t2 
          ON t1.block_key = t2.block_key 
         AND t1.block_type = t2.block_type
        WHERE t1.block_type = '{block_type}'
        GROUP BY t1.block_key
        """
        df = self.conn.execute(query).fetchdf()
        
        os.makedirs("output/intermediate", exist_ok=True)
        stats_path = f"output/intermediate/block_stats_{block_type}_{matched_source}.tsv"
        df.to_csv(stats_path, sep='\t', index=False)
        
        total_keys = len(df)
        oversized = len(df[df['estimated_pairs'] > self.max_block_pairs]) if not df.empty else 0
        total_estimated = df['estimated_pairs'].sum() if not df.empty else 0
        
        logger.info(f"[{block_type} | {matched_source}] Total Keys: {total_keys} | "
                    f"Oversized (> {self.max_block_pairs}): {oversized} | "
                    f"Total Estimated Pairs: {total_estimated}")
        
        return df

    def generate_candidates(self, block_type: str, matched_source: str, flag_col: str):
        """
        Executes the join for safe blocks and inserts directly into `candidates`.
        Includes checkpointing to prevent re-running completed passes.
        """
        pass_checkpoint = f"{block_type}_{matched_source}_COMPLETE"
        
        # Check if pass already completed entirely
        self.conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id VARCHAR PRIMARY KEY)")
        exists = self.conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE checkpoint_id = '{pass_checkpoint}'").fetchone()[0]
        if exists > 0:
            logger.info(f"Checkpoint found for '{pass_checkpoint}'. Skipping generation.")
            return
            
        logger.info(f"Generating candidates for '{block_type}' against '{matched_source}'")
        
        # We explicitly join s1_blocks and matched_source_blocks, but ONLY for blocks
        # where the pair count <= self.max_block_pairs
        
        # We explicitly compute the block frequencies first and materialize safe_keys
        # to guarantee the DuckDB optimizer does not execute an unsafe Cartesian product.
        self.conn.execute("DROP TABLE IF EXISTS tmp_s1_cnt")
        self.conn.execute("DROP TABLE IF EXISTS tmp_s2_cnt")
        self.conn.execute("DROP TABLE IF EXISTS safe_keys")
        
        self.conn.execute(f"""
        CREATE TEMP TABLE tmp_s1_cnt AS
        SELECT block_key, COUNT(DISTINCT entity_id) as s1_size
        FROM s1_blocks
        WHERE block_type = '{block_type}'
        GROUP BY block_key
        """)
        
        self.conn.execute(f"""
        CREATE TEMP TABLE tmp_s2_cnt AS
        SELECT block_key, COUNT(DISTINCT entity_id) as s2_size
        FROM {matched_source}_blocks
        WHERE block_type = '{block_type}'
        GROUP BY block_key
        """)
        
        self.conn.execute(f"""
        CREATE TEMP TABLE safe_keys AS
        SELECT t1.block_key
        FROM tmp_s1_cnt t1
        JOIN tmp_s2_cnt t2 ON t1.block_key = t2.block_key
        WHERE CAST(t1.s1_size AS BIGINT) * CAST(t2.s2_size AS BIGINT) <= {self.max_block_pairs}
        """)
        
        # 2. Process block keys in batches and insert candidates incrementally
        total_safe_keys = self.conn.execute("SELECT COUNT(*) FROM safe_keys").fetchone()[0]
        logger.info(f"Total safe keys for {block_type}: {total_safe_keys}")
        
        chunk_size = 50_000
        batch_idx = 0
        import time
        
        for offset in range(0, total_safe_keys, chunk_size):
            batch_checkpoint = f"{block_type}_{matched_source}_batch_{batch_idx}_offset_{offset}"
            
            # Check if this batch is already done
            batch_exists = self.conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE checkpoint_id = '{batch_checkpoint}'").fetchone()[0]
            if batch_exists > 0:
                logger.info(f"Skipping completed batch: {batch_checkpoint}")
                batch_idx += 1
                continue
                
            chunk_end = min(offset + chunk_size, total_safe_keys)
            logger.info(f"CANDIDATE_BATCH_START: {block_type} -> {matched_source} (Batch {batch_idx}, Keys {offset} to {chunk_end})")
            t0 = time.time()
            
            # Create a chunk of safe keys
            self.conn.execute(f"""
            CREATE TEMP TABLE safe_keys_chunk AS
            SELECT block_key FROM safe_keys 
            ORDER BY block_key 
            LIMIT {chunk_size} OFFSET {offset}
            """)
            
            # Insert candidates for this chunk
            insert_query = f"""
            INSERT INTO candidates (source1_entity_id, matched_entity_id, matched_source, {flag_col})
            SELECT 
                s1.entity_id, 
                s2.entity_id, 
                '{matched_source}', 
                TRUE
            FROM safe_keys_chunk sk
            JOIN s1_blocks s1 
              ON sk.block_key = s1.block_key AND s1.block_type = '{block_type}'
            JOIN {matched_source}_blocks s2 
              ON sk.block_key = s2.block_key AND s2.block_type = '{block_type}'
            ON CONFLICT (source1_entity_id, matched_entity_id) 
            DO UPDATE SET {flag_col} = TRUE;
            """
            self.conn.execute(insert_query)
            
            # Release intermediate state
            self.conn.execute("DROP TABLE safe_keys_chunk")
            
            # Mark batch complete ONLY after successful transaction
            self.conn.execute(f"INSERT INTO checkpoints VALUES ('{batch_checkpoint}')")
            
            # Track progress
            elapsed = time.time() - t0
            logger.info(f"CANDIDATE_BATCH_COMPLETE: {block_type} -> {matched_source} | Batch {batch_idx} | Keys: {chunk_end}/{total_safe_keys} | Time: {elapsed:.2f}s | RSS: {self._get_rss_gb():.3f} GB")
            gc.collect()
            batch_idx += 1
            
        # Clean up global safe keys tables
        self.conn.execute("DROP TABLE tmp_s1_cnt")
        self.conn.execute("DROP TABLE tmp_s2_cnt")
        self.conn.execute("DROP TABLE safe_keys")
        
        # Mark entire pass complete
        self.conn.execute(f"INSERT INTO checkpoints VALUES ('{pass_checkpoint}')")
        
        inserted_count = self.conn.execute(f"SELECT COUNT(*) FROM candidates WHERE {flag_col} = TRUE AND matched_source = '{matched_source}'").fetchone()[0]
        logger.info(f"Candidate generation complete. Candidates with {flag_col}=TRUE from {matched_source}: {inserted_count}")

    def evaluate_recall(self) -> dict:
        """
        Calculates recall of generated candidates against the ground truth table.
        """
        logger.info("Evaluating Candidate Recall...")
        
        # Total true pairs
        total_true = self.conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
        if total_true == 0:
            logger.warning("No ground truth available for recall evaluation.")
            return {"total_true": 0, "retrieved": 0, "recall": 0.0}
            
        # Retrieved true pairs (candidates that exist in ground_truth)
        retrieved_query = """
        SELECT COUNT(*)
        FROM ground_truth gt
        JOIN candidates c
          ON gt.source1_entity_id = c.source1_entity_id 
         AND gt.matched_entity_id = c.matched_entity_id
        """
        retrieved_true = self.conn.execute(retrieved_query).fetchone()[0]
        
        recall = retrieved_true / total_true
        
        # Stats
        total_cands = self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        logger.info(f"Recall: {recall:.4f} ({retrieved_true}/{total_true} true pairs) | Total Candidates: {total_cands}")
        
        return {
            "total_true": total_true,
            "retrieved": retrieved_true,
            "recall": recall,
            "total_candidates": total_cands
        }
        
    def cap_candidates(self, max_candidates_per_s1: int = 100):
        """
        Caps candidates to a maximum per S1 entity.
        Ranks by cheap evidence (exact match flags, block hits).
        """
        logger.info(f"Capping candidates to {max_candidates_per_s1} per S1 entity...")
        
        # Calculate a cheap score based on flags to rank candidates
        self.conn.execute("""
        UPDATE candidates 
        SET score = (
            CAST(from_exact_name AS INTEGER) * 100 +
            CAST(from_exact_addr AS INTEGER) * 50 +
            CAST(from_name_block AS INTEGER) * 10 +
            CAST(from_addr_block AS INTEGER) * 10
        )
        """)
        
        # Delete candidates that don't make the cut
        self.conn.execute(f"""
        DELETE FROM candidates 
        WHERE (source1_entity_id, matched_entity_id) NOT IN (
            SELECT source1_entity_id, matched_entity_id
            FROM (
                SELECT 
                    source1_entity_id, 
                    matched_entity_id,
                    ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY score DESC, matched_entity_id ASC) as rn
                FROM candidates
            ) ranked
            WHERE rn <= {max_candidates_per_s1}
        )
        """)
        
        capped_count = self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        logger.info(f"Capping complete. Total remaining candidates: {capped_count}")
