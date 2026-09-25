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

    def generate_exact_blocks(self):
        """Populate exact name, exact address, and prefix4+house blocks (Phase 3B)."""
        logger.info("Generating exact match and prefix4+house blocking keys...")
        
        for src in ['s1', 's2', 's3']:
            self.conn.execute(f"DELETE FROM {src}_blocks")
            
            # 1. Exact Name
            self.conn.execute(f"""
            INSERT INTO {src}_blocks
            SELECT entity_id, 'exact_name', name_norm 
            FROM {src} 
            WHERE name_norm IS NOT NULL AND name_norm != ''
            """)
            
            # 2. Exact Address
            self.conn.execute(f"""
            INSERT INTO {src}_blocks
            SELECT entity_id, 'exact_addr', addr_norm 
            FROM {src} 
            WHERE addr_norm IS NOT NULL AND addr_norm != ''
            """)
            
            # 3. name_prefix_4 + addr_house_num
            self.conn.execute(f"""
            INSERT INTO {src}_blocks
            SELECT entity_id, 'prefix4_house', SUBSTRING(name_norm, 1, 4) || '_' || split_part(addr_norm, ' ', 1) 
            FROM {src} 
            WHERE name_norm IS NOT NULL AND name_norm != ''
              AND addr_norm IS NOT NULL AND addr_norm != ''
            """)
            
        logger.info("Phase 3B blocking keys populated.")

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
        checkpoint_key = f"{block_type}_{matched_source}"
        
        # Check if pass already completed
        self.conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (pass_name VARCHAR PRIMARY KEY)")
        exists = self.conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE pass_name = '{checkpoint_key}'").fetchone()[0]
        if exists > 0:
            logger.info(f"Checkpoint found for '{checkpoint_key}'. Skipping generation.")
            return
            
        logger.info(f"Generating candidates for '{block_type}' against '{matched_source}'")
        
        # We explicitly join s1_blocks and matched_source_blocks, but ONLY for blocks
        # where the pair count <= self.max_block_pairs
        
        # 1. Create a temporary view of safe keys
        self.conn.execute("DROP VIEW IF EXISTS safe_keys")
        safe_keys_query = f"""
        CREATE VIEW safe_keys AS
        SELECT t1.block_key
        FROM s1_blocks t1
        JOIN {matched_source}_blocks t2 
          ON t1.block_key = t2.block_key 
         AND t1.block_type = '{block_type}' 
         AND t2.block_type = '{block_type}'
        GROUP BY t1.block_key
        HAVING CAST(COUNT(DISTINCT t1.entity_id) AS BIGINT) * CAST(COUNT(DISTINCT t2.entity_id) AS BIGINT) <= {self.max_block_pairs}
        """
        self.conn.execute(safe_keys_query)
        
        # 2. Insert the actual candidate pairs directly into the candidates table via UPSERT
        # We use an UPSERT pattern in DuckDB.
        insert_query = f"""
        INSERT INTO candidates (source1_entity_id, matched_entity_id, matched_source, {flag_col})
        SELECT 
            s1.entity_id, 
            s2.entity_id, 
            '{matched_source}', 
            TRUE
        FROM s1_blocks s1
        JOIN {matched_source}_blocks s2 
          ON s1.block_key = s2.block_key
        JOIN safe_keys sk 
          ON s1.block_key = sk.block_key
        WHERE s1.block_type = '{block_type}' AND s2.block_type = '{block_type}'
        ON CONFLICT (source1_entity_id, matched_entity_id) 
        DO UPDATE SET {flag_col} = TRUE;
        """
        self.conn.execute(insert_query)
        
        # Write checkpoint
        self.conn.execute(f"INSERT INTO checkpoints VALUES ('{checkpoint_key}')")
        
        inserted_count = self.conn.execute(f"SELECT COUNT(*) FROM candidates WHERE {flag_col} = TRUE AND matched_source = '{matched_source}'").fetchone()[0]
        logger.info(f"Candidate generation complete. Candidates with {flag_col}=TRUE from {matched_source}: {inserted_count}")
        self.conn.execute("DROP VIEW IF EXISTS safe_keys")

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
