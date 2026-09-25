import os
import unittest
import duckdb
from src.disk_blocking import DiskBlocker

class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_candidates.duckdb"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.conn = duckdb.connect(self.db_path)
        
        # Setup schema
        self.conn.execute("""
        CREATE TABLE s1 (entity_id VARCHAR, name_norm VARCHAR, addr_norm VARCHAR);
        CREATE TABLE s2 (entity_id VARCHAR, name_norm VARCHAR, addr_norm VARCHAR);
        CREATE TABLE s3 (entity_id VARCHAR, name_norm VARCHAR, addr_norm VARCHAR);
        CREATE TABLE candidates (
            source1_entity_id VARCHAR,
            matched_entity_id VARCHAR,
            matched_source VARCHAR,
            from_exact_name BOOLEAN DEFAULT FALSE,
            from_exact_addr BOOLEAN DEFAULT FALSE,
            from_name_block BOOLEAN DEFAULT FALSE,
            from_addr_block BOOLEAN DEFAULT FALSE,
            score FLOAT DEFAULT 0.0,
            PRIMARY KEY (source1_entity_id, matched_entity_id)
        );
        CREATE TABLE ground_truth (source1_entity_id VARCHAR, matched_entity_id VARCHAR);
        """)
        
        self.blocker = DiskBlocker(self.db_path, ".", max_block_pairs=5)
        
    def tearDown(self):
        self.blocker.close()
        self.conn.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except:
                pass
                
    def test_candidate_generation_rules(self):
        # Insert test data
        
        # EXACT NAME match (S1-1 <-> S2-1)
        self.conn.execute("INSERT INTO s1 VALUES ('S1-1', 'amazon', 'seattle')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-1', 'amazon', 'other')")
        
        # EXACT ADDR match (S1-2 <-> S2-2)
        self.conn.execute("INSERT INTO s1 VALUES ('S1-2', 'diff1', '123 main st')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-2', 'diff2', '123 main st')")
        
        # PREFIX4 + HOUSE match (S1-3 <-> S2-3)
        self.conn.execute("INSERT INTO s1 VALUES ('S1-3', 'amaz inc', '123 broadway')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-3', 'amazon', '123 ave')")
        
        # OVERLAP (S1-4 <-> S2-4 matches on both exact_name and exact_addr)
        self.conn.execute("INSERT INTO s1 VALUES ('S1-4', 'google', 'plex')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-4', 'google', 'plex')")
        
        # S3 match (S1-5 <-> S3-5 on exact_name)
        self.conn.execute("INSERT INTO s1 VALUES ('S1-5', 'apple', 'cupertino')")
        self.conn.execute("INSERT INTO s3 VALUES ('S3-5', 'apple', 'other')")
        
        # OVERSIZED BLOCK (S1-6..S1-10 <-> S2-6..S2-10 on exact_name 'spam')
        # max_block_pairs is 5. 5x5 = 25 > 5. Should be rejected!
        for i in range(6, 11):
            self.conn.execute(f"INSERT INTO s1 VALUES ('S1-{i}', 'spam', 'addr1{i}')")
            self.conn.execute(f"INSERT INTO s2 VALUES ('S2-{i}', 'spam', 'addr2{i}')")

        # Generate blocks
        self.blocker.generate_exact_blocks()
        
        # Run S2 candidates
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        self.blocker.generate_candidates('exact_addr', 's2', 'from_exact_addr')
        self.blocker.generate_candidates('prefix4_house', 's2', 'from_name_block')
        
        # Run S3 candidates
        self.blocker.generate_candidates('exact_name', 's3', 'from_exact_name')
        
        df = self.conn.execute("SELECT * FROM candidates ORDER BY source1_entity_id").fetchdf()
        
        # Assertions
        # 1. exact_name
        c1 = df[(df.source1_entity_id == 'S1-1') & (df.matched_entity_id == 'S2-1')].iloc[0]
        self.assertTrue(c1.from_exact_name)
        
        # 2. exact_addr
        c2 = df[(df.source1_entity_id == 'S1-2') & (df.matched_entity_id == 'S2-2')].iloc[0]
        self.assertTrue(c2.from_exact_addr)
        
        # 3. prefix4+house
        c3 = df[(df.source1_entity_id == 'S1-3') & (df.matched_entity_id == 'S2-3')].iloc[0]
        self.assertTrue(c3.from_name_block)
        
        # 4. Overlap (deduplication check)
        c4 = df[(df.source1_entity_id == 'S1-4') & (df.matched_entity_id == 'S2-4')]
        self.assertEqual(len(c4), 1, "Should be deduplicated to one row")
        self.assertTrue(c4.iloc[0].from_exact_name and c4.iloc[0].from_exact_addr)
        
        # 5. S2/S3 Separation
        c5 = df[(df.source1_entity_id == 'S1-5') & (df.matched_entity_id == 'S3-5')].iloc[0]
        self.assertEqual(c5.matched_source, 's3')
        
        # 6. Oversized Block rejection
        spam = df[df.source1_entity_id.str.startswith('S1-6')]
        self.assertEqual(len(spam), 0, "Oversized block should produce zero candidates")

    def test_candidate_capping_determinism(self):
        # We need to test the cap_candidates function determinism
        # and ensure it reads the canonical config correctly.
        from src.config import cfg
        
        self.conn.execute("INSERT INTO s1 VALUES ('S1-CAP', 'target', 'addr')")
        
        # Insert 10 candidates with exact same score but different matched_entity_id
        # from_exact_name = FALSE (0), from_name_block = TRUE (10)
        # So score = 10 for all
        for i in range(10, 0, -1):
            self.conn.execute(f"INSERT INTO candidates (source1_entity_id, matched_entity_id, from_exact_name, from_name_block) VALUES ('S1-CAP', 'S2-{i:02d}', FALSE, TRUE)")
            
        # Temporarily mock the cfg value for the test
        original_cap = cfg.MAX_CANDIDATES_PER_ENTITY
        try:
            # We mock the attribute on the frozen dataclass for the duration of the test
            object.__setattr__(cfg, 'MAX_CANDIDATES_PER_ENTITY', 3)
            
            self.blocker.cap_candidates(cfg.MAX_CANDIDATES_PER_ENTITY)
            cands = self.conn.execute("SELECT matched_entity_id FROM candidates WHERE source1_entity_id = 'S1-CAP' ORDER BY matched_entity_id").fetchall()
            
            # Should be capped to exactly 3 candidates
            self.assertEqual(len(cands), 3)
            
            # Because scores are tied (10), tie-breaker is matched_entity_id ASC
            # So we expect S2-01, S2-02, S2-03
            self.assertEqual(cands[0][0], 'S2-01')
            self.assertEqual(cands[1][0], 'S2-02')
            self.assertEqual(cands[2][0], 'S2-03')
            
            # Verify removed candidates are totally absent
            missing = self.conn.execute("SELECT COUNT(*) FROM candidates WHERE matched_entity_id = 'S2-04'").fetchone()[0]
            self.assertEqual(missing, 0)
            
            # Run it again on fresh data to verify identical deterministic behavior
            self.conn.execute("DELETE FROM candidates")
            for i in range(10, 0, -1):
                self.conn.execute(f"INSERT INTO candidates (source1_entity_id, matched_entity_id, from_exact_name, from_name_block) VALUES ('S1-CAP', 'S2-{i:02d}', FALSE, TRUE)")
                
            self.blocker.cap_candidates(cfg.MAX_CANDIDATES_PER_ENTITY)
            cands2 = self.conn.execute("SELECT matched_entity_id FROM candidates WHERE source1_entity_id = 'S1-CAP' ORDER BY matched_entity_id").fetchall()
            
            self.assertEqual(cands, cands2, "Deterministic capping failed across runs")
        finally:
            # Restore original cap
            object.__setattr__(cfg, 'MAX_CANDIDATES_PER_ENTITY', original_cap)

    def test_repeated_initialization_safety(self):
        # Insert a little data
        self.conn.execute("INSERT INTO s1 VALUES ('S1-1', 'amazon', 'seattle')")
        
        # First initialization
        self.blocker.generate_exact_blocks()
        
        # Corrupt the blocking table to simulate a failed/stale run
        self.conn.execute("INSERT INTO s1_blocks VALUES ('S1-STALE', 'exact_name', 'stale')")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM s1_blocks").fetchone()[0], 4)
        
        # Manually delete the complete checkpoint to simulate a failed previous run
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'PHASE3B_BLOCKING_KEYS_COMPLETE'")
        
        # Second initialization (should safely DROP and CREATE, clearing the corruption)
        self.blocker.generate_exact_blocks()
        
        # The stale row should be gone, exactly 3 rows should remain
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM s1_blocks").fetchone()[0], 3)
        stale_cnt = self.conn.execute("SELECT COUNT(*) FROM s1_blocks WHERE entity_id = 'S1-STALE'").fetchone()[0]
        self.assertEqual(stale_cnt, 0)
        
        # Ensure it works end-to-end after restart
        self.conn.execute("INSERT INTO s2 VALUES ('S2-1', 'amazon', 'seattle')")
        # Need to clear the checkpoint so it actually runs
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'PHASE3B_BLOCKING_KEYS_COMPLETE'")
        self.blocker.generate_exact_blocks()
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        
        cands = self.conn.execute("SELECT * FROM candidates").fetchdf()
        self.assertEqual(len(cands), 1)

    def test_resume_batch_generation(self):
        """Simulate an interruption and verify resume logic skips completed batches."""
        # 1. Insert enough records to generate multiple batches
        self.conn.execute("INSERT INTO s1 VALUES ('S1-RESUME-1', 'amazon', 'seattle')")
        self.conn.execute("INSERT INTO s1 VALUES ('S1-RESUME-2', 'apple', 'cupertino')")
        self.conn.execute("INSERT INTO s1 VALUES ('S1-RESUME-3', 'google', 'mountain view')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-RESUME-1', 'amazon', 'seattle')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-RESUME-2', 'apple', 'cupertino')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-RESUME-3', 'google', 'mountain view')")
        
        # 2. First execution - normal
        self.blocker.generate_exact_blocks()
        
        # Mock chunk size to force 3 batches (chunk_size=1)
        original_execute = self.conn.execute
        
        # Let's just directly insert a checkpoint for batch 0 and 1!
        self.conn.execute("INSERT INTO checkpoints VALUES ('exact_name_s2_batch_0_offset_0')")
        self.conn.execute("INSERT INTO checkpoints VALUES ('exact_name_s2_batch_1_offset_50000')")
        
        # Run it once. It will skip batch 0 and complete the PASS checkpoint!
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        
        # We expect 0 candidates because the only batch (offset 0) was marked complete.
        cands_skipped = self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        self.assertEqual(cands_skipped, 0)
        
        # Now delete the batch checkpoint AND the pass checkpoint to rerun it successfully.
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'exact_name_s2_batch_0_offset_0'")
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'exact_name_s2_COMPLETE'")
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        
        cands_run = self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        self.assertEqual(cands_run, 3) # 3 matching pairs
        
        # Run it a third time, but the pass is marked complete: 'exact_name_s2_COMPLETE'
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        cands_final = self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        self.assertEqual(cands_final, 3) # Still 3, didn't run again

    def test_backward_compatibility(self):
        """Verify that an existing DuckDB with the old pass_name schema is correctly resumed."""
    def test_backward_compatibility(self):
        """Verify that an existing DuckDB with the old pass_name schema is correctly resumed."""
        # Insert some data and candidate tables as they would exist in an interrupted run
        self.conn.execute("INSERT INTO s1 VALUES ('S1-BW-1', 'amazon', 'seattle')")
        self.conn.execute("INSERT INTO s2 VALUES ('S2-BW-1', 'amazon', 'seattle')")
        
        self.conn.execute("INSERT INTO candidates (source1_entity_id, matched_entity_id, matched_source, from_exact_name, score) VALUES ('S1-BW-1', 'S2-BW-1', 's2', TRUE, 0.0)")
        
        # Explicitly create the old schema: pass_name instead of checkpoint_id
        self.conn.execute("CREATE TABLE checkpoints (pass_name VARCHAR PRIMARY KEY)")
        
        # Insert old pass-level completions
        self.conn.execute("INSERT INTO checkpoints VALUES ('PHASE3B_BLOCKING_KEYS_COMPLETE')")
        self.conn.execute("INSERT INTO checkpoints VALUES ('exact_name_s2_COMPLETE')")
        self.conn.execute("INSERT INTO checkpoints VALUES ('exact_addr_s2_COMPLETE')")
        
        # Insert new batch-level completion for prefix4_house
        self.conn.execute("INSERT INTO checkpoints VALUES ('prefix4_house_s2_batch_0_offset_0')")
        
        # 2. Re-open via our DiskBlocker code
        # Wait, self.blocker already opened it! But self.blocker was initialized in setUp before we made checkpoints table.
        # It doesn't matter because DiskBlocker uses the same DB path dynamically.
        # Validate that CREATE TABLE IF NOT EXISTS doesn't blow up the schema
        # In run_kaggle_candidate_generation, it does CREATE TABLE IF NOT EXISTS checkpoints (pass_name VARCHAR PRIMARY KEY)
        self.conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (pass_name VARCHAR PRIMARY KEY)")
        
        # 3. Verify exactly what the user asked
        # - old checkpoints are still present
        checkpoints_df = self.conn.execute("SELECT * FROM checkpoints").fetchdf()
        self.assertEqual(len(checkpoints_df), 4)
        
        # - no existing candidate rows are deleted
        cands_df = self.conn.execute("SELECT * FROM candidates").fetchdf()
        self.assertEqual(len(cands_df), 1)
        self.assertEqual(cands_df.iloc[0]['source1_entity_id'], 'S1-BW-1')
        
        # - exact_name_s2 completion is recognized (it should instantly return)
        self.blocker.generate_candidates('exact_name', 's2', 'from_exact_name')
        # since it instantly returns, candidates count is still 1
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 1)
        
        # - new batch checkpoints can be written (simulate prefix4_house completion)
        # Note: we skip prefix4_house block generation because PHASE3B_BLOCKING_KEYS_COMPLETE exists!
        # So we must manually inject blocks so the batch generator can run batch 1.
        self.conn.execute("CREATE TABLE IF NOT EXISTS s1_blocks (entity_id VARCHAR, block_type VARCHAR, block_key VARCHAR)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS s2_blocks (entity_id VARCHAR, block_type VARCHAR, block_key VARCHAR)")
        self.conn.execute("INSERT INTO s1_blocks VALUES ('S1-BW-2', 'prefix4_house', 'amaz_seattle')")
        self.conn.execute("INSERT INTO s2_blocks VALUES ('S2-BW-2', 'prefix4_house', 'amaz_seattle')")
        
        # Now run prefix4_house. We already marked batch 0 (offset 0) as complete!
        self.blocker.generate_candidates('prefix4_house', 's2', 'from_name_block')
        
        # candidates should STILL be 1 because batch 0 was skipped.
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 1)
        
        # Now let's test a batch that is NOT complete.
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'prefix4_house_s2_batch_0_offset_0'")
        self.conn.execute("DELETE FROM checkpoints WHERE pass_name = 'prefix4_house_s2_COMPLETE'")
        self.blocker.generate_candidates('prefix4_house', 's2', 'from_name_block')
        
        # Now it should process batch 0 and insert the new candidate
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 2)
        
        # Validate that the new batch checkpoint was successfully written!
        new_batch_exists = self.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE pass_name = 'prefix4_house_s2_batch_0_offset_0'").fetchone()[0]
        self.assertEqual(new_batch_exists, 1)


if __name__ == '__main__':
    unittest.main()
