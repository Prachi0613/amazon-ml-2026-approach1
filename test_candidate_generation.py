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

    def test_candidate_capping(self):
        # We need to test the cap_candidates function
        self.conn.execute("INSERT INTO s1 VALUES ('S1-CAP', 'target', 'addr')")
        for i in range(1, 10):
            self.conn.execute(f"INSERT INTO candidates (source1_entity_id, matched_entity_id, from_exact_name, from_name_block) VALUES ('S1-CAP', 'S2-{i}', {i%2==0}, {i%2!=0})")
            
        self.blocker.cap_candidates(max_candidates_per_s1=3)
        cands = self.conn.execute("SELECT * FROM candidates WHERE source1_entity_id = 'S1-CAP'").fetchdf()
        self.assertEqual(len(cands), 3, "Should be capped to 3 candidates")

if __name__ == '__main__':
    unittest.main()
