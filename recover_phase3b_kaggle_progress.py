import logging
import duckdb
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "output/entity_resolution.duckdb"

def main(db_path=DB_PATH):
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return False
        
    logger.info(f"Opening database: {db_path}")
    conn = duckdb.connect(db_path)
    
    try:
        # Check if checkpoints table exists
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name = 'checkpoints'").fetchall()
        if not tables:
            logger.error("checkpoints table does not exist!")
            return False
            
        # 1. Print current checkpoint schema
        schema = conn.execute("DESCRIBE checkpoints").fetchall()
        logger.info(f"Current checkpoints schema: {schema}")
        
        # Determine column name
        col_name = schema[0][0]
        logger.info(f"Checkpoint column name is: {col_name}")
        
        # 2. Print existing candidates count
        try:
            cand_count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            logger.info(f"Existing candidate count: {cand_count}")
        except duckdb.CatalogException:
            logger.warning("candidates table does not exist.")
            
        # 3. Print existing prefix4_house candidate count
        try:
            prefix4_count = conn.execute("SELECT COUNT(*) FROM candidates WHERE from_name_block = TRUE AND matched_source = 's2'").fetchone()[0]
            logger.info(f"Existing prefix4_house_s2 candidate count: {prefix4_count}")
        except duckdb.CatalogException:
            pass
            
        # 4. Generate the 12 markers
        markers = []
        for batch_idx in range(12):
            offset = batch_idx * 50_000
            markers.append(f"prefix4_house_s2_batch_{batch_idx}_offset_{offset}")
            
        # 5. Check if they already exist
        for marker in markers:
            exists = conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE {col_name} = ?", [marker]).fetchone()[0]
            logger.info(f"Marker '{marker}' exists: {bool(exists)}")
            
        # 6. Insert the markers idempotently
        logger.info("Inserting recovery markers...")
        for marker in markers:
            # DuckDB supports ON CONFLICT (col) DO NOTHING
            conn.execute(f"""
                INSERT INTO checkpoints ({col_name}) 
                VALUES (?) 
                ON CONFLICT ({col_name}) DO NOTHING
            """, [marker])
            
        # 7. Verify all 12 markers
        logger.info("Verifying markers...")
        success = True
        for marker in markers:
            exists = conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE {col_name} = ?", [marker]).fetchone()[0]
            if not exists:
                logger.error(f"Failed to verify marker: {marker}")
                success = False
            else:
                logger.info(f"Verified successfully: {marker}")
                
        # 8. Check for COMPLETE marker just to assure we didn't add it
        complete_exists = conn.execute(f"SELECT COUNT(*) FROM checkpoints WHERE {col_name} = 'prefix4_house_s2_COMPLETE'").fetchone()[0]
        logger.info(f"Pass-level 'prefix4_house_s2_COMPLETE' exists: {bool(complete_exists)}")
        
        if success:
            logger.info("Recovery successful! 12 batch checkpoints injected safely.")
            return True
        else:
            logger.error("Recovery verification failed.")
            return False
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()
