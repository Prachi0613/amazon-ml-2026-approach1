import os
import gc
import sys
import pandas as pd
import numpy as np
from src.config import cfg
from src.blocking import generate_candidates
import string

def run_test(num_s1, num_s2, num_s3, test_name):
    print(f"\n{'='*40}")
    print(f"RUNNING {test_name}")
    print(f"S1={num_s1}, S2={num_s2}, S3={num_s3}")
    print(f"{'='*40}")
    
    # Generate unique strings to avoid cross-product explosion in exact match
    s1 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S1-{i}" for i in range(num_s1)],
        cfg.COL_NAME_NORM: [f"name_{i}_a b c" for i in range(num_s1)],
        cfg.COL_ADDRESS_NORM: [f"addr_{i}_x y z" for i in range(num_s1)],
        cfg.COL_COUNTRY_NORM: ["us"] * num_s1
    })
    
    s2 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S2-{i}" for i in range(num_s2)],
        cfg.COL_NAME_NORM: [f"name2_{i}_a b c" for i in range(num_s2)],
        cfg.COL_ADDRESS_NORM: [f"addr2_{i}_x y z" for i in range(num_s2)],
        cfg.COL_COUNTRY_NORM: ["us"] * num_s2
    })
    
    s3 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S3-{i}" for i in range(num_s3)],
        cfg.COL_NAME_NORM: [f"name3_{i}_a b c" for i in range(num_s3)],
        cfg.COL_ADDRESS_NORM: [f"addr3_{i}_x y z" for i in range(num_s3)],
        cfg.COL_COUNTRY_NORM: ["us"] * num_s3
    })
    
    import diagnostic_tracker
    diagnostic_tracker.stages.clear()
    
    try:
        cands = generate_candidates(s1, s2, s3, cfg)
    except Exception as e:
        print(f"Failed: {e}")
        
if __name__ == '__main__':
    run_test(10_000, 50_000, 50_000, "TEST A")
    run_test(25_000, 100_000, 100_000, "TEST B")
