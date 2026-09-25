import os
import gc
import pandas as pd
import numpy as np
import string
import random

from src.config import cfg, log_memory
from src.blocking import generate_candidates, COL_S1_ID, COL_CAND_ID, COL_CAND_SRC
from src.features import generate_features
from src.model import EntityMatcher, build_candidate_labels

def random_string(length):
    return ''.join(random.choices(string.ascii_lowercase + " ", k=length))

def generate_synthetic_data(num_s1=2000, num_s2=2000, num_s3=2000):
    print(f"Generating synthetic data: {num_s1} S1, {num_s2} S2, {num_s3} S3...")
    
    # S1
    s1 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S1-{i}" for i in range(num_s1)],
        cfg.COL_NAME_NORM: [random_string(15) for _ in range(num_s1)],
        cfg.COL_ADDRESS_NORM: [random_string(25) for _ in range(num_s1)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s1)]
    })
    
    # S2
    s2 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S2-{i}" for i in range(num_s2)],
        cfg.COL_NAME_NORM: [random_string(15) for _ in range(num_s2)],
        cfg.COL_ADDRESS_NORM: [random_string(25) for _ in range(num_s2)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s2)]
    })
    
    # S3
    s3 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S3-{i}" for i in range(num_s3)],
        cfg.COL_NAME_NORM: [random_string(15) for _ in range(num_s3)],
        cfg.COL_ADDRESS_NORM: [random_string(25) for _ in range(num_s3)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s3)]
    })
    
    # Ground truth (some matches)
    gt_list = []
    for i in range(num_s1):
        if i % 10 != 0:
            gt_list.append("")
        else:
            match_s2 = f"S2-{i % num_s2}"
            match_s3 = f"S3-{i % num_s3}"
            gt_list.append(f"{match_s2},{match_s3}")
            
    gt = pd.DataFrame({
        cfg.COL_GT_SOURCE1_ID: s1[cfg.COL_ENTITY_ID],
        cfg.COL_GT_MATCHED_IDS: gt_list
    })
    
    return s1, s2, s3, gt

def run_stress_test():
    log_memory("Start")
    s1, s2, s3, gt = generate_synthetic_data(num_s1=1000, num_s2=2000, num_s3=2000)
    
    # Artificially inject exact matches for recall testing
    for i in range(10):
        s1.loc[i, cfg.COL_NAME_NORM] = "exact match company inc"
        s2.loc[i, cfg.COL_NAME_NORM] = "exact match company inc"
        s3.loc[i, cfg.COL_NAME_NORM] = "exact match company inc"
    
    log_memory("After synthetic data generation")
    
    print("\n--- Testing Blocking ---")
    candidates = generate_candidates(s1, s2, s3, cfg)
    print(f"Candidates generated: {len(candidates)}")
    log_memory("After blocking")
    
    print("\n--- Testing Labels ---")
    y_train = build_candidate_labels(candidates, gt, cfg)
    print(f"Labels generated: {len(y_train)}, Positives: {y_train.sum()}")
    log_memory("After labels")
    
    print("\n--- Testing Feature Generation ---")
    X_train = generate_features(s1, s2, s3, candidates, cfg)
    print(f"Features generated: {X_train.shape}")
    log_memory("After features")
    
    print("\n--- Testing LightGBM ---")
    matcher = EntityMatcher(cfg)
    
    # We pass the same for val just to test it
    matcher.fit(X_train, y_train, X_val=X_train.copy(), y_val=y_train.copy())
    print("Model trained successfully!")
    log_memory("After LightGBM")
    
    print("\nAll memory stress tests passed!")

if __name__ == '__main__':
    run_stress_test()
