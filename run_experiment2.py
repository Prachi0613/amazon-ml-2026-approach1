import os
import gc
import time
import pandas as pd
import numpy as np
import random
from src.config import cfg, log_memory
from src.blocking import generate_candidates, evaluate_candidate_recall, COL_S1_ID

WORDS = ["amazon", "inc", "corp", "ltd", "tech", "data", "solutions", "global", "systems", "network", 
         "services", "cloud", "group", "holdings", "llc", "co", "enterprises", "media", "consulting",
         "main", "street", "park", "avenue", "road", "boulevard", "way", "drive", "lane", "place",
         "new", "york", "san", "francisco", "london", "paris", "tokyo", "delhi", "mumbai", "berlin"]

def generate_random_name(length=3):
    return " ".join(random.choices(WORDS, k=length))

def generate_synthetic_data(num_s1=500_000, num_s2=1_000_000, num_s3=1_000_000):
    print(f"Generating synthetic data: {num_s1} S1, {num_s2} S2, {num_s3} S3...")
    
    s1 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S1-{i}" for i in range(num_s1)],
        cfg.COL_NAME_NORM: [generate_random_name(random.randint(2, 4)) for _ in range(num_s1)],
        cfg.COL_ADDRESS_NORM: [generate_random_name(random.randint(3, 5)) for _ in range(num_s1)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s1)]
    })
    
    s2 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S2-{i}" for i in range(num_s2)],
        cfg.COL_NAME_NORM: [generate_random_name(random.randint(2, 4)) for _ in range(num_s2)],
        cfg.COL_ADDRESS_NORM: [generate_random_name(random.randint(3, 5)) for _ in range(num_s2)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s2)]
    })
    
    s3 = pd.DataFrame({
        cfg.COL_ENTITY_ID: [f"S3-{i}" for i in range(num_s3)],
        cfg.COL_NAME_NORM: [generate_random_name(random.randint(2, 4)) for _ in range(num_s3)],
        cfg.COL_ADDRESS_NORM: [generate_random_name(random.randint(3, 5)) for _ in range(num_s3)],
        cfg.COL_COUNTRY_NORM: ["us" for _ in range(num_s3)]
    })
    
    # Ground truth (inject matches)
    gt_list = []
    for i in range(num_s1):
        if i % 100 != 0:
            gt_list.append("")
        else:
            match_s2 = f"S2-{i % num_s2}"
            match_s3 = f"S3-{i % num_s3}"
            gt_list.append(f"{match_s2},{match_s3}")
            
            s2.loc[i % num_s2, cfg.COL_NAME_NORM] = s1.loc[i, cfg.COL_NAME_NORM]
            s3.loc[i % num_s3, cfg.COL_NAME_NORM] = s1.loc[i, cfg.COL_NAME_NORM]
            
    gt = pd.DataFrame({
        cfg.COL_GT_SOURCE1_ID: s1[cfg.COL_ENTITY_ID],
        cfg.COL_GT_MATCHED_IDS: gt_list
    })
    
    return s1, s2, s3, gt

def run():
    import psutil
    process = psutil.Process(os.getpid())
    peak_ram = 0

    def get_ram():
        nonlocal peak_ram
        r = process.memory_info().rss / (1024 * 1024)
        peak_ram = max(peak_ram, r)
        return r

    s1, s2, s3, gt = generate_synthetic_data(500_000, 1_000_000, 1_000_000)
    print(f"Data generated. RAM: {get_ram():.2f} MB")
    
    # Just run index building to get memory measurements for Ex2
    from src.blocking import _build_exact_index, _build_ngram_index
    
    t0 = time.time()
    exact_name_idx = _build_exact_index(s2, s3, cfg.COL_NAME_NORM, cfg)
    print(f"After exact-name index. RAM: {get_ram():.2f} MB")
    
    exact_addr_idx = _build_exact_index(s2, s3, cfg.COL_ADDRESS_NORM, cfg)
    print(f"After exact-address index. RAM: {get_ram():.2f} MB")
    
    name_vect, X_name_corpus, name_ids, name_sources = _build_ngram_index(s2, s3, cfg.COL_NAME_NORM, cfg)
    print(f"After ngram-name index. RAM: {get_ram():.2f} MB")
    
    addr_vect, X_addr_corpus, addr_ids, addr_sources = _build_ngram_index(s2, s3, cfg.COL_ADDRESS_NORM, cfg)
    print(f"After ngram-address index. RAM: {get_ram():.2f} MB")
    t1 = time.time()
    
    print("\n" + "="*40)
    print("EXPERIMENT 2 RESULTS")
    print("="*40)
    print(f"Peak RAM: {peak_ram:.2f} MB")
    print(f"Index build time: {t1 - t0:.2f} s")

if __name__ == '__main__':
    run()
