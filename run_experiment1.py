import os
import gc
import time
import pandas as pd
import numpy as np
import random
from src.config import cfg, log_memory
from src.blocking import generate_candidates, evaluate_candidate_recall, COL_S1_ID

# Generate realistic-ish synthetic data
WORDS = ["amazon", "inc", "corp", "ltd", "tech", "data", "solutions", "global", "systems", "network", 
         "services", "cloud", "group", "holdings", "llc", "co", "enterprises", "media", "consulting",
         "main", "street", "park", "avenue", "road", "boulevard", "way", "drive", "lane", "place",
         "new", "york", "san", "francisco", "london", "paris", "tokyo", "delhi", "mumbai", "berlin"]

def generate_random_name(length=3):
    return " ".join(random.choices(WORDS, k=length))

def generate_synthetic_data(num_s1=100_000, num_s2=250_000, num_s3=250_000):
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
    
    # Ground truth (inject ~10% exact matches)
    gt_list = []
    for i in range(num_s1):
        if i % 10 != 0:
            gt_list.append("")
        else:
            match_s2 = f"S2-{i % num_s2}"
            match_s3 = f"S3-{i % num_s3}"
            gt_list.append(f"{match_s2},{match_s3}")
            
            # ensure it's a match
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

    s1, s2, s3, gt = generate_synthetic_data(100_000, 250_000, 250_000)
    print(f"Data generated. RAM: {get_ram():.2f} MB")
    
    t0 = time.time()
    candidates = generate_candidates(s1, s2, s3, cfg)
    t1 = time.time()
    
    retrieval_time = t1 - t0
    get_ram()
    
    cand_counts = candidates.groupby(COL_S1_ID).size()
    mean_cands = cand_counts.mean() if not cand_counts.empty else 0
    p95_cands = cand_counts.quantile(0.95) if not cand_counts.empty else 0
    
    from src.blocking import parse_ground_truth_matches
    gt_dict = parse_ground_truth_matches(gt, cfg)
    c_recall_stats = evaluate_candidate_recall(candidates, gt_dict, cfg)[0]
    
    print("\n" + "="*40)
    print("EXPERIMENT 1 RESULTS")
    print("="*40)
    print(f"Peak RAM: {peak_ram:.2f} MB")
    print(f"Retrieval Time (including index build): {retrieval_time:.2f} s")
    print(f"Candidate count: {len(candidates)}")
    print(f"Candidate recall: {c_recall_stats['candidate_recall']:.4f}")
    print(f"Mean candidates/S1: {mean_cands:.2f}")
    print(f"P95 candidates/S1: {p95_cands:.2f}")

if __name__ == '__main__':
    run()
