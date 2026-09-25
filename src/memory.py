import os
import psutil

def get_memory_stats():
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    rss_gb = rss_mb / 1024.0
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 * 1024 * 1024)
    return rss_gb, avail_gb

def get_directory_size_gb(path: str) -> float:
    total = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
    return total / (1024 * 1024 * 1024)

def log_memory_state(stage: str, db_path: str = None, temp_dir: str = None):
    rss_gb, avail_gb = get_memory_stats()
    db_gb = get_directory_size_gb(os.path.dirname(db_path)) if db_path else 0.0
    tmp_gb = get_directory_size_gb(temp_dir) if temp_dir else 0.0
    
    print(f"[MEMORY]")
    print(f"stage={stage}")
    print(f"rss_gb={rss_gb:.3f}")
    print(f"available_gb={avail_gb:.3f}")
    print(f"db_gb={db_gb:.3f}")
    print(f"tmp_gb={tmp_gb:.3f}")
    print("-" * 40)
