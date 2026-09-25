import os
import psutil

process = psutil.Process(os.getpid())
stages = []

def track(operation, phase):
    rss = process.memory_info().rss / (1024 * 1024)
    stages.append({"operation": operation, "phase": phase, "rss": rss})
    
def print_report():
    import psutil
    vm = psutil.virtual_memory()
    print("operation, RSS before, RSS after, delta, system available RAM")
    
    # group by operation
    ops = {}
    for s in stages:
        op = s["operation"]
        if op not in ops:
            ops[op] = {}
        ops[op][s["phase"]] = s["rss"]
        
    for op, data in ops.items():
        if "before" in data and "after" in data:
            before = data["before"]
            after = data["after"]
            delta = after - before
            avail = vm.available / (1024 * 1024)
            print(f"{op} | {before:.2f} MB | {after:.2f} MB | {delta:.2f} MB | {avail:.2f} MB")
