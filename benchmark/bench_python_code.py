import numpy as np
import pandas as pd
import time
import gc
from scipy.spatial import ConvexHull
from datetime import datetime

def run_and_save_benchmarks(version_name, n_dimension, n_sizes):
    results = []

    for n in n_sizes:
        data = np.random.rand(n, n_dimension).astype(np.float64)
        
        times = []
        for _ in range(5):
            gc.collect()
            start_time = time.perf_counter()
            
            hull = ConvexHull(data)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        median_time = np.median(times)
        
        results.append({
            "N": n,
            "Time_ms": median_time,
            "Allocs": 0,
            "Memory_MiB": 0.0 
        })
        
    df = pd.DataFrame(results)
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"bench_data_{version_name}_{date_str}.csv"
    df.to_csv(filename, index=False)
    
    return filename

n_range = [10**2, 10**3, 10**4, 10**5, 10**6]
n_dimension = 3

filename = run_and_save_benchmarks("Qhull_SciPy_V1", n_dimension, n_range)
