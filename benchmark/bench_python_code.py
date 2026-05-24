import numpy as np
import pandas as pd
import time
import gc
from scipy.spatial import ConvexHull
from datetime import datetime

def read_file(filename):
    with open(filename, 'r') as f:
            ligne1 = f.readline().split()
            n = int(ligne1[0])
            d = int(ligne1[1])
            
            data = np.loadtxt(f)
            
    return n, d, data
        

def run_and_save_benchmarks(version_name, n_dimension, n_sizes):
    results = []

    for n in n_sizes:
        sz, dim, data = read_file(f"data/points_{n}_d{n_dimension}.txt")
        
        times = []
        for _ in range(5):
            gc.collect()
            start_time = time.perf_counter()
            
            hull = ConvexHull(data)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        median_time = np.median(times)
        std_time = np.std(times)
        
        results.append({
            "N": n,
            "Time_ms": median_time,
            "Allocs": 0,
            "Memory_MiB": 0.0,
            "Time_std": std_time
        })
        
    df = pd.DataFrame(results)
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"bench_data_{version_name}_{date_str}.csv"
    df.to_csv(filename, index=False)
    
    return filename

n_range = [10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 2*(10**7), 3*(10**7), 4*(10**7)]
n_dimension = 3

filename = run_and_save_benchmarks("Qhull_SciPy_V1", n_dimension, n_range)