using BenchmarkTools
using GPUConvexHull
using KernelAbstractions
using DataFrames, CSV, Dates

function run_and_save_benchmarks(version_name, n_dimension, N_sizes)
    df = DataFrame(N = Int[], Time_ms = Float64[], Allocs = Int[], Memory_MiB = Float64[])

    for N in N_sizes
        data = rand(n_dimension, N)
        b = @benchmark GPUConvexHull.quick_hull(CPU(), $data) samples=5 evals=1
        
        push!(df, (
            N = N, 
            Time_ms = median(b).time / 1e6, 
            Allocs = b.allocs, 
            Memory_MiB = b.memory / 1024^2
        ))
        println("Fini pour N=$N")
    end

    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    filename = "bench_data_$(version_name)_$(date_str).csv"
    CSV.write(filename, df)
    return filename
end

function save_benchmark_results(p, version_name)
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    filename = "bench_$(version_name)_$(date_str).png"
    savefig(p, filename)
end

n_range = [10^2, 10^3, 10^4, 10^5, 10^6]
n_dimension = 3
p = run_and_save_benchmarks("V1", n_dimension, n_range)
