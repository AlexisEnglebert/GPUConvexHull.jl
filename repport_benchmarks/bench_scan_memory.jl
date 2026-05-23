using GPUConvexHull  
using KernelAbstractions
using DataFrames, CSV, Dates
using BenchmarkTools

backend = CPU()

function open_file(path)
    return parse.(Int, readlines(path))
end

data_size = [10, 100, 1000, 10_000, 100_000, 1_000_000]
n_calls = 10_000
workgroup_size = 256

df = DataFrame(N = Int[], Time_ms = Float64[])


for sz in data_size
    cpu_array = open_file("scan_data/$(sz)_input.txt")
    gpu_array = KernelAbstractions.allocate(backend, Int64, sz)
    copy!(gpu_array, cpu_array)
    segment_mem_data_int = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, workgroup_size, sz)
    #Warmup
    GPUConvexHull.ScanPrimitive.scan(segment_mem_data_int, gpu_array, GPUConvexHull.ScanPrimitive.AddOp())
    GPUConvexHull.ScanPrimitive.scan(segment_mem_data_int, gpu_array, GPUConvexHull.ScanPrimitive.AddOp())
    

    b = @benchmark GPUConvexHull.ScanPrimitive.scan($segment_mem_data_int, $gpu_array, GPUConvexHull.ScanPrimitive.AddOp()) samples=n_calls evals=1

    push!(df, (
            N = sz, 
            Time_ms = median(b).time / 1e6, 
    ))

end 

date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM")
filename = "scan_memory_prealloc_benchmark_$(date_str).csv"
CSV.write(filename, df)