using GPUConvexHull  
using KernelAbstractions
using DataFrames, CSV, Dates
using BenchmarkTools
using CUDA

backend = CUDABackend()

function open_file(path)
    return parse.(Int, readlines(path))
end

data_size = [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000] 
n_calls = 10_000
workgroup_size = 256

df = DataFrame(N = Int[], Time_ms = Float64[], time_std = Float64[])


for sz in data_size
    all_array = open_file("scan_data/$(sz)_input.txt")
    gpu_array = KernelAbstractions.allocate(backend, Int64, sz)
    gpu_flags = KernelAbstractions.allocate(backend, Int64, sz)
    copy!(gpu_array, all_array[1:sz])
    copy!(gpu_flags, all_array[sz+1:sz*2])

    segment_mem_data_int = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, workgroup_size, sz)
    #Warmup
    GPUConvexHull.ScanPrimitive.segmented_scan(segment_mem_data_int, gpu_array, gpu_flags, GPUConvexHull.ScanPrimitive.AddOp())
    GPUConvexHull.ScanPrimitive.segmented_scan(segment_mem_data_int, gpu_array, gpu_flags, GPUConvexHull.ScanPrimitive.AddOp())
    
    run_times = Float64[]
    for _ in 1:n_calls
        t = @elapsed GPUConvexHull.ScanPrimitive.segmented_scan(segment_mem_data_int, gpu_array, gpu_flags, GPUConvexHull.ScanPrimitive.AddOp())
        push!(run_times, t)
    end
    push!(df, (
    N = sz, 
    Time_ms = mean(run_times) * 1000, 
    time_std  = std(run_times) * 1000
))

end 

date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM")
filename = "scan_memory_prealloc_benchmark_$(date_str).csv"
CSV.write(filename, df)