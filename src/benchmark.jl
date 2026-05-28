include("QuickhullProfiler.jl")

using CUDA, KernelAbstractions
using GPUConvexHull
using AcceleratedKernels

using Printf

println("Warmup...")
GPUConvexHull.quick_hull_profiled(CUDABackend(), randn(3, 1000), Profiler())
println("Go !\n")


# ─── Lance ───────────────────────────────────────────────────
for n in [10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 2*10^7, 3*10^7, 4*10^7, (4*10^7)+5*10^6, 5*10^7] 
    pts  = GPUConvexHull.read_input_file("../benchmark/data/points_$(n)_d$(3).txt")
    prof = Profiler()
    GPUConvexHull.quick_hull_profiled(CUDABackend(), pts, prof)
    print_report(prof)
    save_csv(prof, "breakdown_N$(n)_D3.csv")
end