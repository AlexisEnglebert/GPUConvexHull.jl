#using GLMakie
using KernelAbstractions
using GPUConvexHull
using BenchmarkTools

backend = CPU()

n_points = 100_000

data1_gpu = KernelAbstractions.allocate(backend, Int64, n_points)
data2_gpu = KernelAbstractions.allocate(backend, Int64, n_points)

data2 = rand(1:10, n_points)
data1 = rand(Bool, n_points)

copy!(data1_gpu, data1)
copy!(data2_gpu, data2)

mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 256, n_points) 

context = GPUConvexHull.create_quickhull_context(backend, 256)

res = @benchmark GPUConvexHull.flag_permute(context.backend, $mem_data, $data2_gpu
        ,$data1_gpu, n_points, 10)

display(res)