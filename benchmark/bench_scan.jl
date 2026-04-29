#using GLMakie
using KernelAbstractions
using GPUConvexHull
using BenchmarkTools

backend = CPU()

n_points = 20_000_000
data1 = rand(n_points)


data1_gpu = KernelAbstractions.allocate(backend, Float64, n_points)
data2_gpu = KernelAbstractions.allocate(backend, Int64, n_points)
data2 = rand(Bool, n_points)

copy!(data1_gpu, data1)
copy!(data2_gpu, data2)

mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Float64, Int64, 256, n_points) 

res = @benchmark GPUConvexHull.segmented_scan(mem_data , $data1_gpu, $data2_gpu, GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0)
display(res)

res = @benchmark GPUConvexHull.segmented_scan(mem_data , $data1_gpu, $data2_gpu, GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0, inclusive=true)
display(res)

res = @benchmark GPUConvexHull.segmented_scan(mem_data , $data1_gpu, $data2_gpu, GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0, backward=true)
display(res)

res = @benchmark GPUConvexHull.segmented_scan(mem_data , $data1_gpu, $data2_gpu, GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0, backward=true, inclusive=true)
display(res)