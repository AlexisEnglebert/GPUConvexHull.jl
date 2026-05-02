#using GLMakie
using KernelAbstractions
using GPUConvexHull
using TimerOutputs
using CUDA

backend = CUDABackend()

# Warm up car ou sinon mon timer est faussé par l'optimisation en runtime.
n_points = 1000
n_dim    = 3 
data = rand(n_dim, n_points)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)
GPUConvexHull.quick_hull(backend, gpu_pts)

n_points = 1_000_000
n_dim    = 3
data = rand(n_dim, n_points)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)

reset_timer!(GPUConvexHull.to)
simplex_idx = GPUConvexHull.quick_hull(backend, gpu_pts)
print_timer(GPUConvexHull.to)