#using GLMakie
using KernelAbstractions
using GPUConvexHull
using TimerOutputs
using CUDA
using Profile
using FlameGraphs
using FileIO
using PProf

backend = CUDABackend()

# Warm up car ou sinon mon timer est faussé par l'optimisation en runtime.
n_points = 10
n_dim    = 3 
data = rand(n_dim, n_points)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)
GPUConvexHull.quick_hull(backend, data)

Profile.clear()

n_points = 100_000
n_dim    = 4
data = rand(n_dim, n_points)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)

reset_timer!(GPUConvexHull.to)
GPUConvexHull.quick_hull(backend, data)
print_timer(GPUConvexHull.to) 

#pprof()
end