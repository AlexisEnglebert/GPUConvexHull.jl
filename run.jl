#using GLMakie
using KernelAbstractions
using GPUConvexHull


backend = CPU()

n_points = 100_000
n_dim    = 3 
data = rand(n_dim, n_points)

# Calcul de l'enveloppe via ton Kernel
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)

simplex_idx = GPUConvexHull.quick_hull(backend, gpu_pts)
