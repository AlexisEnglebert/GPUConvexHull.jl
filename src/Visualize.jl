#using GLMakie
using KernelAbstractions
import GeometryBasics: Mesh, Point3f, TriangleFace

include(joinpath(@__DIR__, "..", "src", "qhull_kernel_abstraction.jl"))
using .qhull_kernel_abstraction

# Lecture du fichier (Matrice dim x N)
points = qhull_kernel_abstraction.read_input_file("test_3d_input.txt")
dim, pt_cnt = size(points)

# Calcul de l'enveloppe via ton Kernel
gpu_pts = KernelAbstractions.zeros(CPU(), Float64, (dim, pt_cnt))
copyto!(gpu_pts, points)
simplex_idx = qhull_kernel_abstraction.quick_hull(gpu_pts, pt_cnt, dim)
