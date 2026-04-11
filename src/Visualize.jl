using GLMakie
using KernelAbstractions
using DelaunayTriangulation
import GeometryBasics: Mesh, Point3f, TriangleFace

include(joinpath(@__DIR__, "..", "src", "qhull_kernel_abstraction.jl"))
using .qhull_kernel_abstraction

# Lecture du fichier (Matrice dim x N)
points = qhull_kernel_abstraction.read_input_file("test_3d_input.txt")
dim, pt_cnt = size(points)

# Calcul de l'enveloppe via ton Kernel
gpu_pts = KernelAbstractions.zeros(CPU(), Float64, (dim, pt_cnt))
copyto!(gpu_pts, points)
convex_hull = qhull_kernel_abstraction.quick_hull(gpu_pts, pt_cnt, dim)

# Préparation des données pour Makie
all_points  = [Point3f(points[:, i]) for i in 1:size(points, 2)]

hull_pts_matrix = Array(convex_hull)
hull_nodes = [Point3f(hull_pts_matrix[:, i]) for i in 1:size(hull_pts_matrix, 2)]

fig = Figure(resolution = (1200, 800))
ax  = Axis3(fig[1, 1], title = "Résultat QuickHull GPU", aspect = :data)

#mesh!(ax, m, color = (:orange, 0.4), shading = MultiLightShading)
#wireframe!(ax, m, color = :black, linewidth = 1)

scatter!(ax, hull_nodes, color = :red,            markersize = 12,  label = "Sommets Enveloppe")
scatter!(ax, all_points, color = :grey,    markersize = 12,   label = "Points Internes")

axislegend(ax)
DataInspector(fig)
display(fig)

if !isinteractive()
    println("Fenêtre ouverte. Appuyez sur Entrée pour terminer.")
    readline()
end