using Test
using GPUConvexHull
using KernelAbstractions

function test_qhull(points, backend, expected)
    gpu_pts = KernelAbstractions.zeros(backend, Float64, size(points))
    copyto!(gpu_pts, points)

    println("size: ", size(gpu_pts))
    @show KernelAbstractions.get_backend(gpu_pts)
    sol = GPUConvexHull.quick_hull(backend, gpu_pts)
    @test sol == expected
end

@testset "Cube" begin
    points = [-1.0 0.0 1.0 0.0;
              0.0 1.0 0.0 -1.0]
    expected = [0.0  0.0  -1.0  1.0;
                -1.0  1.0   0.0  0.0]
    test_qhull(points, CPU(), expected)
end

@testset "2D with colinear points " begin
    points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0;
               2.0  0.0 -2.0  0.0  3.0  1.0 ]

    test_qhull(points, CPU(), points)
end