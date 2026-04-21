using Test
using GPUConvexHull
using KernelAbstractions

function test_qhull(points, backend, expected)
    gpu_pts = KernelAbstractions.zeros(backend, Float64, size(points))
    copyto!(gpu_pts, points)

    println("size: ", size(gpu_pts))
    @show KernelAbstractions.get_backend(gpu_pts)
    @show sol = GPUConvexHull.quick_hull(backend, gpu_pts)
    @test sol.hull_points == expected
end

@testset "2D square" begin
    points = [-1.0 0.0 1.0 0.0
              0.0 1.0 0.0 -1.0]

    expected = [0.0  1.0  0.0  -1.0
               -1.0  0.0  1.0  0.0]
    test_qhull(points, CPU(), expected)
end

@testset "Simple 3D shape" begin # https://www.geogebra.org/3d/m5gdytzk
    points = [-4 -5 -3 -3 2 1 -5 -1 -1 -2
               3  1  1  4 1 -5 -3 -3 1 -1
              -3 -3 -3  1 1 -4 3  -1 1 0]

    expected = [ -5.0   1.0  2.0  -3.0  -4.0  -5.0
                 -3.0  -5.0  1.0   4.0   3.0   1.0
                  3.0  -4.0  1.0   1.0  -3.0  -3.0]

    test_qhull(points, CPU(), expected)
end


@testset "2D with colinear points " begin
    points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0 0.0
               2.0  0.0 -2.0  0.0  3.0  1.0 0.0]
    # Note: the point (-1, 1) is colinear with the edge (0,2)-(2,0) and should not be part of the convex hull. Since we wants the minimum set of points.
    expected = [ -2.0 0.0  2.0 3.0 0.0 
                  0.0 -2.0 0.0 3.0 2.0] 
    test_qhull(points, CPU(), expected)

end