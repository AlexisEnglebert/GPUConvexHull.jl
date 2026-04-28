using Test
using GPUConvexHull
using KernelAbstractions

function test_qhull(points, backend, expected)
    gpu_pts = KernelAbstractions.zeros(backend, Float64, size(points))
    copyto!(gpu_pts, points)

    sol = GPUConvexHull.quick_hull(backend, gpu_pts)

    sol_set = Set(eachcol(round.(sol.hull_points, digits=9)))
    expected_set = Set(eachcol(round.(expected, digits=9)))
    @show sol_set
    @test isempty(setdiff(sol_set, expected_set)) && isempty(setdiff(expected_set, sol_set))
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

@testset "2D lots of points inside a square" begin
    corners = [-1.0  1.0  1.0 -1.0
               -1.0 -1.0  1.0  1.0]

    internal_points = (rand(2, 100) .- 0.5) .* 1.5

    points = hcat(corners, internal_points)
    expected = corners

    test_qhull(points, CPU(), expected)
end

@testset "2D circle" begin
    theta = range(0, 2pi, length=17)[1:end-1]

    points = hcat([cos(t) for t in theta], [sin(t) for t in theta])'
    expected = points

    test_qhull(points, CPU(), expected)
end

# TODO: parfois il fail, faut investiguer... (à mon avis c'est quand les points sont trop proches (internal pts = 500))
@testset "3D cube with lots of points inside" begin 

    corners = [-1.0  1.0 -1.0  1.0 -1.0  1.0 -1.0  1.0
               -1.0 -1.0  1.0  1.0 -1.0 -1.0  1.0  1.0
               -1.0 -1.0 -1.0 -1.0  1.0  1.0  1.0  1.0]

    internal_points = (rand(3, 50) .- 0.5) .* 1.8

    points = hcat(internal_points, corners)
    expected = corners

    test_qhull(points, CPU(), expected)
end

@testset "4D Tesseract with lots of points inside" begin
    corners = zeros(Float64, 4, 16)
    idx = 1
    for x in [-1.0, 1.0], y in [-1.0, 1.0], z in [-1.0, 1.0], w in [-1.0, 1.0]
        corners[:, idx] = [x, y, z, w]
        idx += 1
    end

    internal_points = (rand(4, 1000) .- 0.5) .* 1.6

    points = hcat(corners, internal_points)
    expected = corners

    test_qhull(points, CPU(), expected)
end

 #=@testset "2D square (All points on the edges)" begin #TODO: Foire ce test, la sortie est correcte mais certains points pourraient être discord (colinéaire avec le convexhull)
    points = zeros(Float64, 2, 25)
    idx = 1
    for x in -2:2, y in -2:2
        points[:, idx] = [x, y]
        idx += 1
    end

    expected = [-2.0  2.0  2.0 -2.0
                -2.0 -2.0  2.0  2.0]
    
    test_qhull(points, CPU(), expected)
end =#

@testset "2D rotated square with points inside" begin
    points = [ 0.0  1.0  0.0 -1.0  0.5  0.5 -0.5 -0.5  0.0
               1.0  0.0 -1.0  0.0  0.5 -0.5 -0.5  0.5  0.0]
    
    expected = [ 0.0  1.0  0.0 -1.0;
                 1.0  0.0 -1.0  0.0]

    test_qhull(points, CPU(), expected)
end

@testset "2D Triangle n==dim" begin
    points = [0.0  1.0  0.5  0.5
              0.0  0.0  1.0  0.2]
              
    expected = [0.0  1.0  0.5;
                0.0  0.0  1.0]
    test_qhull(points, CPU(), expected)
end

@testset "multiple same points" begin
    points = fill(3.14, 2, 10)
    expected = fill(3.14, 2, 1)
    
    test_qhull(points, CPU(), expected)
end