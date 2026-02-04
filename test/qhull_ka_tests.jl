using Test
using KernelAbstractions

#include(joinpath(@__DIR__, "..", "src", "qhull_kernel_abstraction.jl"))
include(joinpath(@__DIR__, "..", "src", "MinMaxReduction.jl"))
include(joinpath(@__DIR__, "..", "src", "ScanPrimitive.jl"))
#using .qhull_kernel_abstraction
using .MinMaxReduction
using .ScanPrimitive

@testset "Segmented Scan" begin
    backend = CPU()
    @test segmented_scan(backend, [1, 2, 3, 4], [0, 0, 0, 0], (a, b) -> a + b) == [0, 1, 3, 6]
end

@testset "Min Max reduce" begin
    backend = CPU()
    data = [2, 4, 6, 8, 10]
    out = min_max_reduce(data, 16, backend)
    
    @test out.min == 2
    @test out.imin == 1
    @test out.max == 10
    @test out.imax == 5

    data = [1, 2, 3, 4, 0]
    out = min_max_reduce(data, 16, backend)
    
    @test out.min == 0
    @test out.imin == 5
    @test out.max == 4
    @test out.imax == 4

        
end