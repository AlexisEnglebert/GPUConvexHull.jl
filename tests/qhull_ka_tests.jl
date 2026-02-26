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
    @testset "exclusive forward scan NO segment" begin
        @test segmented_scan(backend, [1, 2, 3, 4], [1, 0, 0, 0], (a, b) -> a + b) == [0, 1, 3, 6]
    end
    @testset "exclusive forward scan with segments" begin
        @test segmented_scan(backend, [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], (a, b) -> a + b) == [0, 1, 2, 0, 1, 0]
    end
    @testset "inclusive forward scan NO segments" begin
        @test segmented_scan(backend, [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], (a, b) -> a + b, false, true) == [1, 2, 3, 4, 5, 6]
    end
    @testset "inclusive forward scan with segments" begin
        @test segmented_scan(backend, [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], (a, b) -> a + b, false, true) == [1, 2, 3, 1, 2, 1]
    end
    @testset "exclusive backward scan NO segments" begin
        @test segmented_scan(backend, [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0], (a, b) -> a + b, true, false) == [3, 2, 2, 2, 1, 0]
    end
    @testset "exclusive backward scan with segments" begin
        @test segmented_scan(backend, [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0], (a, b) -> a + b, true, false) == [1, 0, 0, 2, 1, 0]
    end
    @testset "inclusive backward scan NO segments" begin
        @test segmented_scan(backend, [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0], (a, b) -> a + b, true, true) == [4, 3, 2, 2, 2, 1]
    end
    @testset "inclusive backward scan with segments" begin
        @test segmented_scan(backend, [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0], (a, b) -> a + b, true, true) == [2, 1, 0, 2, 2, 1]
    end
 
end

@testset "Min Max reduce" begin
    backend = CPU()
    
    @testset "default minmax" begin
        data = [3, 2, 6, 10, 8]
        out = min_max_reduce(data, 16, backend)
        
        @test out.min == 2
        @test out.imin == 2
        @test out.max == 10
        @test out.imax == 4
    end

    @testset "minmax at the edges of the array" begin
        data = [1, 2, 3, 4, 0]
        out = min_max_reduce(data, 16, backend)
        
        @test out.min == 0
        @test out.imin == 5
        @test out.max == 4
        @test out.imax == 4
    end

    @testset "single ellement in array" begin
        data = [1]
        out = min_max_reduce(data, 16, backend)

        @test out.min == 1
        @test out.imin == 1
        @test out.max == 1
        @test out.imax == 1
    end

    @testset "default multi-block minmax" begin
        data = [4, 1, 3, 4, 5, 7, 4]
        out = min_max_reduce(data, 4, backend)

        @test out.min == 1
        @test out.imin == 2
        @test out.max == 7
        @test out.imax == 6
    end 

    @testset "multi-block minmax at the edge of the array" begin
        data = [0, 1, 3, 4, 5, 7, 10]
        out = min_max_reduce(data, 4, backend)

        @test out.min == 0
        @test out.imin == 1
        @test out.max == 10
        @test out.imax == 7
    end 

    @testset "multiple min" begin
        data = [1, 1, 1, 1, 2]
        out = min_max_reduce(data, 4, backend)

        @test out.min == 1
        @test out.imin == 1
        @test out.max == 2
        @test out.imax == 5
    end

    @testset "multuple max" begin
        data = [1, 2, 3 ,4, 5, 5]
        out = min_max_reduce(data, 4, backend)

        @test out.min == 1
        @test out.imin == 1
        @test out.max == 5
        @test out.imax == 5
    end

end