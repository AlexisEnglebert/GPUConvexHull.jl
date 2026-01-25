using Test
using KernelAbstractions

include(joinpath(@__DIR__, "..", "src", "qhull_kernel_abstraction.jl"))
using .qhull_kernel_abstraction

@testset "Segmented Scan" begin
    @test segmented_scan([1, 2, 3, 4], [0, 0, 0, 0], (a, b) -> a + b) == [0, 1, 3, 6]
end

@testset "Min Max reduce" begin
    backend = CPU()
    min_max_values_ker = minmax_reduce(backend, 16)
    data = [2, 4, 6, 8, 10]
    out = [(Inf64, Inf64)]
    min_max_values_ker(data, out, ndrange=length(data))
    @test out == [(2, 1)]
    data = [1, 2, 3, 4, 0]
    min_max_values_ker(data, out, ndrange=length(data))
    @test out  == [(0, 5)]
    
end