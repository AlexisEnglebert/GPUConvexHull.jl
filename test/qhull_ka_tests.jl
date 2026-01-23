using Test

include(joinpath(@__DIR__, "..", "src", "qhull_kernel_abstraction.jl"))
using .qhull_kernel_abstraction

@testset "Segmented Scan" begin
    @test segmented_scan([1, 2, 3, 4], [0, 0, 0, 0], (a, b) -> a + b) == [0, 1, 3, 6]
end