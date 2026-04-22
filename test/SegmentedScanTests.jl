using Test
using KernelAbstractions
using GPUConvexHull

@testset "Segmented Scan" begin
    backend = CPU()
    @testset "exclusive forward scan NO segment" begin
        mem_data = GPUConvexHull.ScanPrimitive.allocate_scan_memory_arrays(backend, Int64, Int64, 8, 4)
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 2, 3, 4], [1, 0, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0)) == [0, 1, 3, 6]
    end
    mem_data = GPUConvexHull.ScanPrimitive.allocate_scan_memory_arrays(backend, Int64, Int64, 8, 6)
    @testset "exclusive forward scan with segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], GPUConvexHull.ScanPrimitive.AddOp(), identity=0)) == [0, 1, 2, 0, 1, 0]
    end
    @testset "inclusive forward scan NO segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=false, inclusive=true)) == [1, 2, 3, 4, 5, 6]
    end
    @testset "inclusive forward scan with segments" begin
         @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=false, inclusive=true)) == [1, 2, 3, 1, 2, 1]
    end
    @testset "exclusive backward scan NO segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=false)) == [3, 2, 2, 2, 1, 0]
    end
    @testset "exclusive backward scan with segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=false)) == [1, 0, 0, 2, 1, 0]
    end
    @testset "inclusive backward scan NO segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=true)) == [4, 3, 2, 2, 2, 1]
    end
    @testset "inclusive backward scan with segments" begin
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=true)) == [2, 1, 0, 2, 2, 1]
    end

    @testset "segmented scan with float" begin
        mem_data = GPUConvexHull.ScanPrimitive.allocate_scan_memory_arrays(backend, Float64, Int64, 8, 6)
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(backend, mem_data, [1.0, 2.5, 0, 0, 3.0, 0.5], [1, 0, 0, 0, 0, 0], GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0, backward=false, inclusive=true)) == [1.0, 3.5, 3.5, 3.5, 6.5, 7]
        
    end
end