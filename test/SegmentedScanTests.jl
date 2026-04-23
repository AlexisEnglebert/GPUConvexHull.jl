using Test
using KernelAbstractions
using GPUConvexHull

@testset "Segmented Scan" begin
    backend = CPU()
    @testset "exclusive forward scan NO segment" begin
        mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 8, 4)
        values = [1, 2, 3, 4]
        flags  = [1, 0, 0, 0]
        expected = [0, 1, 3, 6]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0)) == expected
    end
    mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 8, 6)
    @testset "exclusive forward scan with segments" begin
        values = [1, 1, 1, 1, 1, 1]
        flags  = [1, 0, 0, 1, 0, 1]
        expected = [0, 1, 2, 0, 1, 0]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0)) == expected
    end
    @testset "inclusive forward scan NO segments" begin
        values = [1, 1, 1, 1, 1, 1]
        flags  = [1, 0, 0, 0, 0, 0]
        expected = [1, 2, 3, 4, 5, 6]

        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=false, inclusive=true)) == expected
    end
    @testset "inclusive forward scan with segments" begin
        values = [1, 1, 1, 1, 1, 1]
        flags  = [1, 0, 0, 1, 0, 1]
        expected = [1, 2, 3, 1, 2, 1]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=false, inclusive=true)) == expected
    end
    @testset "exclusive backward scan NO segments" begin
        values = [1, 1, 0, 0, 1, 1]
        flags  = [1, 0, 0, 0, 0, 0]
        expected = [3, 2, 2, 2, 1, 0]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=false)) == expected
    end
    @testset "exclusive backward scan with segments" begin
        values = [1, 1, 0, 0, 1, 1]
        flags  = [1, 0, 0, 1, 0, 0]
        expected = [1, 0, 0, 2, 1, 0]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=false)) == expected
    end
    @testset "inclusive backward scan NO segments" begin
        values = [1, 1, 0, 0, 1, 1]
        flags  = [1, 0, 0, 0, 0, 0]
        expected = [4, 3, 2, 2, 2, 1]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=true)) == expected
    end
    @testset "inclusive backward scan with segments" begin
        values = [1, 1, 0, 0, 1, 1]
        flags  = [1, 0, 0, 1, 0, 0]
        expected = [2, 1, 0, 2, 2, 1]

        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=true, inclusive=true)) == expected
    end
    @testset "segmented scan with float" begin
        mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Float64, Int64, 8, 6)
        values = [1.0, 2.5, 0, 0, 3.0, 0.5]
        flags  = [1, 0, 0, 0, 0, 0]
        expected = [1.0, 3.5, 3.5, 3.5, 6.5, 7.0]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0.0, backward=false, inclusive=true)) == expected
    end

    @testset "Min & Max Operations" begin
        mem_data_max = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 8, 6)

        @testset "inclusive forward scan MaxOp" begin
            values = [1, 5, 2, 8, 3, 9]
            flags  = [1, 0, 0, 1, 0, 0]
            expected = [1, 5, 5, 8, 8, 9]
            @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data_max, values, flags, GPUConvexHull.ScanPrimitive.MaxOp(), identity=typemin(Int64), backward=false, inclusive=true)) == expected
        end

        @testset "inclusive backward scan MaxOp" begin
            values = [1, 5, 2, 8, 3, 9]
            flags  = [1, 0, 0, 1, 0, 0]
            expected = [5, 5, 2, 9, 9, 9]
            @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data_max, values, flags, GPUConvexHull.ScanPrimitive.MaxOp(), identity=typemin(Int64), backward=true, inclusive=true)) == expected
        end
    end

    @testset "Multiblock Segmented scan" begin
        n_large = 20
        mem_data_large = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 2, n_large)

        @testset "No segments" begin
            values = ones(Int64, n_large)
            flags = zeros(Int64, n_large)
            flags[1] = 1
            expected = collect(1:n_large)

            result = Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data_large, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), backward=false, inclusive=true))
            @test result == expected
        end

        @testset "Segment between two blocks" begin
            values = ones(Int64, n_large)
            flags = zeros(Int64, n_large)
            flags[1] = 1
            flags[10] = 1

            expected = vcat(collect(1:9), collect(1:11))
            result = Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data_large, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), backward=false, inclusive=true))
            @test result == expected
        end

        @testset "Segment between two blocks reversed" begin
            values = ones(Int64, n_large)
            flags = zeros(Int64, n_large)
            flags[1] = 1
            flags[10] = 1

            expected = vcat(reverse(collect(1:9)), reverse(collect(1:11)))
            
            result = Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data_large, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), backward=true, inclusive=true))
            @test result == expected
        end
    end

    @testset "Every flag is 1" begin
        mem_data = GPUConvexHull.ScanPrimitive.create_scan_primitive_context(backend, Int64, Int64, 8, 8)
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        flags  = [1, 1, 1, 1, 1, 1, 1, 1]

        expected_incl = [1, 2, 3, 4, 5, 6, 7, 8]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), backward=false, inclusive=true)) == expected_incl

        expected_excl = [0, 0, 0, 0, 0, 0, 0, 0]
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), backward=false, inclusive=false)) == expected_excl
    end

    @testset "Everything is zero" begin
        values = zeros(Int64, 8)
        flags  = [1, 0, 0, 0, 1, 0, 0, 0]
        expected = zeros(Int64, 8)
        @test Array(GPUConvexHull.ScanPrimitive.segmented_scan(mem_data, values, flags, GPUConvexHull.ScanPrimitive.AddOp(), identity=0, backward=false, inclusive=true)) == expected
    end
end