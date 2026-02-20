module qhull_kernel_abstraction
using KernelAbstractions
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "ScanPrimitive.jl"))
include(joinpath(@__DIR__, "..", "src", "MinMaxReduction.jl"))
using .ScanPrimitive
using .MinMaxReduction


###### Les paramètres ici sont pour tester, ils devront être setup lorsque la librairie.
backend = CPU()

"""
segment_mask_kernel(b, p, out, identity_val)

Nom un peu fancy du papier, ça permet juste de fill l'array out avec les index de p

# Examples
```jldoctest
julia> a ....
```
"""
@kernel function segment_mask_kernel(b, p, out)
    i = @index(Global)
    out[i] = typemax(Int64)
    if b[i] == 1
        out[i] = p[i] + 1
    end
end

@kernel function permute_data_kernel(out, data, flags, perm, @uniform dim)
    global_id = @index(Global)
    if flags[global_id] == 1
        for d in 1:dim
            out[d, perm[global_id]+1] = data[d, global_id] 
        end
    end
end

"""
compact(b, s, n)


removes elements within segments.

Compact takes in an array of booleans `b` and
discards data from `s`, which are marked as false by the
boolean array

```
"""
function compact(flags, segments, data, in_length, dim)
    global_head = KernelAbstractions.zeros(backend, Int, in_length)
    global_head[1] = 1
    p = segmented_scan(backend, flags, global_head, (a, b) -> a + b) 
    n = p[in_length]+flags[in_length] # On ajoute le b[n] car on est en exclusive scan =))
    println("IIIIII : " , p)
    println("Flags : ", flags)
    
    head_indices = KernelAbstractions.allocate(backend, Int64, in_length)
    segment_mask_kernel(backend, 16)(flags, p, head_indices, ndrange=in_length)
    println("Segmented mask kernell output: ", head_indices)
    propagated_heads = segmented_scan(backend, head_indices, segments, min, true, true)
    
    println("???")
    out_points = KernelAbstractions.allocate(backend, Float64, (dim, n))    
    permute_data_kernel(backend, 16)(out_points, data, flags, p, dim, ndrange = in_length)
    println("Out segments is : ", propagated_heads)
    println("Out data is : ", out_points)
    return out_points, propagated_heads
end


function compute_hyperplane(simplex_points)
    dim = size(simplex_points, 1)
    M = hcat(simplex_points, ones(dim))

    U, S, V = svd(M, full=true)

    last_eigen_vector = V[:, end]
    a = last_eigen_vector[1:end-1]
    b = last_eigen_vector[end]

    return a, b
end

@inline function signed_distance(n::AbstractVector, b::Real, p::AbstractVector)
    return dot(n, p) + b
end


function distance_from_hyperplane(points, hyp_points)
    normal, offset = compute_hyperplane(hyp_points)
    println("Normal : ", normal, " Offset: ", offset)
    out_flags = fill(0, length(points))
    println("distance input points: ", points)
    # Parallelisable ?
    for (index, p) in enumerate(eachcol(points))
        println(normal, "    ", p)
        dist = signed_distance(normal, offset, p) / norm(normal)
        if dist > 0
            out_flags[index] = 1
        elseif dist < 0
            out_flags[index] = 2
        else
            out_flags[index] = 0
        end
        println(index, p, out_flags[index])
    end
    println(out_flags)
    return out_flags
end


# O(n) :( -> On pourrait le faire en O(log n) sur gpu je pense bien à voir si ça vaut le coups dans le movement des donnés ect...
# Vu qu'on réduit à chaque fois la taille ça pourrait être inréressant d'avoir un système hybride GPU > threshold au sinon CPU
function mask(data, pattern) 
    output = fill(0, length(data))
    for (index, val) in enumerate(data)
        if val == pattern output[index] = 1 end
    end
    return output
end

@kernel function flag_permute_kernell(forwardScanArray, backwardScanArray, sh, flags, size, outPermutation)
    # TODO
    # On fait la somme de tous les backward scan des flags avant le notre pour trouver l'offset
    # Je pense que c'est assez innéficace on sait très bien le faire sur GPU non ? 
    global_id = @index(Global)
    
    if global_id ≤ size
        offset = 0
        for idx=1 : flags[global_id]-1
            offset += backwardScanArray[global_id, idx]
        end
        # TODO: Check la fin car suspect
        outPermutation[global_id] = offset + sh[global_id] + forwardScanArray[global_id, flags[global_id]]
    end
end

@kernel function add_segment_kernel(backawardScanArray, segments, sh, n_flags)
    global_id = @index(Global)
    if global_id < length(backawardScanArray)
        for i=1 : n_flags
            # TODO Peut être remplacé avec du branchless pour éviter le Masking du gpu à cause du if
            if backawardScanArray[global_id, i] + sh[global_id] == global_id
                segments[global_id] = 1
            end
        end
    end
end

#TODO je pense vraiment qu'on peut optimiser ça et trouver une autre routine ça pourrait faire gagner
# pas mal de perf je pense vu que c'est une des composante principale
function flag_permute(flags, segments, data_size, n_flags)
    maskedArray = Matrix{Int64}(undef, data_size, n_flags)
    scanArray = Matrix{Int64}(undef, data_size, n_flags)
    backscanArray = Matrix{Int64}(undef, data_size, n_flags)
    
    st = similar(segments)
    #TODO en parallèle normalement mais bon vu que je pige RIEN à ce qu'on me veut alors je fais en séquentiel pour l'instant
    for id=1:length(segments)
        if segments[id] == 1
            st[id] = id
        else
            st[id] = 0
        end
    end

    sh = segmented_scan(backend, st, segments,(a, b) -> a+b, false, true)

    for i=1:n_flags
        @views maskedArray[:, i] .= mask(flags, i)
        @views scanArray[:, i] .=  segmented_scan(backend, maskedArray[:, i], segments, (a, b) -> a+b)
        @views backscanArray[:, i] .= segmented_scan(backend, scanArray[:, i], segments, max, true, true) 
    end
    # Now the flag permute kernell :)
    outPermutation = similar(flags)
    flag_permute_kernell(backend, 16)(scanArray, backscanArray, sh, flags, data_size, outPermutation, ndrange=first(size(scanArray)))

    println(scanArray, " length is : ", length(scanArray), " size to cmp ", size(scanArray))
    # Add segment kernell
    add_segment_kernel(backend, 16)(backscanArray, segments, sh, n_flags, ndrange=first(size(scanArray)))

    println("Out permutation is :", outPermutation)
    println("New segments flags: ", segments)
    println("INPUT FLAG: ", flags)
    println("Masked Array: ", maskedArray)
    println("***")
    println("Scan array :", scanArray)
    println("***")
    println("Back array : ", backscanArray)

    return outPermutation
end


@kernel function max_distance_kernel(distances, data, @uniform normal, @uniform offsets, @uniform n_points, @uniform dimensions)
    thread_id = @index(Local)
    global_id = @index(Global)

    if global_id ≤ n
        #TODO ça on peut le unroll vu que la dimension est fixe normalement
        dist = 0.0
        for d=1:dimensions
            dist += points[d, global_id] * normal[d]
        end
        distances[global_id] = dist + offset
    ends

end

function compute_simplex(points, dim, backend)
    points_idx = Set{UInt32}()
    
    for d in 1:dim
        res = min_max_reduce(points[d, :], 16, backend)
        
        push!(points_idx, res.imin)
        push!(points_idx, res.imax)
    end
    synchronize(backend)
    return collect(points_idx)
end


function quick_hull(points, n_points, dim)
    convex_hull_bounds = []
    segments = fill(0, n_points)
    segments[1] = 1

    # Create a simplex by finding min & max points allong all dimensions
    simplex_idx = compute_simplex(points, dim, backend)
    simplex_matrix = points[:, simplex_idx]

    println("Simplex index : ", simplex_idx)
    println("Simplex_matrix: ", simplex_matrix)

    remove_flags = fill(1, n_points)
    for idx in simplex_idx
        remove_flags[idx] = 0
    end
    
    restant, rest_segment = compact(remove_flags, segments, points, n_points, dim)

    dist_flags = distance_from_hyperplane(restant, simplex_matrix)
    # Now that we have our simplex perfom the algorithm

    #TODO : compact avec les segments
    temp_segments = fill(0, length(restant))
    temp_segments[1] = 1

    flag_permute(dist_flags, temp_segments, length(temp_segments), 2)    
    
    # TODO dans la boucle while presque tout se fait côté GPU avec le CUDA array
    
end


points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0 ;
           2.0  0.0 -2.0  0.0  3.0  1.0 ]

quick_hull(points, 6, 2)

end
