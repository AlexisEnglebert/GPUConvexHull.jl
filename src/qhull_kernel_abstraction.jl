module qhull_kernel_abstraction
using KernelAbstractions
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "ScanPrimitive.jl"))
using .ScanPrimitive

export minmax_reduce

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


struct MinMax{T, I} # Type and Index
    min::T
    imin::I
    max::T
    imax::I
end

# On dois définir c'est quoi le zéro de MinMax pour pouvoir gérer ça côté GPU
# HORRIBLE 1H pour trouver ça, même les template en c++ c'est plus simple mdr
Base.zero(::Type{MinMax{T, I}}) where {T, I} = MinMax{T, I}(typemax(T), zero(I), typemax(T), zero(I))

@kernel function min_max_reduce_block_kernel(values, output, @uniform n)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    smin  = @localmem(Float64, first(@groupsize))
    smax  = @localmem(Float64, first(@groupsize))
    simin = @localmem(UInt32, first(@groupsize))
    simax = @localmem(UInt32, first(@groupsize))

    if global_id ≤ n
        smin[thread_id]  = values[global_id]
        smax[thread_id]  = values[global_id]
        simin[thread_id] = global_id
        simax[thread_id] = global_id
    else
        smin[thread_id]  = typemax(eltype(values))
        smax[thread_id]  = typemin(eltype(values))
        simin[thread_id] = global_id
        simax[thread_id] = global_id
    end
    
    @synchronize()

    for steps in 1:ceil(Int64, log2((first(@groupsize()))))
        limit = ceil(Int64, first(@groupsize()) / (2^steps))
        if thread_id ≤ limit 
            if @inbounds smin[thread_id] > smin[thread_id + limit]
                 @inbounds smin[thread_id] = smin[thread_id + limit]
                 @inbounds simin[thread_id] = simin[thread_id + limit]
            end

            if @inbounds smax[thread_id] < smax[thread_id + limit]
                 @inbounds smax[thread_id] = smax[thread_id + limit]
                 @inbounds simax[thread_id] = simax[thread_id + limit]
            end
        end
        @synchronize()
    end

    if thread_id == 1
        output[block_id] = MinMax{Float64, UInt32}(smin[1], simin[1], smax[1], simax[1])
    end
end

@kernel function min_max_reduce_block_kernel_minmax(values, output, @uniform n)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)
    @uniform group_size = first(@groupsize)

    smin  = @localmem(Float64,  group_size)
    smax  = @localmem(Float64,  group_size)
    simin = @localmem(UInt32, group_size)
    simax = @localmem(UInt32, group_size)

    if global_id ≤ n
        smin[thread_id]  = values[global_id].min
        smax[thread_id]  = values[global_id].max
        simin[thread_id] = values[global_id].imin
        simax[thread_id] = values[global_id].imax
    else
        smin[thread_id]  = typemax(Float64)
        smax[thread_id]  = typemin(Float64)
        simin[thread_id] = 0
        simax[thread_id] = 0
    end
    
    @synchronize()

    for steps in 1:ceil(Int64, log2((first(@groupsize()))))
        limit = ceil(Int64, first(@groupsize()) / (2^steps))
        if thread_id ≤ limit 
            if @inbounds smin[thread_id] > smin[thread_id + limit]
                 @inbounds smin[thread_id] = smin[thread_id + limit]
                 @inbounds simin[thread_id] = simin[thread_id + limit]
            end

            if @inbounds smax[thread_id] < smax[thread_id + limit]
                 @inbounds smax[thread_id] = smax[thread_id + limit]
                 @inbounds simax[thread_id] = simax[thread_id + limit]
            end
        end
        @synchronize()
    end

    if thread_id == 1
        output[block_id] = MinMax{Float64, UInt32}(smin[1], simin[1], smax[1], simax[1])
    end
end

function min_max_reduce(values, workgroupsSize, backend)
    
    n_groups = cld(length(values), workgroupsSize)
    partial_minmax_block = KernelAbstractions.zeros(backend, MinMax{eltype(values), UInt32}, n_groups)

    min_max_reduce_block_kernel(backend, workgroupsSize)(values, partial_minmax_block, length(values), ndrange=n_groups*workgroupsSize)
    synchronize(backend)
    println("Wesh : ", partial_minmax_block )
    while(length(partial_minmax_block) > 1)
        n_remainder_groups = cld(length(partial_minmax_block), workgroupsSize)
        # TODO: SPEED UP technique de double buffering
        remainder_output = KernelAbstractions.zeros(backend, MinMax{eltype(values), UInt32}, n_remainder_groups)
        min_max_reduce_block_kernel_minmax(backend, workgroupsSize)(partial_minmax_block, remainder_output, length(partial_minmax_block),
        ndrange=n_remainder_groups*workgroupsSize)

        synchronize(backend)
        partial_minmax_block = remainder_output
    end
    print("FINAL IS ", partial_minmax_block[1])
    return partial_minmax_block[1]
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

function distance_from_hyperplane(points, hyp_points)
    normal, offset = compute_hyperplane(hyp_points)
    println("Normal : ", normal, " Offset: ", offset)
    out_flags = fill(0, length(points))
    println("distance input points: ", points)
    # Parallelisable ?
    for (index, p) in enumerate(eachcol(points))
        println(normal, "    ", p)
        dist = ((normal' * p) + offset) / norm(normal)
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
    end

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

    #TODO c'est pas juste
    flag_permute(dist_flags, temp_segments, length(temp_segments), 2)    
    
    # TODO dans la boucle while presque tout se fait côté GPU avec le CUDA array
    # TODO utiliser KernelAbstractions.zeros, pour créer des listes dans la mémoire du GPU et 
    # donc ne pas devoir transférer les donnés du CPU au GPU à chaque fois qu'on lance un kernell
end
 
#= values = [3, 1, 7,  2, 4, 1, 6, 5, 1, 1, 1, 1, 1, 1, 1, 1]
flags =  [1, 0, 0,  0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
oplus(a, b) = a + b
segmented_scan(values, flags, oplus)


flag = [0, 0, 0, 1, 0, 0, 0, 0]
data = [1, 2, 3, 4, 5, 6, 7, 8]

segmented_scan(data, flag, oplus)
segmented_scan(data, flag, oplus, true)
segmented_scan(data, flag, oplus, true, true)

compact_bool_data = [1, 0, 1, 0, 1, 1, 1, 0]
compact_data      = [0, 1, 2, 3, 4, 5, 6, 7]    
#compact_kernell = compact(backend, 4)
compact(compact_bool_data, compact_data, length(compact_bool_data))

=#


#min_max_data = 1:100
#min_max_reduce(min_max_data, 10, backend)


points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0 ;
           2.0  0.0 -2.0  0.0  3.0  1.0 ]
quick_hull(points, 6, 2)


#=
min_max_values = [10, 3 ,4, 1, 9, 8, 2, 2, 0]
min_max_values_ker = minmax_reduce(backend, 16) #Must be a power of two !!!
out = [(0, 0)] # TODO mieux comprendre là mdr
min_max_values_ker(min_max_values, out,  ndrange=length(min_max_values)) 
println("out is : ", out)=#
end
