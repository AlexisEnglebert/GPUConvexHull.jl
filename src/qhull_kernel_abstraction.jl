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
    out_points = KernelAbstractions.allocate(backend, Float64, Int.((dim, n)))    
    permute_data_kernel(backend, 16)(out_points, data, flags, p, dim, ndrange = in_length)
    println("Out segments is : ", propagated_heads)
    println("Out data is : ", out_points)
    return out_points, propagated_heads
end


function compute_hyperplane(simplex_points)
    dim = size(simplex_points, 1)
    M = hcat(simplex_points', ones(dim))

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
    #println("Hyp points : ", hyp_points)
    normal, offset = compute_hyperplane(hyp_points)
    #println("Normal : ", normal, " Offset: ", offset)
    #println("points:  ", points, "size : ", size(points))
    out_flags = fill(0, size(points)[2])
    #println("distance input points: ", points)
    # Parallelisable ?
    for (index, p) in enumerate(eachcol(points))
        dist = signed_distance(normal, offset, p) / norm(normal)
        #println(normal, "    ", p, dist)

        if dist > 0
            out_flags[index] = 1
        elseif dist < 0
            out_flags[index] = 2
        else
            out_flags[index] = 0
        end
        println(index, p, out_flags[index])
    end
    println("outflag: ", out_flags)
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

@kernel function add_segment_kernel(backscanArray, segments, sh, n_flags)
    id = @index(Global)

    if id <= size(backscanArray, 1)
        loc = 0
        for i = 1:(n_flags-1)
            loc += backscanArray[id, i]
            if loc > 0 && (sh[id] + loc) == id
                segments[id] = 1
            end
        end
    end
end


#TODO je pense vraiment qu'on peut optimiser ça et trouver une autre routine ça pourrait faire gagner
# pas mal de perf je pense vu que c'est une des composante principale
"""
flag_permute(flags, segments, data_size, n_flags)

give a permutation vector to permute flags in given segments in order to sort them.

# Examples
```jldoctest
julia> flag_permute(.....)
```
"""
function flag_permute(flags, segments, data_size, n_flags, segments_start)

    maskedArray = Matrix{Int64}(undef, data_size, n_flags)
    scanArray = Matrix{Int64}(undef, data_size, n_flags)
    backscanArray = Matrix{Int64}(undef, data_size, n_flags)
    
    println("-- Flag permute with input --")
    println(flags)
    println(segments)


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

    #println(scanArray, " length is : ", length(scanArray), " size to cmp ", size(scanArray))
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

@kernel function permute_points_kernel(out, data, perm, @uniform dim)
    global_id = @index(Global)
    for d in 1:dim
        out[d, perm[global_id]] = data[d, global_id]
    end
end

function permute_points(backend, points, perm, dim)
    out = KernelAbstractions.allocate(backend, Float64, (dim, size(points, 2)))
    permute_points_kernel(backend, 16)(out, points, perm, dim, ndrange=size(points, 2))
    return out
end


function propagate_segment_idx(segments)
    n = length(segments)
    global_head = KernelAbstractions.zeros(backend, Int, n)
    global_head[1] = 1
    seg_id = segmented_scan(backend, segments, global_head, (a, b) -> a + b, false, true)
    synchronize(backend)
    return seg_id, seg_id[n]
end

@kernel function distance_to_face_kernel(distances, points, seg_id, normals, offsets, @uniform dim)
    gid = @index(Global)
    n = length(seg_id)
    if gid <= n
        sid = seg_id[gid]
        acc = 0.0
        for d in 1:dim
            acc += points[d, gid] * normals[d, sid]
        end
        distances[gid] = acc + offsets[sid]
    end
end

@inline function oriented_hyperplane_with_inside(face, inside_point)
    n, b = compute_hyperplane(face)
    if signed_distance(n, b, inside_point) > 0
        n = -n
        b = -b
    end

    return n, b
end

function rebuild_face_hyperplanes(face_points, inside_point, dim)
    n_segments = length(face_points)
    normals = KernelAbstractions.allocate(backend, Float64, (dim, n_segments))
    offsets = KernelAbstractions.allocate(backend, Float64, n_segments)
    for id in 1:n_segments
        normal, offset = oriented_hyperplane_with_inside(face_points[id], inside_point)
        normals[:, id] = normal
        offsets[id] = offset
    end
    synchronize(backend)
    return normals, offsets
end

@kernel function mark_farthest_candidate_kernel(cand_idx, distances, seg_max, @uniform eps)
    i = @index(Global)
    if i <= length(cand_idx)
        m = seg_max[i]
        if m > 0 && abs(distances[i] - m) <= eps
            cand_idx[i] = i
        else
            cand_idx[i] = 0
        end
    end
end

function quick_hull(points, n_points, dim)
    convex_hull_bounds = []
    segments = fill(0, n_points)
    segments[1] = 1
    segments_start = [1]

    # Create a simplex by finding min & max points allong all dimensions
    simplex_idx = compute_simplex(points, dim, backend)
    simplex_idx = simplex_idx[1:dim] #TODO pour l'instant c'est la hess mais on prends que les d premiers points pour le simplex
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
    println("Restant: ", restant)
    #TODO : compact avec les segments
    temp_segments = fill(0, size(restant)[2])
    temp_segments[1] = 1

    outPermutation = flag_permute(dist_flags, temp_segments, length(temp_segments), 2, segments_start)    
    restant = permute_points(backend, restant, outPermutation, dim)

    println("Restant après permutation: ", restant)
    face_points = [simplex_matrix for _ in 1:3]
    inside_point = vec(sum(points; dims=2) ./ n_points)
    face_normals, face_offsets = rebuild_face_hyperplanes(face_points, inside_point, dim)

    while size(restant, 2) > 0
        # For each segments we compute the hyperplane corresponding to them.
        seg_id, n_segs = propagate_segment_idx(temp_segments)
        n = size(restant, 2)

        if n_segs == 0
            break
        end
        
        distances = KernelAbstractions.allocate(backend, Float64, n)
        distance_to_face_kernel(backend, 16)(distances, restant, seg_id, face_normals, face_offsets, dim, ndrange=n)
        synchronize(backend)
        
        println(distances)
        println(temp_segments)
        

        # On propage le point le plus loins à travers le segement (forward + backward pass)
        prefix_max = segmented_scan(backend, distances, temp_segments, max,  false, true, typemin(Float64))
        seg_max    = segmented_scan(backend, prefix_max, temp_segments, max, true,  true, typemin(Float64))
        
        println(prefix_max)
        println(seg_max)
        println(temp_segments)


        cand_idx = KernelAbstractions.allocate(backend, Int64, n)
        mark_farthest_candidate_kernel(backend, 16)(cand_idx, distances, seg_max, 1e-12, ndrange=n)
        synchronize(backend)

        prefix_far = segmented_scan(backend, cand_idx, temp_segments, max, false, true)
        far_idx_p  = segmented_scan(backend, prefix_far, temp_segments, max, true,  true)

    end
end


points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0 ;
           2.0  0.0 -2.0  0.0  3.0  1.0 ]

quick_hull(points, 6, 2)

end
