using KernelAbstractions
using LinearAlgebra
using TimerOutputs
using AcceleratedKernels

using .ScanPrimitive
using .MinMaxReduction

const EPSILON = 1e-9
const to = TimerOutput()
disable_timer!(to)

mutable struct QhullData
    vertices::Vector{Vector{Int}}
    neighbors::Vector{Vector{Int}}
    normals::Vector{Vector{Float64}}
    offsets::Vector{Float64}
    active::Vector{Bool}
end

mutable struct QhullResult
    hull_points::Matrix{Float64}
    hull_indices::Vector{Int}
    convex_hull_bounds::Vector{Vector{Float64}}
    #hull_faces::Vector{Vector{Int}}
end

struct QuickHullContext{BACKEND}
    
    backend::BACKEND
    workgroup_size::Int64
    default_segment::AbstractArray
end

function create_quickhull_context(backend, workgroup_size, n_points)
    default_segment = zeros(n_points)
    default_segment[1] = 1
    default_segment_gpu = KernelAbstractions.zeros(backend, Int64, n_points)
    copy!(default_segment_gpu, default_segment)

    return QuickHullContext(
        backend,
        workgroup_size,
        default_segment_gpu)
end

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

@kernel function permute_data_kernel(out::AbstractVecOrMat, data::AbstractVecOrMat, flags, perm, @uniform dim)
    global_id = @index(Global)
    if flags[global_id] == 1
        if dim == 1 # TODO: remove le if ici
            out[perm[global_id]+1] = data[global_id]
        else
            for d in 1:dim
                out[d, perm[global_id]+1] = data[d, global_id]
            end
        end
    end
end

@kernel function permute_sp_kernel(sp_out, sp, flags, p)
    id = @index(Global)
    if flags[id] == 1
        sp_out[p[id] + 1] = sp[id]
    end
end

@kernel function detect_heads_kernel(new_segments, sp_out, n)
    id = @index(Global)
    if id == 1
        new_segments[1] = 1
    else
        new_segments[id] = sp_out[id] != sp_out[id - 1] ? 1 : 0
    end
end

@kernel function permute_indices_kernel(out, data, flags, perm)
    global_id = @index(Global)
    if flags[global_id] == 1
        out[perm[global_id]+1] = data[global_id]
    end
end

@kernel function permute_kernel(out, data, perm)
    global_id = @index(Global)
    out[global_id] = data[perm[global_id]]
end
"""
compact(flags, segments, data, length, dim)

removes elements within segments.

Compact takes in an array of booleans `b` and
discards data from `s`, which are marked as false by the
boolean array

Returns a vector p, a permutation of data and a vector sp, a permutation of segments head.

```
"""
function compact(context::QuickHullContext, segment_mem_data, flags, segments, data, original_ids, in_length, dim)
    p = segmented_scan(segment_mem_data,flags, context.default_segment , ScanPrimitive.AddOp())

    p_last    = Vector{eltype(p)}(undef, 1)
    flag_last = Vector{eltype(flags)}(undef, 1)

    copyto!(p_last,    1, p,     in_length, 1)
    copyto!(flag_last, 1, flags, in_length, 1)
    n = Int(p_last[1]) + Int(flag_last[1]) # On ajoute le b[n] car on est en exclusive scan =))

    head_indices = KernelAbstractions.allocate(context.backend, Int64, in_length)
    segment_mask_kernel(context.backend, context.workgroup_size)(flags, p, head_indices, ndrange=in_length)
    KernelAbstractions.synchronize(context.backend)

    propagated_heads = segmented_scan(segment_mem_data, head_indices, segments, ScanPrimitive.MinOp(), backward=false, inclusive=true, identity=typemax(Int64))

    out_points = KernelAbstractions.allocate(context.backend, Float64, Int.((dim, n)))
    permute_data_kernel(context.backend, context.workgroup_size)(out_points, data, flags, p, dim, ndrange = in_length)
    KernelAbstractions.synchronize(context.backend)

    sp_out = KernelAbstractions.zeros(context.backend, Int64, n)
    permute_sp_kernel(context.backend, context.workgroup_size)(sp_out, propagated_heads, flags, p, ndrange=in_length)
    KernelAbstractions.synchronize(context.backend)

    new_segments = KernelAbstractions.zeros(context.backend, Int, n)
    detect_heads_kernel(context.backend, context.workgroup_size)(new_segments, sp_out, n, ndrange=n)
    KernelAbstractions.synchronize(context.backend)

    out_ids = KernelAbstractions.allocate(context.backend, Int64, n)
    permute_indices_kernel(context.backend, context.workgroup_size)(out_ids, original_ids, flags, p, ndrange=in_length)
    KernelAbstractions.synchronize(context.backend)

    return out_points, out_ids, new_segments, p, flags, n
end


function compute_hyperplane(simplex_points)
    dim = size(simplex_points, 1)
    n_pts = size(simplex_points, 2)
    M = hcat(Array(simplex_points)', ones(n_pts))

    U, S, V = svd(M, full=true)

    last_eigen_vector = V[:, end]
    a = last_eigen_vector[1:end-1]
    b = last_eigen_vector[end]

    return a, b
end

@inline function signed_distance(normal::AbstractVector, offset::Real, point::AbstractVector)
    return dot(normal, Array(point)) + offset
end

@kernel function mask_2d_kernel(output, data, n_flags)
    i, j = @index(Global, NTuple)
    output[i, j] = data[i] == j ? 1 : 0
end

@kernel function flag_permute_kernell(forwardScanArray, backwardScanArray, sh, flags, size, outPermutation)
    global_id = @index(Global)

    if global_id ≤ size
        offset = 0
        for idx=1 : flags[global_id]-1
            offset += backwardScanArray[global_id, idx]
        end
        # Position finale = Le début du sous segment dans le segment  + position du head + le nombre de flag à gauche (la position du flag dans le sous segment).
        outPermutation[global_id] = offset + sh[global_id] + (forwardScanArray[global_id, flags[global_id]] - 1)
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

# TODO à voir si c'est vraiment utile ou pas.
@kernel function mark_segment_id(output, segments)
    global_id = @index(Global)
    output[global_id] = (segments[global_id] == 1) ? global_id : 0
end

@kernel function rebuild_segments_kernel!(new_segments, flags, permutation)
    global_id = @index(Global)
    n = length(flags)
    
    if global_id == 1
        new_segments[1] = 1
    elseif global_id <= n
        curr_idx = permutation[global_id]
        prev_idx = permutation[global_id-1]
        
        if flags[curr_idx] != flags[prev_idx]
            new_segments[global_id] = 1
        else
            new_segments[global_id] = 0
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
function flag_permute(context::QuickHullContext, flags, segments)
    out_permutation = AcceleratedKernels.sortperm(flags, block_size=context.workgroup_size)
    # Recreate segments heads
    rebuild_segments_kernel!(context.backend, context.workgroup_size)(segments, flags, out_permutation, ndrange=length(segments))
    return out_permutation
end


function compute_simplex(context::QuickHullContext, result::QhullResult, mesh::QhullData, points, dim::Int64)
    
    points_idx = Set{UInt32}()

    for d in 1:dim
        res = min_max_reduce(points[d, :], 16, context.backend)

        push!(points_idx, res.imin)
        push!(points_idx, res.imax)

        if length(points_idx) >= dim + 1 # Early break car balek on a déjà nos dim+1 points
            break
        end
    end
    KernelAbstractions.synchronize(context.backend)

    # Au cas où on a plusieurs points qui ont la même coordonnée min ou max (cas très rare donc on se le permet de faire sur cpu)
    # On utilise la méthode des points les plus éloignés. Enfaite on devrait arriver ici que si il nous manque 1 points
    # Si il manque plus d'un point c'est que les points d'entrés sont définies comme dans une dimension d alors qu'ils sont
    # dans la dimension d-1.
    pts_all = Array(points)

    while length(points_idx) < dim + 1
        idx = collect(points_idx)
        avail = setdiff(1:size(pts_all, 2), idx)

        p0 = pts_all[:, idx[1]]
        V  = pts_all[:, avail] .- p0 # On centre les points sur p0

        if length(idx) == 1
            sq_distances = sum(abs2, V, dims=1)[:]
        else
            B = pts_all[:, idx[2:end]] .- p0
            residuals = V - B * (B \ V) # Parmis nos points centrés en V on projette nos points sur le sous-espace générés par idxs
            sq_distances = sum(abs2, residuals, dims=1)[:]
        end
        max_sq_dist, best_loc = findmax(sq_distances)
        push!(points_idx, UInt32(avail[best_loc]))

    end

    simplex_idx = collect(points_idx)[1:dim+1]
    simplex_matrix = points[:, simplex_idx]

    barycenter = sum(simplex_matrix, dims=2)[:] ./ (dim + 1)
    for i in 1:dim+1
        face_v = [simplex_idx[j] for j in 1:dim+1 if j != i]
        push!(mesh.vertices, face_v)
        push!(mesh.active, true)

        n, b = compute_hyperplane(points[:, face_v])
        if signed_distance(n, b, barycenter) > 0
            n = -n
            b = -b
        end
        push!(mesh.normals, n)
        push!(mesh.offsets, b)
        push!(mesh.neighbors, zeros(Int, dim))
    end

    for i in 1:(dim + 1)
        other_faces = [j for j in 1:(dim + 1) if j != i]
        for (k, target_face) in enumerate(other_faces)
            mesh.neighbors[i][k] = target_face
        end
    end

    for i in 1:size(simplex_matrix, 2)
        push!(result.convex_hull_bounds, copy(simplex_matrix[:, i]))
        push!(result.hull_indices, simplex_idx[i])
    end
end

@kernel function permute_points_kernel(out, data, perm, @uniform dim)
    global_id = @index(Global)
    for d in 1:dim
        out[d, global_id] = data[d, perm[global_id]]
    end
end

function propagate_segment_idx(context::QuickHullContext, segment_mem_data, segments)
    n = length(segments)
    
    seg_id = segmented_scan(segment_mem_data, segments, context.default_segment, ScanPrimitive.AddOp(), backward=false, inclusive=true)
    KernelAbstractions.synchronize(context.backend)
    return seg_id, Array(seg_id)[n]
end

@kernel function distance_to_face_kernel(distances, points, seg_id, seg_to_face_map, normals, offsets)
    global_id = @index(Global)
    n = length(seg_id)
    if global_id <= n
        segment_id = seg_id[global_id]
        face_id = seg_to_face_map[segment_id]
            acc = 0.0
        for d in 1:dim
            acc += points[d, global_id] * normals[d, face_id]
        end
        distances[global_id] = acc + offsets[face_id]
    end
end

@kernel function mark_farthest_candidate_kernel(cand_idx, distances, seg_max)
    global_index = @index(Global)
    if global_index <= length(cand_idx)
        m = seg_max[global_index]
        if m > 0 && abs(distances[global_index] - m) <= EPSILON
            cand_idx[global_index] = global_index
        else
            cand_idx[global_index] = 0
        end
    end
end


@kernel function compact_data_to_segment_kernel(out_idx, out_dist, segments, seg_id, seg_far_idx, seg_distance)
    i = @index(Global)
    if i <= length(seg_far_idx)
        if segments[i] == 1
            d = seg_distance[i]
            out_dist[seg_id[i]] = d
            out_idx[seg_id[i]] = (d > 0) ? seg_far_idx[i] : 0
        end
    end
end

function get_visible_faces(mesh, point_idx, face_id, points)
    visible_faces = Set{Int}()
    # On trouve toutes les faces visibles depuis le point à insérer (en gros toutes les faces pour lesquelles le point est du côté positif)
    # Pour ça on fait un BFS à partir de la face dont le point faisait partis de base, puis on regarde les voisins etc...
    queue = [face_id]
    push!(visible_faces, face_id)
    #println("Starting BFS with face id ", face_id)
    while !isempty(queue)
        current_face_id = popfirst!(queue)

        for neighbour in mesh.neighbors[current_face_id]
            if neighbour == 0 || neighbour in visible_faces || mesh.active[neighbour] == false
                continue
            end

            normal = mesh.normals[neighbour]
            offset = mesh.offsets[neighbour]

            if signed_distance(normal, offset, points[:, point_idx]) > EPSILON
                push!(queue, neighbour)
                push!(visible_faces, neighbour)
            end
        end
    end

    return visible_faces
end

# TODO: J'ai vu qu'il y avais moyens de faire un BFS sur GPU en utilisant une représentation RCS du graphe. À voir si j'ai le temps.
function insert_point_and_update_mesh(mesh::QhullData, point_idx::Int64, face_id::Int64, points, dim::Int64)
    # Maintenant qu'on a nos faces visible on trouve l'horizon entre les faces visibles et les faces non visibles.
    #TODO: paralléliser sur GPU ?
    visible_faces = get_visible_faces(mesh, point_idx, face_id, points)

    horizon_ridges = Tuple{Vector{Int64}, Int64, Int64}[]
    for f in visible_faces
        for (k, neighbor) in enumerate(mesh.neighbors[f])
            if neighbor > 0 && mesh.active[neighbor] && !(neighbor in visible_faces)
                ridge_verts = [mesh.vertices[f][j] for j in 1:dim if j != k]
                push!(horizon_ridges, (ridge_verts, neighbor, f))
            end
        end
        mesh.active[f] = false
    end

    # Crée les nouvelles faces avec l'hyperplan généré
    new_face_indices = Int[]
    for (ridge_verts, invisible_neighbor, old_visible_face) in horizon_ridges
        new_verts = copy(ridge_verts)
        push!(new_verts, point_idx)

        push!(mesh.vertices, new_verts)
        new_f_idx = length(mesh.vertices)
        push!(new_face_indices, new_f_idx)
        push!(mesh.active, true)

        pts_matrix = points[:, new_verts]
        normal, offset = compute_hyperplane(pts_matrix)

        # On prends le point qui n'est plus sur le ridge.
        inside_v_idx = setdiff(mesh.vertices[old_visible_face], ridge_verts)[1]
        inside_pt = points[:, inside_v_idx]

        if signed_distance(normal, offset, inside_pt) > EPSILON
            normal = -normal
            offset = -offset
        end

        push!(mesh.normals, normal)
        push!(mesh.offsets, offset)
        push!(mesh.neighbors, zeros(Int, dim))
        for i in 1:dim
            if mesh.neighbors[invisible_neighbor][i] == old_visible_face
                mesh.neighbors[invisible_neighbor][i] = new_f_idx
            end
        end
    end

    for i in 1:length(new_face_indices)
        f1 = new_face_indices[i]
        for j in (i+1):length(new_face_indices)
            f2 = new_face_indices[j]
            shared = intersect(mesh.vertices[f1], mesh.vertices[f2])
            if length(shared) == dim - 1
                k1 = findfirst(x -> !(x in shared), mesh.vertices[f1])
                mesh.neighbors[f1][k1] = f2
                k2 = findfirst(x -> !(x in shared), mesh.vertices[f2])
                mesh.neighbors[f2][k2] = f1
            end
        end
        ridge_neighbor_idx = horizon_ridges[i][2]
        k_base = findfirst(x -> x == point_idx, mesh.vertices[f1])
        mesh.neighbors[f1][k_base] = ridge_neighbor_idx
    end
end

@kernel function assign_face_to_point_kernel(face_flags, compact_flags, points, normals, offsets, @uniform dim)
    global_id = @index(Global)
    if global_id <= size(points, 2)
        max_d = EPSILON
        best_face = 0
        for f in 1:size(normals, 2)
            d = 0.0
            for d_i in 1:dim
                d += normals[d_i, f] * points[d_i, global_id]
            end
            d += offsets[f]
            if d > max_d
                max_d = d
                best_face = f
            end
        end
        face_flags[global_id] = best_face
        compact_flags[global_id] = (best_face == 0) ? 0 : 1
    end
end

function prepare_output(context, result)
    result.hull_points = sort_points_counter_clockwise(hcat(result.convex_hull_bounds...))
    
    return result
end


function _quick_hull_implem(context::QuickHullContext, segment_mem_data_float::ScanPrimitive.ScanPrimitiveContext, 
        segment_mem_data_int::ScanPrimitive.ScanPrimitiveContext, points, n_points::Int64 = size(points, 2), dim::Int64 = size(points, 1))

        @timeit to "Quickhull implem" begin

        result = QhullResult(zeros(Float64, dim, 0), Int64[], Vector{Float64}[])
        n_iter = 0

        # All Allocations
        original_ids_cpu = collect(1:n_points)
        original_ids = KernelAbstractions.allocate(context.backend, Int64, n_points)
        copyto!(original_ids, original_ids_cpu)

        mesh = mesh = QhullData(
            Vector{Int}[],
            Vector{Int}[],
            Vector{Float64}[],
            Float64[],
            Bool[]
        )
        @timeit to "Compute simplex" begin
            # Create a simplex by finding min & max points allong all dimensions
            compute_simplex(context, result, mesh, points, dim)
        end
        n_active = length(mesh.active)
        active_normals = hcat(mesh.normals...)
        active_offsets = vcat(mesh.offsets...)

        face_normals_gpu = KernelAbstractions.allocate(context.backend, Float64, (dim, n_active))
        face_offsets_gpu = KernelAbstractions.allocate(context.backend, Float64, n_active)
        copyto!(face_normals_gpu, active_normals)
        copyto!(face_offsets_gpu, active_offsets)

        
        face_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, n_points)
        compact_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, n_points)
        
        @timeit to "Assign face to point" begin
            # Remove simplex points & points inside the simplex from input data
            assign_face_to_point_kernel(context.backend, context.workgroup_size)(face_flags_gpu, compact_flags_gpu, points, face_normals_gpu, face_offsets_gpu, dim, ndrange=n_points)
            KernelAbstractions.synchronize(context.backend)
        end
        restant, original_ids, rest_segment, cp, cflags, cn = compact(context, segment_mem_data_int,compact_flags_gpu, context.default_segment, points, original_ids, n_points, dim)

        @timeit to "Remove points inside simplex" begin
            # Remove null flags
            compacted_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, size(restant, 2))
            permute_indices_kernel(context.backend, context.workgroup_size)(compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n_points)
            KernelAbstractions.synchronize(context.backend)

            compacted_flags_cpu = Array(compacted_flags_gpu)
            current_unique_flags = sort(unique(compacted_flags_cpu))
            n_total_flag = length(current_unique_flags)
            mapping = Dict(old => new for (new, old) in enumerate(current_unique_flags))
            mapped_flags_cpu = [mapping[f] for f in compacted_flags_cpu]


            mapped_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, size(restant, 2))
            copyto!(mapped_flags_gpu, mapped_flags_cpu)

            out_perm = flag_permute(context,mapped_flags_gpu, rest_segment)

            restant_new = KernelAbstractions.zeros(context.backend, Float64, (dim, size(restant, 2)))
            ids_new = KernelAbstractions.zeros(context.backend, Int64, size(restant, 2))

            permute_points_kernel(context.backend, context.workgroup_size)(restant_new, restant, out_perm, dim, ndrange=size(restant, 2))
            KernelAbstractions.synchronize(context.backend)

            permute_kernel(context.backend, context.workgroup_size)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))
            KernelAbstractions.synchronize(context.backend)

            mapped_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, length(compacted_flags_gpu))
            copyto!(mapped_flags_gpu, [mapping[f] for f in compacted_flags_cpu])

            restant = restant_new
            original_ids = ids_new
        end
        n = size(restant, 2)

        while size(restant, 2) > 0
            @timeit to "Quickhull looop" begin
                #println("######### NEW ITTERATION $n_iter ############")
                #TODO: ce code sert un peu à rien ....
                 @timeit to "Get farthest points" begin
                    seg_id, n_segs = propagate_segment_idx(context, segment_mem_data_int,rest_segment)
                    n = size(restant, 2)
                    
                    if n_segs == 0
                        break
                    end
                    
                    @timeit to "distance_to_face_kernel" begin
                    distances = KernelAbstractions.allocate(context.backend, Float64, n)
                    seg_to_face_map_gpu = KernelAbstractions.allocate(context.backend, Int64, length(current_unique_flags))
                    copyto!(seg_to_face_map_gpu, current_unique_flags)
                    
                    distance_to_face_kernel(context.backend, context.workgroup_size)(distances, restant, seg_id, seg_to_face_map_gpu, face_normals_gpu, face_offsets_gpu, ndrange=n)
                    KernelAbstractions.synchronize(context.backend)
                    end

                    @timeit to "Propagate farthest point" begin
                    # On propage le point le plus loins et on le met au début du segment.
                        prefix_max = segmented_scan(segment_mem_data_float, distances, rest_segment, ScanPrimitive.MaxOp(),  backward=false, inclusive=true, identity=typemin(Float64))
                        seg_max    = segmented_scan(segment_mem_data_float, prefix_max, rest_segment, ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Float64))

                    end

                    @timeit to "Mark farthest candidate" begin
                    cand_idx = KernelAbstractions.allocate(context.backend, Int64, n)
                    mark_farthest_candidate_kernel(context.backend, context.workgroup_size)(cand_idx, distances, seg_max, ndrange=n)
                    KernelAbstractions.synchronize(context.backend)
 
                    #ts1 = now()
                    prefix_far = segmented_scan(segment_mem_data_int, cand_idx, rest_segment, ScanPrimitive.MaxOp(), backward=false, inclusive=true, identity=typemin(Int64))
                    far_idx_p  = segmented_scan(segment_mem_data_int, prefix_far, rest_segment, ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Int64))

                    far_idx  = KernelAbstractions.allocate(context.backend, Int64,  n_segs)
                    far_dist = KernelAbstractions.allocate(context.backend, Float64, n_segs)
                    compact_data_to_segment_kernel(context.backend, context.workgroup_size)(far_idx, far_dist, rest_segment, seg_id, far_idx_p, seg_max, ndrange=n)
                    

                    far_idx = Array(far_idx)
                    far_dist = Array(far_dist)
                    original_ids_cpu = Array(original_ids)
                    active_face_indices = findall(mesh.active)
                    end
                end
                 @timeit to "Add possible point to convexhull" begin
                
                candidates = NamedTuple{(:dist, :local_idx, :seg_idx), Tuple{Float64, Int64, Int64}}[]
                for i in 1:n_segs
                    if far_idx[i] != 0 && far_dist[i] > EPSILON
                        push!(candidates, (dist = far_dist[i], local_idx = far_idx[i], seg_idx = i))
                    end
                end

                # TODO: tri sur GPU (radixsort)
                sort!(candidates, by = x -> x.dist, rev = true)

                # Getting indenpendant points to add to void add conflict
                to_remove_faces = Set{Int}()
                points_to_insert = Tuple{Int64, Int64}[]

                for cand in candidates
                    global_point_idx = original_ids_cpu[cand.local_idx]
                    local_face_idx = current_unique_flags[cand.seg_idx]
                    global_face_id = active_face_indices[local_face_idx]

                    visible_faces = get_visible_faces(mesh, global_point_idx, global_face_id, points)

                    conflict = false
                    for f in visible_faces
                        if f in to_remove_faces
                            conflict = true
                            break
                        end
                    end

                    if !conflict
                        push!(points_to_insert, (global_point_idx, global_face_id))
                        union!(to_remove_faces, visible_faces)
                    end
                end

                for (point_idx, face_id) in points_to_insert
                    push!(result.convex_hull_bounds, points[:, point_idx])
                    push!(result.hull_indices, point_idx)
                    insert_point_and_update_mesh(mesh, point_idx, face_id, points, dim)
                end

                n_active = sum(mesh.active)
                active_normals = zeros(Float64, dim, n_active)
                active_offsets = zeros(Float64, n_active)
                idx = 1
                for i in 1:length(mesh.active)
                    if mesh.active[i]
                        active_normals[:, idx] = mesh.normals[i]
                        active_offsets[idx] = mesh.offsets[i]
                        idx += 1
                    end
                end

                face_normals_gpu = KernelAbstractions.allocate(context.backend, Float64, (dim, n_active))
                face_offsets_gpu = KernelAbstractions.allocate(context.backend, Float64, n_active)
                copyto!(face_normals_gpu, active_normals)
                copyto!(face_offsets_gpu, active_offsets)

                face_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, n)
                compact_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, n)
                assign_face_to_point_kernel(context.backend, context.workgroup_size)(face_flags_gpu, compact_flags_gpu, restant, face_normals_gpu, face_offsets_gpu, dim, ndrange=n)
                KernelAbstractions.synchronize(context.backend)
                end

                @timeit to "Remove points inside convexhull" begin

                @timeit to "Compact" begin
                    global_seg_reset = zeros(Int64, size(restant, 2))
                    if length(global_seg_reset) > 0
                        global_seg_reset[1] = 1
                    end
                    copyto!(rest_segment, global_seg_reset)

                    # Now Remove points inside
                    restant, original_ids, rest_segment, cp, cflags, cn = compact(context, segment_mem_data_int, compact_flags_gpu, rest_segment, restant, original_ids, n, dim)

                    if length(restant) == 0
                        break
                    end
                end

                @timeit to "create data for falg permute & permute indices" begin
                    @timeit "Permute indices kernell" begin
                    compacted_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, size(restant, 2))
                    permute_indices_kernel(context.backend, context.workgroup_size)(compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n)
                    KernelAbstractions.synchronize(context.backend)
                    end
                compacted_flags_cpu = Array(compacted_flags_gpu)
                current_unique_flags = sort(unique(compacted_flags_cpu))
                n_total_flag = length(current_unique_flags)
                mapping = Dict(old => new for (new, old) in enumerate(current_unique_flags))
                mapped_flags_cpu = [mapping[f] for f in compacted_flags_cpu]

                mapped_flags_gpu = KernelAbstractions.allocate(context.backend, Int64, size(restant, 2))
                copyto!(mapped_flags_gpu, mapped_flags_cpu)
                end
                @timeit to "flag permute" begin
                out_perm = flag_permute(context, mapped_flags_gpu, rest_segment)
                end

                @timeit to "permute points" begin
                new_restant = KernelAbstractions.zeros(context.backend, Float64, (dim, size(restant, 2)))
                permute_points_kernel(context.backend, context.workgroup_size)(new_restant, restant, out_perm, dim, ndrange=size(restant, 2))
                end

                @timeit to "permute kernel" begin   
                ids_new = KernelAbstractions.allocate(context.backend, Int64, size(restant, 2))
                permute_kernel(context.backend, context.workgroup_size)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))
                end
                restant = new_restant
                original_ids = ids_new
                n_iter += 1
                end
            end
        end

        println("N_iter: ", n_iter)
        return prepare_output(context, result)
    end

    @show to
end

function quick_hull(backend, points, n_points::Int64 = size(points, 2), workgroup_size::Int64 = 256)
    context = create_quickhull_context(backend, workgroup_size, n_points)
    segment_mem_data_float = create_scan_primitive_context(backend, Float64, Int64, workgroup_size, n_points)
    segment_mem_data_int = create_scan_primitive_context(backend, Int64, Int64, workgroup_size, n_points)

    return _quick_hull_implem(context, segment_mem_data_float, segment_mem_data_int, points)
end

function read_input_file(path)
    f = open(path)
    pt_cnt, dim = map((x) -> parse(Int, x), split(readline(f), " "))
    points = zeros(Float64, dim, pt_cnt)
    for line in range(1, pt_cnt)
        points[:, line] =  map((x) -> parse(Float64, x), split(readline(f), " "))
    end
    close(f)

    return points
end

function sort_points_counter_clockwise(points)
    points = Array(points)
    barycenter = sum(points, dims=2)[:] ./ size(points, 2)
    angles = atan.(points[2, :] .- barycenter[2], points[1, :] .- barycenter[1])
    sorted_indices = sortperm(angles)
    return points[:, sorted_indices]
end
