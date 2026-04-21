#module qhull_kernel_abstraction
using KernelAbstractions
using LinearAlgebra
#using GLMakie, ColorSchemes

using .ScanPrimitive
using .MinMaxReduction

EPSILON = 1e-9


mutable struct QhullData
    vertices::Vector{Vector{Int}}   
    neighbors::Vector{Vector{Int}}
    normals::Vector{Vector{Float64}}
    offsets::Vector{Float64}
    active::Vector{Bool}
end

# TODO: BIG TRUC: HARMONISER LES WORKGROUP SIZE

###### Les paramètres ici sont pour tester, ils devront être setup lorsque la librairie.
#backend = CPU()
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
    @print("segment_mask_kernel: ", b[i], " & ", p[i], "\n")
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
    out[perm[global_id]] = data[global_id]
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
function compact(backend, flags, segments, data, original_ids, in_length, dim)
     #TODO: ça; ça peut être mieux, faut juste voir comment mieux faire pour les donnés GPU
    global_head_cpu = zeros(Int64, in_length)
    global_head_cpu[1] = 1
    
    global_head = KernelAbstractions.zeros(backend, Int64, in_length)
    copyto!(global_head, global_head_cpu)

    p = segmented_scan(backend, flags, global_head, ScanPrimitive.AddOp()) 

    p_last    = Vector{eltype(p)}(undef, 1)
    flag_last = Vector{eltype(flags)}(undef, 1)
    copyto!(p_last,    1, p,     in_length, 1)
    copyto!(flag_last, 1, flags, in_length, 1)
    n = Int(p_last[1]) + Int(flag_last[1]) # On ajoute le b[n] car on est en exclusive scan =))

    #println("IIIIII : " , p)
    #println("Flags : ", flags)
    #println("segments: ", segments)
    head_indices = KernelAbstractions.allocate(backend, Int64, in_length)
    segment_mask_kernel(backend, 16)(flags, p, head_indices, ndrange=in_length)
    #println("Segmented mask kernell output: ", head_indices)

    propagated_heads = segmented_scan(backend, head_indices, segments, ScanPrimitive.MinOp(), backward=false, inclusive=true, identity=typemax(Int64))
    
    println("???")
    out_points = KernelAbstractions.allocate(backend, Float64, Int.((dim, n)))    
    permute_data_kernel(backend, 16)(out_points, data, flags, p, dim, ndrange = in_length)
    
    sp_out = KernelAbstractions.zeros(backend, Int64, n)
    permute_sp_kernel(backend, 16)(sp_out, propagated_heads, flags, p, ndrange=in_length)

    new_segments = KernelAbstractions.zeros(backend, Int, n)
    detect_heads_kernel(backend, 16)(new_segments, sp_out, n, ndrange=n)

    println("Ici le prob à mon avis : ", sp_out)
    out_ids = KernelAbstractions.allocate(backend, Int64, n)
    permute_indices_kernel(backend, 16)(out_ids, original_ids, flags, p, ndrange=in_length)

    println("Propagated heads : ", propagated_heads)
    println("Out segments is : ", new_segments)
    println("Out data is : ", out_points)
    println("P (perm ):  ", p)
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


function distance_from_hyperplane(backend, points, hyp_points)
    # Ici on le fait sur CPU, car tous les backend sont pas supportés avec la SVD et le dot(par exemple OneAPI).
    cpu_hyper_points = Array(hyp_points)
    cpu_points = Array(points)

    normal, offset = compute_hyperplane(cpu_hyper_points)
    println("SIMPLEX HYPER PLANE PARAMETERS: ", normal)
    println("PARAMETERS: ", offset)

    out_flags = fill(0, size(cpu_points)[2])
    for (index, p) in enumerate(eachcol(cpu_points))
        dist = signed_distance(normal, offset, p) / norm(normal)

        if dist > EPSILON
            out_flags[index] = 1
        elseif dist < -EPSILON
            out_flags[index] = 2
        else
            out_flags[index] = 0
        end
        #println(index, p, out_flags[index])
    end
    #println("outflag: ", out_flags)

    #Transfer to GPU
    out_flags_gpu = KernelAbstractions.allocate(backend, Int64, length(out_flags))
    copyto!(out_flags_gpu, out_flags)
    return out_flags_gpu
end


# O(n) :( -> On pourrait le faire en O(log n) sur gpu je pense bien à voir si ça vaut le coups dans le movement des donnés ect...
# Vu qu'on réduit à chaque fois la taille ça pourrait être inréressant d'avoir un système hybride GPU > threshold au sinon CPU
@kernel function mask_2d_kernel(output, data, n_flags)
    i, j = @index(Global, NTuple)
    output[i, j] = data[i] == j ? 1 : 0
end

#=function mask(data, pattern) 
    output = fill(0, length(data))
    
    for (index, val) in enumerate(data)
        if val == pattern 
            output[index] = 1 
        end
    end

    # Transfer sur GPU 
    gpu_output = KernelAbstractions.allocate(backend, Int64, length(data))
    copyto!(gpu_output, output)
    return gpu_output
end=#

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
        @print("Offset : ", offset, "  sh: ", sh[global_id], " ", forwardScanArray[global_id, flags[global_id]], "\n")
        
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
#TODO : PAS BON ÇA MON COCO
function flag_permute(backend, flags, segments, data_size, n_flags)

    if any(p -> p > n_flags || p < 1, flags)
        error("Flag values should be between 1 and n_flags")
    end

    #maskedArray = Matrix{Int64}(undef, data_size, n_flags)
    #scanArray = Matrix{Int64}(undef, data_size, n_flags)
    #backscanArray = Matrix{Int64}(undef, data_size, n_flags)
    
    maskedArray   = KernelAbstractions.zeros(backend, Int64, (length(flags), n_flags))
    scanArray     = KernelAbstractions.zeros(backend, Int64, (length(flags), n_flags))
    backscanArray = KernelAbstractions.zeros(backend, Int64, (length(flags), n_flags))

    # OUAAAIIIIS UNE MATRICE DE KERNELLL
    mask_2d_kernel(backend, (16, 16))(maskedArray, flags, n_flags, ndrange=(length(flags), n_flags))
    KernelAbstractions.synchronize(backend)

    println("-- Flag permute with input --")
    println(flags)
    println(segments)


    st = similar(segments)
    mark_segment_id(backend, 16)(st, segments, ndrange=length(segments))
    
    #TODO en parallèle normalement mais bon vu que je pige RIEN à ce qu'on me veut alors je fais en séquentiel pour l'instant
    #= for id=1:length(segments)
        if segments[id] == 1
            st[id] = id
        else
            st[id] = 0
        end 
    end=#

    sh = segmented_scan(backend, st, segments, ScanPrimitive.AddOp(), backward=false, inclusive=true)
    println("CIIII")
    
    #TODO: de la merde ça pour le GPU .....
    #=for i=1:n_flags
        @views maskedArray[:, i] .= mask(flags, i)
        @views scanArray[:, i] .=  segmented_scan(backend, maskedArray[:, i], segments, ScanPrimitive.AddOp())
        @views backscanArray[:, i] .= segmented_scan(backend, scanArray[:, i], segments, ScanPrimitive.MaxOp(), backward=true, inclusive=true, identity=typemin(Int64)) 
    end=#

    for i = 1:n_flags
        col_masked   = @view maskedArray[:, i]
        col_scan     = @view scanArray[:, i]
        col_backscan = @view backscanArray[:, i]
        
        col_scan     .= segmented_scan(backend, col_masked, segments, ScanPrimitive.AddOp(), inclusive=true)
        col_backscan .= segmented_scan(backend, col_scan,   segments, ScanPrimitive.MaxOp(), backward=true, inclusive=true, identity=typemin(Int64))
    end
    
    # Now the flag permute kernell :)
    outPermutation = similar(flags)
    flag_permute_kernell(backend, 16)(scanArray, backscanArray, sh, flags, data_size, outPermutation, ndrange=first(size(scanArray)))

    #println(scanArray, " length is : ", length(scanArray), " size to cmp ", size(scanArray))
    # Add segment kernell
    add_segment_kernel(backend, 16)(backscanArray, segments, sh, n_flags, ndrange=first(size(scanArray)))

    println("Out permutation is :", outPermutation)
    #println("New segments flags: ", segments)
    #println("INPUT FLAG: ", flags)
    #println("Masked Array: ", maskedArray)
    #println("***")
    println("Scan array :", scanArray)
    println("***")
    println("Back array : ", backscanArray)

    return outPermutation
end


@kernel function max_distance_kernel(distances, data, @uniform normal, @uniform offsets, @uniform n_points, @uniform dimensions)
    thread_id = @index(Local)
    global_id = @index(Global)

    if global_id ≤ n_points
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
    KernelAbstractions.synchronize(backend)
    return collect(points_idx)
end

@kernel function permute_points_kernel(out, data, perm, @uniform dim)
    global_id = @index(Global)
    for d in 1:dim
        out[d, perm[global_id]] = data[d, global_id]
    end
end

function permute_points(backend, points, perm, dim)
    out = KernelAbstractions.zeros(backend, Float64, (dim, size(points, 2)))
    permute_points_kernel(backend, 16)(out, points, perm, dim, ndrange=size(points, 2))
    return out
end


function propagate_segment_idx(backend, segments)
    println("propagate_segment_idx: ", segments)
    n = length(segments)
    global_head = zeros(n)
    global_head[1] = 1

    global_head_gpu = KernelAbstractions.allocate(backend, Int64, n)
    copyto!(global_head_gpu, global_head)

    seg_id = segmented_scan(backend, segments, global_head_gpu, ScanPrimitive.AddOp(), backward=false, inclusive=true)
    KernelAbstractions.synchronize(backend)
    return seg_id, Array(seg_id)[n]
end

@kernel function distance_to_face_kernel(distances, points, seg_id, seg_to_face_map, normals, offsets, @uniform dim)
    gid = @index(Global)
    n = length(seg_id)
    if gid <= n
        sid = seg_id[gid]
        face_col = seg_to_face_map[sid]
        acc = 0.0
        for d in 1:dim
            acc += points[d, gid] * normals[d, face_col]
        end
        distances[gid] = acc + offsets[face_col]
        @print("Distance to face kernel: point ", gid, " with seg id ", sid, " mapped to face ", face_col, " has distance ", distances[gid], "\n")
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

function rebuild_face_hyperplanes_from_heads(backend, face_points, points, segments, dim)
    seg_id, n_segs = propagate_segment_idx(backend, segments)

    @warn "Cette fonction n'est pas correcte"

    println("N segment: ", n_segs)
    normals = fill(0.0, (dim, n_segs))
    offsets = fill(0.0, n_segs)

    # Pour chaque head, on oriente la face correspondante pour que head soit côté positif
    for i in 1:length(segments)
        if Array(segments)[i] == 1 # TODO: MEH ça doit ralentire de bz ça.
            sid = Array(seg_id)[i]
            face = face_points[sid]
            @views sample = points[:, i]

            n, b = compute_hyperplane(face)
            if signed_distance(n, b, sample) < 0
                n = -n
                b = -b
            end
            normals[:, sid] = n
            offsets[sid] = b
        end
    end

    normals_gpu = KernelAbstractions.allocate(backend, Float64, (dim, n_segs))
    offsets_gpu = KernelAbstractions.allocate(backend, Float64, n_segs)
    
    copyto!(normals_gpu, normals)
    copyto!(offsets_gpu, offsets)
    return normals_gpu, offsets_gpu
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

"""
is_inside_simplex(simplex, point, dim)

Returns true if the point is inside or on the edge of the simplex. Returns false otherwise.
# Examples
```jldoctest
julia> is_inside_simplex(.....)
```
"""
function is_inside_simplex(simplex, point, dim)
    println("simplex : ", simplex)
    rn = simplex[:, end] 
    T = simplex[:, 1:end-1] .- rn 
    T_inv = inv(T)
    
    λ = T_inv * (point .- rn) 
    λ_n = 1.0 - sum(λ)
    println(λ, " &  ", λ_n)
    for λ_i in λ
        #println("each lambda i : ", λ_i)
        if λ_i <= -EPSILON
            return false
        end
    end
    if λ_n <= -EPSILON
        return false
    end
    return true
end

@kernel function compact_data_to_segment(out_idx, out_dist, segments, seg_id, seg_far_idx, seg_distance)
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
    println("Starting BFS with face id ", face_id)
    while !isempty(queue)
        current_face_id = popfirst!(queue)

        for neighbour in mesh.neighbors[current_face_id]
            if neighbour == 0 || neighbour in visible_faces || mesh.active[neighbour] == false
                continue
            end

            normal = mesh.normals[neighbour]
            offset = mesh.offsets[neighbour]

            # println("Checking neighbor ", neighbour, " with normal ", normal, " and offset ", offset)
            # println("Current face normal: ", mesh.normals[current_face_id], " offset: ", mesh.offsets[current_face_id])
            # println("Point to insert: ", points[:, point_idx], " signed distance: ", signed_distance(normal, offset, points[:, point_idx]))

            if signed_distance(normal, offset, points[:, point_idx]) > EPSILON
                push!(queue, neighbour)
                push!(visible_faces, neighbour)
            end
        end
    end

    println("Visible faces: ", visible_faces)
    return visible_faces
end

# TODO: J'ai vu qu'il y avais moyens de faire un BFS sur GPU en utilisant une représentation RCS du graphe. À voir si j'ai le temps. 
function insert_point_and_update_mesh(mesh, point_idx, face_id, points, dim)

    # Maintenant qu'on a nos faces visible on trouve l'horizon entre les faces visibles et les faces non visibles.
    #TODO: paralléliser sur GPU ?
    visible_faces = get_visible_faces(mesh, point_idx, face_id, points)

    horizon_ridges = [] 
    for f in visible_faces
        for (k, neighbor) in enumerate(mesh.neighbors[f])
            if neighbor > 0 && mesh.active[neighbor] && !(neighbor in visible_faces)
                ridge_verts = [mesh.vertices[f][j] for j in 1:dim if j != k]
                push!(horizon_ridges, (ridge_verts, neighbor, f))
            end
        end
        mesh.active[f] = false
    end

    println("Horizon ridges: ", horizon_ridges)

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

    println("Mesh after insertion: ")
    for i in 1:length(mesh.vertices)
        println("Face ", i, ": vertices ", mesh.vertices[i], " neighbors ", mesh.neighbors[i], " normal ", mesh.normals[i], " offset ", mesh.offsets[i], " active ", mesh.active[i])
    end
end

@kernel function assign_face_to_point(face_flags, compact_flags, points, normals, offsets, @uniform dim)
    global_id = @index(Global)
    if global_id <= size(points, 2)
        max_d = 1e-9 # EPSILON
        best_face = 0
        for f in 1:size(normals, 2)
            d = 0.0
            for d_i in 1:dim
                d += normals[d_i, f] * points[d_i, global_id]
            end
            d += offsets[f]
            
            if d > max_d
                @print("New best face for point ", global_id, " is face ", f, " with distance ", d, "\n")
                max_d = d
                best_face = f
            end
        end
        face_flags[global_id] = best_face
        compact_flags[global_id] = (best_face == 0) ? 0 : 1 
    end
end

function quick_hull(backend, points, n_points = size(points, 2), dim = size(points, 1))
    convex_hull_bounds = []
    n_iter = 0
    # On init tout à 2 car on a 2 facettes au début.
    mesh = QhullData([], [], [], [], [])

    # Create a simplex by finding min & max points allong all dimensions
    simplex_idx = compute_simplex(points, dim, backend)

    println("simplex idx: ", simplex_idx)
    simplex_idx = simplex_idx[1:dim+1]
    simplex_matrix = points[:, simplex_idx]

    barycenter = barycenter = sum(simplex_matrix, dims=2)[:] ./ (dim + 1)
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

    original_ids_cpu = collect(1:n_points)
    original_ids = KernelAbstractions.allocate(backend, Int64, n_points)
    copyto!(original_ids, original_ids_cpu)


     for i in 1:size(simplex_matrix, 2)
        push!(convex_hull_bounds, copy(simplex_matrix[:, i]))
    end

    segments_cpu = fill(0, n_points)
    segments_cpu[1] = 1
    segments = KernelAbstractions.zeros(backend, Int64, n_points)
    copyto!(segments, segments_cpu)

    remove_flags = fill(1, n_points)
    for idx in simplex_idx
        remove_flags[idx] = 0
    end

    remove_flags_gpu = KernelAbstractions.allocate(backend, Int64, n_points)
    copyto!(remove_flags_gpu, remove_flags)

    println("remove flags: ", remove_flags_gpu, " segments: ", segments, points, original_ids, n_points, dim)
    restant, original_ids, rest_segment = compact(backend, remove_flags_gpu, segments, points, original_ids, n_points, dim)

    n_active = length(mesh.active)
    active_normals = hcat(mesh.normals...)
    active_offsets = vcat(mesh.offsets...)

    face_normals_gpu = KernelAbstractions.allocate(backend, Float64, (dim, n_active))
    face_offsets_gpu = KernelAbstractions.allocate(backend, Float64, n_active)
    copyto!(face_normals_gpu, active_normals)
    copyto!(face_offsets_gpu, active_offsets)

    n = size(restant, 2)
    face_flags_gpu = KernelAbstractions.allocate(backend, Int64, n)
    compact_flags_gpu = KernelAbstractions.allocate(backend, Int64, n)

    assign_face_to_point(backend, 16)(face_flags_gpu, compact_flags_gpu, restant, face_normals_gpu, face_offsets_gpu, dim, ndrange=n)
    KernelAbstractions.synchronize(backend)
    restant, original_ids, rest_segment, cp, cflags, cn = compact(backend, compact_flags_gpu, rest_segment, restant, original_ids, n, dim)
    
    
    compacted_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
    permute_indices_kernel(backend, 16)(compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n)
    KernelAbstractions.synchronize(backend)

    compacted_flags_cpu = Array(compacted_flags_gpu)
    current_unique_flags = sort(unique(compacted_flags_cpu))
    n_total_flag = length(current_unique_flags)
    mapping = Dict(old => new for (new, old) in enumerate(current_unique_flags))
    mapped_flags_cpu = [mapping[f] for f in compacted_flags_cpu]
    
    
    mapped_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
    copyto!(mapped_flags_gpu, mapped_flags_cpu)
    
    out_perm = flag_permute(backend, mapped_flags_gpu, rest_segment, size(restant, 2), n_active)
    
    restant_new = KernelAbstractions.zeros(backend, Float64, (dim, size(restant, 2)))
    ids_new = KernelAbstractions.zeros(backend, Int64, size(restant, 2))
    
    permute_points_kernel(backend, 16)(restant_new, restant, out_perm, dim, ndrange=size(restant, 2))
    permute_kernel(backend, 16)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))


    compacted_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
    permute_indices_kernel(backend, 16)(compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=size(restant, 2))
    

    cf_cpu = Array(compacted_flags_gpu)
    current_unique_flags = sort(unique(cf_cpu))
    mapping = Dict(old => new for (new, old) in enumerate(current_unique_flags))
    mapped_flags_gpu = KernelAbstractions.allocate(backend, Int64, length(cf_cpu))
    copyto!(mapped_flags_gpu, [mapping[f] for f in cf_cpu])
    
    restant = restant_new
    original_ids = ids_new

    println("Simplex index : ", simplex_idx)
    println("Simplex_matrix: ", simplex_matrix)


    while size(restant, 2) > 0
        println("######### NEW ITTERATION $n_iter ############")

        if n_iter > 5
            @warn "Too many iterations, something might be wrong."
            break
        end

        seg_id, n_segs = propagate_segment_idx(backend, rest_segment)
        n = size(restant, 2)
        println("RESTANT: ", n)
        println(restant)
        if n_segs == 0
            break
        end
        
        distances = KernelAbstractions.allocate(backend, Float64, n)
        println("Face normals gpu: ", face_normals_gpu)
        println("Face offsets gpu: ", face_offsets_gpu)

        seg_to_face_map_gpu = KernelAbstractions.allocate(backend, Int64, length(current_unique_flags))
        copyto!(seg_to_face_map_gpu, current_unique_flags)

        distance_to_face_kernel(backend, 16)(distances, restant, seg_id, seg_to_face_map_gpu, face_normals_gpu, face_offsets_gpu, dim, ndrange=n)
        KernelAbstractions.synchronize(backend)
        
        println("Restant: ", restant)
        println("Distances : ", distances)
        #println(rest_segment)
        

        # On propage le point le plus loins à travers le segement (forward + backward pass)
        prefix_max = segmented_scan(backend, distances, rest_segment, ScanPrimitive.MaxOp(),  backward=false, inclusive=true, identity=typemin(Float64))
        seg_max    = segmented_scan(backend, prefix_max, rest_segment, ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Float64))

        cand_idx = KernelAbstractions.allocate(backend, Int64, n)
        mark_farthest_candidate_kernel(backend, 16)(cand_idx, distances, seg_max, 1e-12, ndrange=n)
        KernelAbstractions.synchronize(backend)

        #println("candIdx: ", cand_idx)
        prefix_far = segmented_scan(backend, cand_idx, rest_segment, ScanPrimitive.MaxOp(), backward=false, inclusive=true, identity=typemin(Int64))
        far_idx_p  = segmented_scan(backend, prefix_far, rest_segment, ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Int64))
        
        far_idx  = KernelAbstractions.allocate(backend, Int64,  n_segs)
        far_dist = KernelAbstractions.allocate(backend, Float64, n_segs)
        compact_data_to_segment(backend, 16)(far_idx, far_dist, rest_segment, seg_id, far_idx_p, seg_max, ndrange=n)
        
        #TODO: ÇA comment a être le bordel.
        far_idx = Array(far_idx)
        far_dist = Array(far_dist)
        original_ids_cpu = Array(original_ids)
        active_face_indices = findall(mesh.active)
        println("far_idx:", far_idx)

        candidates = []
        for i in 1:n_segs
            if far_idx[i] != 0 && far_dist[i] > EPSILON
                push!(candidates, (dist = far_dist[i], local_idx = far_idx[i], seg_idx = i))
            end
        end

        # TODO: tri sur GPU (radixsort)
        sort!(candidates, by = x -> x.dist, rev = true)
        println("Original ids CPU: ", original_ids_cpu)

        # Getting indenpendant points to add to void add conflict
        to_remove_faces = Set{Int}() 
        points_to_insert = []
        println("Candidates triés : ", candidates)
        println("current compacteed flags gpu flags : ", compacted_flags_gpu)
        for cand in candidates
            global_point_idx = original_ids_cpu[cand.local_idx]
            local_face_idx = current_unique_flags[cand.seg_idx]
            global_face_id = active_face_indices[local_face_idx]

            println("Processing candidate point ", global_point_idx, " with distance ", cand.dist, " in segment ", cand.seg_idx)
            
            visible_faces = get_visible_faces(mesh, global_point_idx, global_face_id, points)

            conflict = false
            for f in visible_faces
                if f in to_remove_faces
                    conflict = true
                    break
                end
            end

            if !conflict
                println("++++++++ Point ", global_point_idx, " is independent and will be inserted.")
                push!(points_to_insert, (global_point_idx, global_face_id))
                union!(to_remove_faces, visible_faces) 
            end
        end

        println("Sur $(length(candidates)) candidats , $(length(points_to_insert)) points indépendants insérés!")

        for (point_idx, face_id) in points_to_insert
            println("Adding to the convexhull : ", points[:, point_idx], " with face id ", face_id)
            push!(convex_hull_bounds, points[:, point_idx])
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

        face_normals_gpu = KernelAbstractions.allocate(backend, Float64, (dim, n_active))
        face_offsets_gpu = KernelAbstractions.allocate(backend, Float64, n_active)
        copyto!(face_normals_gpu, active_normals)
        copyto!(face_offsets_gpu, active_offsets)

        face_flags_gpu = KernelAbstractions.allocate(backend, Int64, n)
        compact_flags_gpu = KernelAbstractions.allocate(backend, Int64, n)
        assign_face_to_point(backend, 16)(face_flags_gpu, compact_flags_gpu, restant, face_normals_gpu, face_offsets_gpu, dim, ndrange=n)
        KernelAbstractions.synchronize(backend)

        #println("Compact flag to remove inside points: ", compact_flags)
        println("Point restant : ", restant)
        println("Face flags : ", face_flags_gpu)
        println("Compact flags : ", compact_flags_gpu)

        global_seg_reset = zeros(Int64, size(restant, 2))
        if length(global_seg_reset) > 0
            global_seg_reset[1] = 1
        end
        copyto!(rest_segment, global_seg_reset)

        # Now Remove points inside
        restant, original_ids, rest_segment, cp, cflags, cn = compact(backend, compact_flags_gpu, rest_segment, restant, original_ids, n, dim)
        
        println("Restant after compact: ", restant)
        println("Rest segment after compact: ", rest_segment)
        println("Original ids after compact: ", original_ids)

        if length(restant) == 0
            println(convex_hull_bounds)
            break
        end
        
        compacted_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
        println("Compacting flags for permutation, cflags: ", cflags, " cp: ", cp, " cn: ", cn)
        permute_indices_kernel(backend, 16)(compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n)
        KernelAbstractions.synchronize(backend)

        println("Compacted flags GPU: ", compacted_flags_gpu)
        
        compacted_flags_cpu = Array(compacted_flags_gpu)
        current_unique_flags = sort(unique(compacted_flags_cpu))
        n_total_flag = length(current_unique_flags)
        mapping = Dict(old => new for (new, old) in enumerate(current_unique_flags))
        mapped_flags_cpu = [mapping[f] for f in compacted_flags_cpu]
        
        mapped_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
        copyto!(mapped_flags_gpu, mapped_flags_cpu)

        out_perm = flag_permute(backend, mapped_flags_gpu, rest_segment, size(restant, 2), n_active)
        new_restant = permute_points(backend, restant, out_perm, dim)
        ids_new = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
        
        println("Original ids before perm: ", original_ids)
        permute_kernel(backend, 16)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))
        #println("Out permutation pour la prochaine ittération : ", out_perm)
        #println("Pour la prochaine ittération :", restant)
        println("Pour la prochaine ittération: ", current_unique_flags)
        
        restant = new_restant
        original_ids = ids_new
        println("-----------------------------------------------------", restant)
        n_iter += 1
    end

    sorted_hull = sort_points_counter_clockwise(hcat(convex_hull_bounds...))

    return sorted_hull
end

# ---- Truc pour lire les fichiers
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
    barycenter = sum(points, dims=2)[:] ./ size(points, 2)
    angles = atan.(points[2, :] .- barycenter[2], points[1, :] .- barycenter[1])
    sorted_indices = sortperm(angles)
    return points[:, sorted_indices]
end

#=
points = [ 0.0 -2.0  0.0  2.0  3.0 -1.0 ;
           2.0  0.0 -2.0  0.0  3.0  1.0 ]

points = [-1.0 0.0 1.0 0.0 ;
           0.0 1.0 0.0 -1.0]

exit(-1)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (2,4))
copyto!(gpu_pts, points)

println("size: ", size(gpu_pts))
quick_hull(gpu_pts, 4, 2)=#

#end
