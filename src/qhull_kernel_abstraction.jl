module qhull_kernel_abstraction
using KernelAbstractions
using LinearAlgebra
using GLMakie, ColorSchemes

include(joinpath(@__DIR__, "..", "src", "ScanPrimitive.jl"))
include(joinpath(@__DIR__, "..", "src", "MinMaxReduction.jl"))

using .ScanPrimitive
using .MinMaxReduction

EPSILON = 1e-9

# TODO: BIG TRUC: HARMONISER LES WORKGROUP SIZE

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

"""
compact(flags, segments, data, length, dim)

removes elements within segments.

Compact takes in an array of booleans `b` and
discards data from `s`, which are marked as false by the
boolean array

Returns a vector p, a permutation of data and a vector sp, a permutation of segments head.

```
"""
function compact(flags, segments, data, in_length, dim)
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


    #println("Propagated heads : ", propagated_heads)
    #println("Out segments is : ", new_segments)
    #println("Out data is : ", out_points)
    #println("P (perm ):  ", p)
    return out_points, new_segments
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


function distance_from_hyperplane(points, hyp_points)
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
function flag_permute(flags, segments, data_size, n_flags)

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


function propagate_segment_idx(segments)
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

function rebuild_face_hyperplanes_from_heads(face_points, points, segments, dim)
    seg_id, n_segs = propagate_segment_idx(segments)

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

function quick_hull(points, n_points, dim)
    convex_hull_bounds = []
    segments_cpu = fill(0, n_points)
    segments_cpu[1] = 1
    
    segments = KernelAbstractions.zeros(backend, Int64, n_points)
    copyto!(segments, segments_cpu)

    # Create a simplex by finding min & max points allong all dimensions
    simplex_idx = compute_simplex(points, dim, backend)

    println("simplex idx: ", simplex_idx)
    simplex_idx = simplex_idx[1:dim] #TODO pour l'instant c'est la hess mais on prends que les d premiers points pour le simplex

    ##################### MAKIGL ##################### 
    #= all_points  = [Point3f(points[:, i]) for i in 1:size(points, 2)]
    #hull_pts_matrix = Array(convex_hull)
    simplex_nodes = [Point3f(points[:, i]) for i in simplex_idx]

    println("simplex nodes ", simplex_nodes)

    fig = Figure(resolution = (1200, 800))
    ax  = Axis3(fig[1, 1], title = "Résultat QuickHull GPU", aspect = :data)
    mesh!(ax, simplex_nodes, [1, 2, 3])
    scatter!(ax, points, color = :grey, markersize = 12,   label = "Tous les points")
    scatter!(ax, simplex_nodes, color = :red, markersize = 20,   label = "Simplex")
    =#
   

    
    simplex_matrix = points[:, simplex_idx]

    for i in 1:size(simplex_matrix, 2)
        push!(convex_hull_bounds, copy(simplex_matrix[:, i]))
    end

    println("Simplex index : ", simplex_idx)
    println("Simplex_matrix: ", simplex_matrix)

    remove_flags = fill(1, n_points)
    for idx in simplex_idx
        remove_flags[idx] = 0
    end
    
    remove_flags_GPU = KernelAbstractions.allocate(backend, eltype(remove_flags), Int(length(remove_flags)))
    copyto!(remove_flags_GPU, remove_flags)

    restant, rest_segment = compact(remove_flags_GPU, segments, points, n_points, dim)
    println("SEGMENT APRÈS LE COMPACT: ", rest_segment)

    dist_flags = distance_from_hyperplane(restant, simplex_matrix)
    # Now that we have our simplex perfom the algorithm
    println("Restant après le compact de la face: ", restant)
    #TODO : compact avec les segments

    outPermutation = flag_permute(dist_flags, rest_segment, length(rest_segment), 2)    
    println("Outpermutation: ", outPermutation)

    restant = permute_points(backend, restant, outPermutation, dim)
    println("REST Segment: ", rest_segment)

    println("Restant après permutation: ", restant)
    face_points = [simplex_matrix, simplex_matrix]
    
    #=for k in 1:dim
        cols = [j for j in 1:dim if j != k] 
        face_points[k] = simplex_matrix[:, cols]
    end=#

    println("face_points : ", face_points)
    normal, offset = compute_hyperplane(face_points[1])

    face_normals = fill(0.0, (dim, 2))
    face_offsets = fill(0.0, 2)

    face_normals[:,1] = normal
    face_offsets[1] = offset
    
    face_normals[:, 2] = -normal
    face_offsets[2] = -offset
    
    #face_normals, face_offsets = rebuild_face_hyperplanes_from_heads(face_points, restant, rest_segment, dim)

    println("normals: ", face_normals)
    println("offset:", face_offsets)
    n_faces = size(face_normals, 2)
    #=arrow_directions = [Vec3f(face_normals[:, i]) for i in 1:n_faces]
    arrow_points = [Point3f(face_offsets[i] .* face_normals[:, i]) for i in 1:n_faces]
    colors = [ColorSchemes.tab10[i] for i in 1:n_faces]

    println("Arrow points :", arrow_points)
    println("Arrow directions :", arrow_directions)
    
    arrows!(ax, arrow_points, arrow_directions;
        arrowsize = 0.1,
        linewidth  = 0.05,
        color      = colors
    ) =#
    println(face_normals, face_offsets)
    

    # On affiche les 2 normals des plans 

    #=axislegend(ax)
    DataInspector(fig)
    display(fig) 
    
     if !isinteractive()
        println("Fenêtre ouverte. Appuyez sur Entrée pour terminer.")
        readline()
    end

    =#


    while size(restant, 2) > 0
        println("######### NEW ITTERATION ############")
        # For each segments we compute the hyperplane corresponding to them.
        seg_id, n_segs = propagate_segment_idx(rest_segment)
        n = size(restant, 2)
        println("RESTANT: ", n)
        if n_segs == 0
            break
        end
        
        distances = KernelAbstractions.allocate(backend, Float64, n)
        distance_to_face_kernel(backend, 16)(distances, restant, seg_id, face_normals, face_offsets, dim, ndrange=n)
        KernelAbstractions.synchronize(backend)
        
        println("Restant: ", restant)
        println("Distances : ", distances)
        #println(rest_segment)
        

        # On propage le point le plus loins à travers le segement (forward + backward pass)
        prefix_max = segmented_scan(backend, distances, rest_segment, ScanPrimitive.MaxOp(),  backward=false, inclusive=true, identity=typemin(Float64))
        seg_max    = segmented_scan(backend, prefix_max, rest_segment, ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Float64))
        
        #println(prefix_max)
        #println(seg_max)
        #println(rest_segment)


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

        println("far_idx:", far_idx)
        #return;
        #println(far_dist)
        # Maintenant on forme les différents simplexes et on fait le test de pts comme avant la boucle while
        
        # Toutes les nouvelles faces créé
        # TODO: refactor cette partie, elle est moche
        children_faces = Vector{Vector{Matrix{Float64}}}(undef, n_segs)

        for seg_id in 1:n_segs
            # Le cas où le segment ne sert plus à rien (pas de pts le plus loin)
            if far_idx[seg_id] == 0 || far_dist[seg_id] < EPSILON
                continue
            end

            println("Adding to the convexhull : ", restant[:, far_idx[seg_id]])
            push!(convex_hull_bounds, copy(restant[:, far_idx[seg_id]]))

            face = face_points[seg_id]

            # Quand on ajoute notre point on crée k nouvelle faces et on en supprime une: 
            child_face = Vector{Matrix{Float64}}(undef, dim)

            for k in 1:dim # On enlève le vertex k pour le remplacer par p.
                cols = Vector{Int}(undef, dim - 1)
                t = 1
                @inbounds for j in 1:dim
                    if j != k
                        cols[t] = j
                        t += 1
                    end
                end
                child_face[k] = hcat(face[:, cols], restant[:, far_idx[seg_id]]) # dim x dim
            end
            children_faces[seg_id] = child_face
        end

        #println(children_faces)

        #Test each point for inclusion within the simplex. If it is inside, then it cannot be part of the convex hull. Throw the point out.
        compact_flags = fill(0, n)
        for point_idx in 1:n
            seg = Array(seg_id)[point_idx]
            far_id  = far_idx[seg]

            if far_id == 0 || far_dist[seg] <= 0 || point_idx == far_id
                continue
            end
            
            simplex_matrix = hcat(face_points[seg], restant[:, far_id])
            point_to_test = restant[:, point_idx]

            println("Testing if point : ", point_to_test, "is inside the simplex: ", simplex_matrix)
            # Si le point est dans le simplexe, on l'enlève
            if !is_inside_simplex(simplex_matrix, point_to_test, dim)
                println("IT IS NOT INSIDEEEEEE")
                compact_flags[point_idx] = 1
            end
        end

        #println("Compact flag to remove inside points: ", compact_flags)
        println("Point restant : ", restant)

        compact_flag_gpu = KernelAbstractions.allocate(backend, eltype(compact_flags), length(compact_flags))
        copyto!(compact_flag_gpu, compact_flags)
        # Now Remove points inside
        restant, rest_segment = compact(compact_flag_gpu, rest_segment, restant, n, dim)

        if length(restant) == 0
            println(convex_hull_bounds)
            break
        end

        #println("Restant avant la permutation: ", restant)
        
        # Now on assigne un flag pour chaque nouvelles faces et on aura fini :))))))))))
        face_flag = fill(0, size(restant, 2))
        total_of_flags = 0

        flag_counter = zeros(size(restant,2))
        n_total_flag = 0
        
        #TODO: BIG TRUC, SEQUENCE DE FLAAAAAGG  
        for point_idx in 1:size(restant, 2)
            max_dist = EPSILON
            best_face_idx = 0
            face_counter = 1             
            
            for seg_id in 1:n_segs
                if !isassigned(children_faces, seg_id)
                    continue
                end                
                for face in children_faces[seg_id]
                    normal, offset = compute_hyperplane(face)
                    distance = signed_distance(normal, offset, restant[:, point_idx])
                    
                    if distance > max_dist
                        max_dist = distance
                        best_face_idx = face_counter
                    end
                    face_counter += 1
                end
            end

           
        end
        
        # TODO: surement moyen de faire plus rapide que ça .
        unique_flags = sort(unique(face_flag))
        n_total_flag = length(unique_flags)
        mapping = Dict(old => new for (new, old) in enumerate(unique_flags))
        face_flag = [mapping[f] for f in face_flag]
        
        #println("face flags: ", face_flag)
        #println("rest_segment: ", rest_segment)
        
        out_perm = flag_permute(face_flag, rest_segment, length(face_flag), n_total_flag)
        restant = permute_points(backend, restant, out_perm, dim)
        
        #println("Out permutation pour la prochaine ittération : ", out_perm)
        #println("Pour la prochaine ittération :", restant)
        #println("Pour la prochaine ittération: ", rest_segment)

        # On recrée notre liste de points pour chaque face
        new_face_list = Matrix{Float64}[]
        for i in 1:n_segs
            if isassigned(children_faces, i)
                for f in children_faces[i]
                    push!(new_face_list, f)
                end
            end
        end
        face_points = new_face_list

        face_normals, face_offsets = rebuild_face_hyperplanes_from_heads(face_points, restant, rest_segment, dim)

    end

    hull_matrix = hcat(convex_hull_bounds...)
    println("Convex hull points : ", hull_matrix)
    return hull_matrix
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

end
