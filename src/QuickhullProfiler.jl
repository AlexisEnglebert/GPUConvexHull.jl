"""
QuickhullProfiler — instrumentation complète pour GPUQuickhull
===============================================================

Usage minimal :
    include("quickhull_instrumentation.jl")
    prof = QuickhullProfiler()
    result = quick_hull_profiled(backend, points, prof)
    print_report(prof)
    plot_breakdown(prof, n_points)   # nécessite Plots.jl

Catégories mesurées :
    :pcie_transfer   — Array() GPU→CPU dans la boucle chaude
    :gpu_alloc       — KernelAbstractions.allocate dans la boucle
    :kernel_assign   — assign_face_to_point_kernel
    :kernel_distance — distance_to_face_kernel
    :kernel_compact  — compact() (scan + permute kernels)
    :kernel_sort     — flag_permute (radixsort)
    :scan_primitive  — segmented_scan (propagation max)
    :cpu_mesh_bfs    — get_visible_faces (BFS CPU)
    :cpu_mesh_update — insert_point_and_update_mesh
    :cpu_candidates  — tri + conflict check des candidats
    :sync_overhead   — KernelAbstractions.synchronize isolés

Note sur la précision :
    CUDA.@elapsed insère des CUDA Events hardware → précision ~0.5µs.
    KernelAbstractions.synchronize() est appelé avant chaque mesure
    pour éviter que l'asynchronisme GPU ne pollue les temps CPU.
"""

using CUDA
using KernelAbstractions
using Printf          # ← ajoute cette ligne
export Profiler, reset!, record_time!, print_report, plot_breakdown

# ─── Struct principale ────────────────────────────────────────────────────────

mutable struct Profiler
    times   ::Dict{Symbol, Float64}   # temps cumulés par catégorie (secondes)
    counts  ::Dict{Symbol, Int}        # nb d'appels par catégorie
    n_iters ::Int                      # nb d'itérations de la boucle principale
    n_points_initial ::Int
    dim     ::Int

    function Profiler()
        cats = [:pcie_transfer, :gpu_alloc, :kernel_assign, :kernel_distance,
            :kernel_compact, :kernel_sort, :kernel_mark_farthest,  # ← séparé
            :scan_primitive, :ak_reduce,                            # ← nouveau
            :compute_simplex,                                        # ← nouveau
            :cpu_mesh_bfs, :cpu_mesh_update, :cpu_candidates, :sync_overhead]
        new(
            Dict(c => 0.0 for c in cats),
            Dict(c => 0   for c in cats),
            0, 0, 0
        )
    end
end

function reset!(p::Profiler)
    for k in keys(p.times)
        p.times[k]  = 0.0
        p.counts[k] = 0
    end
    p.n_iters = 0
end

# ─── Macro de mesure GPU (CUDA Events) ───────────────────────────────────────

"""
    @gpu_time(prof, category, expr)

Mesure le temps GPU de `expr` via CUDA Events et l'accumule dans `prof`.
Appelle synchronize() avant pour vider le pipeline GPU.
"""
macro gpu_time(prof, category, expr)
    return quote
        local _p  = $(esc(prof))
        local _c  = $(esc(category))
        # Vider le pipeline GPU pour isoler la mesure
        CUDA.synchronize()
        local _t = CUDA.@elapsed $(esc(expr))
        _p.times[_c]  += _t
        _p.counts[_c] += 1
        _t
    end
end

"""
    @cpu_time(prof, category, expr)

Mesure le temps CPU+GPU de `expr` (avec synchronize final).
Pour les sections qui mélangent CPU et GPU (BFS, mesh update).
"""
macro cpu_time(prof, category, expr)
    return quote
        local _p = $(esc(prof))
        local _c = $(esc(category))
        local _t0 = time_ns()
        $(esc(expr))
        CUDA.synchronize()
        local _t1 = time_ns()
        _p.times[_c]  += (_t1 - _t0) * 1e-9
        _p.counts[_c] += 1
    end
end

# ─── Rapport texte ────────────────────────────────────────────────────────────

const CATEGORY_LABELS = Dict(
    :pcie_transfer   => "PCIe Transfers (GPU→CPU)",
    :gpu_alloc       => "GPU Allocations (cudaMalloc)",
    :kernel_assign   => "Kernel: assign_face_to_point",
    :kernel_distance => "Kernel: distance_to_face",
    :kernel_compact  => "Compact (scan + permute)",
    :kernel_sort     => "Sort (flag_permute / radixsort)",
    :scan_primitive  => "Segmented scan (propagation max)",
    :cpu_mesh_bfs    => "CPU: BFS get_visible_faces",
    :cpu_mesh_update => "CPU: insert_point_and_update_mesh",
    :cpu_candidates  => "CPU: candidates sort + conflict check",
    :sync_overhead   => "Synchronize overhead",
    :kernel_mark_farthest => "Kernel: mark_farthest_candidate",
    :ak_reduce            => "AcceleratedKernels.reduce (n_segs)",
    :compute_simplex      => "Compute simplex (init)",
)

function print_report(p::Profiler)
    total = sum(values(p.times))
    if total == 0.0
        println("⚠️  Aucun temps enregistré. As-tu lancé quick_hull_profiled ?")
        return
    end

    println("\n" * "="^72)
    println("  GPUQuickhull — Breakdown de performance")
    println("  N=$(p.n_points_initial) points | dim=$(p.dim) | $(p.n_iters) itérations")
    println("="^72)
    @printf("  %-38s  %8s  %6s  %7s\n", "Catégorie", "Temps(ms)", "%", "Appels")
    println("-"^72)

    # Trier par temps décroissant
    sorted = sort(collect(p.times), by=x->x[2], rev=true)
    for (cat, t) in sorted
        pct   = 100.0 * t / total
        label = get(CATEGORY_LABELS, cat, string(cat))
        cnt   = p.counts[cat]
        @printf("  %-38s  %8.2f  %5.1f%%  %7d\n", label, t*1000, pct, cnt)
    end
    println("-"^72)
    @printf("  %-38s  %8.2f  %5.1f%%\n", "TOTAL", total*1000, 100.0)
    println("="^72)

    # Résumé GPU vs CPU
    gpu_time = p.times[:kernel_assign] + p.times[:kernel_distance] +
               p.times[:kernel_compact] + p.times[:kernel_sort] +
               p.times[:kernel_mark_farthest] +
               p.times[:scan_primitive] + p.times[:ak_reduce]  
    cpu_time = p.times[:cpu_mesh_bfs] + p.times[:cpu_mesh_update] +
               p.times[:cpu_candidates]
    overhead = p.times[:pcie_transfer] + p.times[:gpu_alloc] + p.times[:sync_overhead]

    println("\n  Résumé :")
    @printf("    Calcul GPU effectif : %6.1f%%\n", 100*gpu_time/total)
    @printf("    Calcul CPU (mesh)   : %6.1f%%\n", 100*cpu_time/total)
    @printf("    Overhead (PCIe+alloc+sync) : %6.1f%%\n", 100*overhead/total)
    println()
end

# ─── Génération figure matplotlib via PyCall (optionnel) ─────────────────────

"""
    save_csv(prof, filename)

Sauvegarde le breakdown dans un CSV récupérable en local via scp.
"""
function save_csv(p::Profiler, filename::String)
    total = sum(values(p.times))
    open(filename, "w") do f
        println(f, "category,time_ms,percent,calls")
        for (cat, t) in sort(collect(p.times), by=x->x[2], rev=true)
            pct = 100.0 * t / total
            cnt = p.counts[cat]
            println(f, "$(cat),$(round(t*1000, digits=4)),$(round(pct, digits=2)),$(cnt)")
        end
    end
    println("CSV sauvegardé : $filename")
end


#=
# =============================================================================
#  VERSION INSTRUMENTÉE DE _quick_hull_implem
#  Copie-colle ceci dans quickhull.jl en remplacement de la fonction originale,
#  après avoir ajouté `using .QuickhullProfiler` en haut du fichier.
# =============================================================================

using Printf
using .QuickhullProfiler: Profiler, @gpu_time, @cpu_time, print_report, plot_breakdown

"""
    quick_hull_profiled(backend, points, prof; workgroup_size=256)

Version instrumentée de quick_hull. Remplit `prof` avec les temps de chaque section.

Exemple :
    prof = Profiler()
    for n in [1_000, 10_000, 100_000, 1_000_000]
        pts = randn(3, n)
        reset!(prof)
        quick_hull_profiled(CUDABackend(), pts, prof)
        print_report(prof)
    end
"""
function quick_hull_profiled(backend, points, prof::Profiler,
                              n_points::Int64 = size(points, 2),
                              workgroup_size::Int64 = 256)

    prof.n_points_initial = n_points
    prof.dim = size(points, 1)
    dim = prof.dim

    context = create_quickhull_context(backend, workgroup_size, points, n_points)
    segment_mem_data_float = create_scan_primitive_context(backend, Float64, Int64, workgroup_size, n_points)
    segment_mem_data_int   = create_scan_primitive_context(backend, Int64,   Int64, workgroup_size, n_points)

    gpu_pts = KernelAbstractions.zeros(backend, Float64, (dim, n_points))
    copyto!(gpu_pts, points)

    # ── Init ──────────────────────────────────────────────────────────────────
    result = QhullResult(zeros(Float64, dim, 0), Int64[], Vector{Float64}[])
    mesh   = QuickhullData(Vector{Int}[], Vector{Int}[], Vector{Float64}[], Float64[], Bool[])
    compute_simplex(context, result, mesh, gpu_pts, dim)

    original_ids_cpu = collect(1:n_points)
    original_ids = KernelAbstractions.allocate(backend, Int64, n_points)
    copyto!(original_ids, original_ids_cpu)

    # ── Normals GPU : mesure allocation initiale (hors boucle, informatif) ───
    n_active = length(mesh.active)
    active_normals = hcat(mesh.normals...)
    active_offsets = vcat(mesh.offsets...)

    face_normals_gpu = KernelAbstractions.allocate(backend, Float64, (dim, n_active))
    face_offsets_gpu = KernelAbstractions.allocate(backend, Float64, n_active)
    copyto!(face_normals_gpu, active_normals)
    copyto!(face_offsets_gpu, active_offsets)

    face_flags_gpu   = KernelAbstractions.allocate(backend, Int64, n_points)
    compact_flags_gpu = KernelAbstractions.allocate(backend, Int64, n_points)

    # ── Assign initial ────────────────────────────────────────────────────────
    @gpu_time(prof, :kernel_assign,
        assign_face_to_point_kernel(backend, workgroup_size)(
            face_flags_gpu, compact_flags_gpu, gpu_pts,
            face_normals_gpu, face_offsets_gpu, dim, ndrange=n_points))

    # ── Compact initial ───────────────────────────────────────────────────────
    @gpu_time(prof, :kernel_compact, begin
        restant, original_ids, cp, cflags, cn = compact(
            context, segment_mem_data_int, compact_flags_gpu,
            gpu_pts, original_ids, n_points, dim)
    end)

    rest_segment        = KernelAbstractions.zeros(backend, Int64, cn)
    compacted_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))

    @gpu_time(prof, :kernel_compact, begin
        permute_indices_kernel(backend, workgroup_size)(
            compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n_points)
        KernelAbstractions.synchronize(backend)
    end)

    @gpu_time(prof, :kernel_sort,
        out_perm = flag_permute(context, compacted_flags_gpu, rest_segment))

    restant_new       = KernelAbstractions.zeros(backend, Float64, (dim, size(restant, 2)))
    ids_new           = KernelAbstractions.zeros(backend, Int64, size(restant, 2))
    point_to_face_flags = KernelAbstractions.zeros(backend, Int64, size(restant, 2))

    @gpu_time(prof, :kernel_compact, begin
        permute_points_kernel(backend, workgroup_size)(restant_new, restant, out_perm, dim, ndrange=size(restant, 2))
        permute_kernel(backend, workgroup_size)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))
        permute_kernel(backend, workgroup_size)(point_to_face_flags, compacted_flags_gpu, out_perm, ndrange=size(restant, 2))
        KernelAbstractions.synchronize(backend)
    end)

    restant      = restant_new
    original_ids = ids_new

    # ── Boucle principale ─────────────────────────────────────────────────────
    while size(restant, 2) > 0
        n = size(restant, 2)
        n_segs = AcceleratedKernels.reduce((x,y)->x+y, rest_segment; init=0, neutral=0, block_size=workgroup_size)
        n_segs == 0 && break
        prof.n_iters += 1

        # 1. Distance à la face
        distances = nothing
        @gpu_time(prof, :gpu_alloc,
            distances = KernelAbstractions.allocate(backend, Float64, n))

        @gpu_time(prof, :kernel_distance,
            distance_to_face_kernel(backend, workgroup_size)(
                distances, restant, point_to_face_flags,
                face_normals_gpu, face_offsets_gpu, dim, ndrange=n))

        # 2. Propagation du max par segment (segmented scan)
        prefix_max = nothing
        seg_max    = nothing
        @gpu_time(prof, :scan_primitive, begin
            prefix_max = segmented_scan(segment_mem_data_float, distances, rest_segment,
                ScanPrimitive.MaxOp(), backward=false, inclusive=true, identity=typemin(Float64))
            seg_max    = segmented_scan(segment_mem_data_float, prefix_max, rest_segment,
                ScanPrimitive.MaxOp(), backward=true,  inclusive=true, identity=typemin(Float64))
        end)

        # 3. Candidats les plus lointains
        cand_idx    = nothing
        compact_mask = nothing
        far_idx     = nothing
        far_dist    = nothing

        @gpu_time(prof, :gpu_alloc, begin
            cand_idx     = KernelAbstractions.allocate(backend, Int64, n)
            compact_mask = KernelAbstractions.allocate(backend, Int64, n)
        end)

        @gpu_time(prof, :kernel_compact, begin
            mark_farthest_candidate_kernel(backend, workgroup_size)(cand_idx, distances, seg_max, ndrange=n)
            KernelAbstractions.synchronize(backend)
            create_flag_mask(backend, workgroup_size)(compact_mask, cand_idx, ndrange=tuple(n))
            far_idx_gpu  = compact(context, segment_mem_data_int, compact_mask, cand_idx,    original_ids, n, 1, compact_only_data=true)
            far_dist_gpu = compact(context, segment_mem_data_int, compact_mask, seg_max,     original_ids, n, 1, compact_only_data=true)
        end)

        # 4. Transfert PCIe GPU→CPU  ← souvent le plus lent sur petits N
        @gpu_time(prof, :pcie_transfer, begin
            far_idx  = Array(far_idx_gpu)
            far_dist = Array(far_dist_gpu)
        end)

        active_face_indices = findall(mesh.active)

        # 5. Tri candidats + détection conflits (CPU pur)
        candidates = nothing
        points_to_insert = nothing
        @cpu_time(prof, :cpu_candidates, begin
            candidates = NamedTuple{(:dist, :local_idx, :seg_idx), Tuple{Float64, Int64, Int64}}[]
            for i in 1:n_segs
                if far_idx[i] != 0 && far_dist[i] > EPSILON
                    push!(candidates, (dist=far_dist[i], local_idx=far_idx[i], seg_idx=i))
                end
            end
            sort!(candidates, by=x->x.dist, rev=true)

            to_remove_faces  = Set{Int}()
            points_to_insert = Tuple{Int64, Int64}[]
            cand_local_indices = [cand.local_idx for cand in candidates]
            cand_global_ids    = Array(@view original_ids[cand_local_indices])
            cand_faces         = Array(@view point_to_face_flags[cand_local_indices])

            for (i, cand) in enumerate(candidates)
                global_point_idx = cand_global_ids[i]
                global_face_id   = active_face_indices[cand_faces[i]]
                global_face_id in to_remove_faces && continue

                vis = get_visible_faces(context, mesh, global_point_idx, global_face_id, gpu_pts)
                conflict = any(f in to_remove_faces for f in vis)
                if !conflict
                    push!(points_to_insert, (global_point_idx, global_face_id))
                    union!(to_remove_faces, vis)
                end
            end
        end)

        # 6. BFS + mise à jour du mesh (CPU pur, dominant sur petits N)
        @cpu_time(prof, :cpu_mesh_update, begin
            for (point_idx, face_id) in points_to_insert
                push!(result.convex_hull_bounds, gpu_pts[:, point_idx])
                push!(result.hull_indices, point_idx)
                insert_point_and_update_mesh(context, mesh, point_idx, face_id, gpu_pts, dim)
            end
        end)

        # 7. Reconstruire les normals sur GPU
        n_active_new = sum(mesh.active)
        active_normals_new = zeros(Float64, dim, n_active_new)
        active_offsets_new = zeros(Float64, n_active_new)
        idx = 1
        for i in 1:length(mesh.active)
            if mesh.active[i]
                active_normals_new[:, idx] = mesh.normals[i]
                active_offsets_new[idx]    = mesh.offsets[i]
                idx += 1
            end
        end

        @gpu_time(prof, :gpu_alloc, begin
            face_normals_gpu = KernelAbstractions.allocate(backend, Float64, (dim, n_active_new))
            face_offsets_gpu = KernelAbstractions.allocate(backend, Float64, n_active_new)
        end)
        copyto!(face_normals_gpu, active_normals_new)
        copyto!(face_offsets_gpu, active_offsets_new)

        # 8. Assign + compact + sort (réassigner les points restants)
        @gpu_time(prof, :gpu_alloc, begin
            face_flags_gpu    = KernelAbstractions.allocate(backend, Int64, n)
            compact_flags_gpu = KernelAbstractions.allocate(backend, Int64, n)
        end)

        @gpu_time(prof, :kernel_assign,
            assign_face_to_point_kernel(backend, workgroup_size)(
                face_flags_gpu, compact_flags_gpu, restant,
                face_normals_gpu, face_offsets_gpu, dim, ndrange=n))

        @gpu_time(prof, :kernel_compact, begin
            restant, original_ids, cp, cflags, cn = compact(
                context, segment_mem_data_int, compact_flags_gpu, restant, original_ids, n, dim)
            length(restant) == 0 && break
        end)

        rest_segment = KernelAbstractions.allocate(backend, Int64, cn)

        @gpu_time(prof, :kernel_compact, begin
            compacted_flags_gpu = KernelAbstractions.allocate(backend, Int64, size(restant, 2))
            permute_indices_kernel(backend, workgroup_size)(
                compacted_flags_gpu, face_flags_gpu, cflags, cp, ndrange=n)
            KernelAbstractions.synchronize(backend)
        end)

        @gpu_time(prof, :kernel_sort,
            out_perm = flag_permute(context, compacted_flags_gpu, rest_segment))

        new_restant = KernelAbstractions.zeros(backend, Float64, (dim, size(restant, 2)))
        ids_new     = KernelAbstractions.allocate(backend, Int64, size(restant, 2))

        @gpu_time(prof, :kernel_compact, begin
            permute_points_kernel(backend, workgroup_size)(new_restant, restant, out_perm, dim, ndrange=size(restant, 2))
            permute_kernel(backend, workgroup_size)(ids_new, original_ids, out_perm, ndrange=size(restant, 2))
            point_to_face_flags = KernelAbstractions.zeros(backend, Int64, size(restant, 2))
            permute_kernel(backend, workgroup_size)(point_to_face_flags, compacted_flags_gpu, out_perm, ndrange=size(restant, 2))
            KernelAbstractions.synchronize(backend)
        end)

        restant      = new_restant
        original_ids = ids_new
    end

    return prepare_output(context, result)
end


# =============================================================================
#  SCRIPT DE BENCHMARK — génère les données pour les figures du mémoire
# =============================================================================

"""
    run_benchmark(backend; dims=[2,3,4], ns=nothing, repeats=3)

Lance le benchmark complet et affiche les résultats.
Exemple :
    run_benchmark(CUDABackend())
"""
function run_benchmark(backend; dims=[2, 3, 4],
                       ns = [10, 100, 1_000, 10_000, 100_000, 1_000_000],
                       repeats = 3)

    println("\n" * "█"^60)
    println("  GPUQuickhull — Benchmark complet")
    println("█"^60)

    results = Dict()

    for d in dims
        println("\n── Dimension $d ──────────────────────────────────────────")
        for n in ns
            times_per_cat = Dict{Symbol, Vector{Float64}}()
            total_times   = Float64[]

            for rep in 1:repeats
                pts  = randn(d, n)
                prof = Profiler()
                t0   = time_ns()
                quick_hull_profiled(backend, pts, prof)
                t1   = time_ns()

                push!(total_times, (t1-t0)*1e-6)  # ms
                for (cat, t) in prof.times
                    v = get!(times_per_cat, cat, Float64[])
                    push!(v, t*1000)  # ms
                end
            end

            # Médiane sur les répétitions
            med_total  = median(total_times)
            med_cats   = Dict(k => median(v) for (k,v) in times_per_cat)
            results[(d, n)] = (total=med_total, cats=med_cats)

            @printf("  N=%8d  total=%8.2f ms", n, med_total)

            # Top 2 catégories
            top2 = sort(collect(med_cats), by=x->x[2], rev=true)[1:min(2,end)]
            for (cat, t) in top2
                pct = 100*t/med_total
                print("  | $(rpad(string(cat), 20)) $(lpad(round(pct,digits=1), 5))%")
            end
            println()
        end
    end

    return results
end
=#