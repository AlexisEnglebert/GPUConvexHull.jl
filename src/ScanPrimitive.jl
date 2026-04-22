module ScanPrimitive
using KernelAbstractions

export segmented_scan, allocate_scan_memory_arrays

# On définit ici les différents type d'opération qu'on peut faire avec le segmented-scan.
# On les définit comme des struct car OneAPI n'aime pas les "types anonyme" et donc ici notre struct a un type bien précis ...

struct MinOp end
struct MaxOp end
struct AddOp end

(::MinOp)(a, b) = a < b ? a : b
(::MaxOp)(a, b) = a > b ? a : b
(::AddOp)(a, b) = a + b

#TODO: remove mutable.
struct ScanMemoryArrays{T, F, B}
    partial_values::T
    partial_flags::F
    blocks_last_value::T
    blocks_last_flag::F
    blocks_last_tree_flag::F
    tmp_flags::F
    backend::B
    nb_block::Int
end

"""
allocate_scan_memory_arrays(backend, val_type, flag_type, workgroup_size, n)

workgroup_size: size of each workgroup, n: the maximal size of the input array.

Allocate memory arrays for the segmented scan operation.
Returns a `ScanMemoryArrays` struct containing all the allocated arrays.
"""
function allocate_scan_memory_arrays(backend, val_type, flag_type, workgroup_size, n)
    nb_block = cld(n, 2*workgroup_size)

    partial_values        = KernelAbstractions.allocate(backend, val_type, Int(n))
    partial_flags         = KernelAbstractions.zeros(backend, flag_type,  Int(n))

    blocks_last_value     = KernelAbstractions.allocate(backend, val_type, Int(nb_block))
    blocks_last_flag      = KernelAbstractions.zeros(backend, flag_type,  Int(nb_block))
    blocks_last_tree_flag = KernelAbstractions.zeros(backend, flag_type,  Int(nb_block))

    tmp_flags             = KernelAbstractions.zeros(backend, flag_type,  Int(n))

    return ScanMemoryArrays(partial_values, partial_flags, blocks_last_value,
     blocks_last_flag, blocks_last_tree_flag, tmp_flags, backend, nb_block)
end

# Hmm à voir si on peut pas utiliser des grid en 3 dim pour rendre le calcul plus vite ?
# TODO: Regarder la nouvelle méthode "LightScan" qui permetterais de compute les scan primitives plus rappidement.
# TODO: Voir comment CUB fait car il fait d'une manière différente ...
# TODO: être consitant dans le nom des choses !!!
# TODO: move toutes les primitives de scan dans un autre fichier car là ce n'est plus tenable
# TODO: Faire en sorte d'utiliser l'opérateur d'identité du oplus à la place de 0 pour l'instant...
@kernel function segmented_scan_inner_block_downsweep!(out, partial_values, partial_flags, offset_block, in_flags,
    size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T = 0) where {Op, T, TREE_SIZE}

global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))

     # Load all partial array in shared memory
    temp = @localmem(eltype(partial_values), TREE_SIZE)
    input_f = @localmem(eltype(in_flags), TREE_SIZE)
    f = @localmem(eltype(in_flags), TREE_SIZE)

    # Init tout à 0 comme ça on évite de faire une branche pour le padding:))
    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id

    temp[idx1] = identity; f[idx1] = 0; input_f[idx1] = 0
    temp[idx2] = identity; f[idx2] = 0; input_f[idx2] = 0


    if (2*global_id-1) <= size
        temp[idx1] = partial_values[2*global_id-1]
        f[idx1] = partial_flags[2*global_id-1]
        input_f[idx1] = in_flags[2*global_id-1]
    end

    if (2*global_id) <= size
        temp[idx2] = partial_values[2*global_id]
        f[idx2] = partial_flags[2*global_id]
        input_f[idx2] = in_flags[2*global_id]
    end

    @synchronize()
     if thread_id == 1
         temp[TREE_SIZE] = offset_block[block_id]
     end

    for shift in (tree_power-1):-1:0
        offset = 1 << shift
        if thread_id <= (TREE_SIZE >> (shift + 1))
            t = temp[offset*(thread_id*2)-offset] # Partie de gauche
            temp[offset*(thread_id*2)-offset] = temp[offset*(thread_id*2)]

            if input_f[offset*(thread_id*2)-offset+1] == 1
                temp[offset*(thread_id*2)] = identity
            elseif f[offset*(thread_id*2)-offset] == 1
                temp[offset*(thread_id*2)] = t
            else
                temp[offset*(thread_id*2)] = oplus(temp[offset*(thread_id*2)], t)
            end
            f[offset*(thread_id*2)-offset] = 0
        end
        @synchronize()
    end

    @synchronize()
    if ((2*global_id)-1 <= size)
        out[2*global_id-1] = temp[(2*thread_id)-1]
    end

    if ((2*global_id) <= size)
        out[2*global_id] = temp[(2*thread_id)]
    end
end

@kernel function segmented_scan_inner_block_upsweep!(partial_values,
    partial_flags, block_values, block_tree_flags, block_flags, values, size::Integer, flags,
    oplus::Op, ::Val{TREE_SIZE}, identity::T = 0) where {Op, T, TREE_SIZE}

    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))

    # Each thread treats 2 values so in shared block memory there is 2 * block_size elements
    temp = @localmem(eltype(values), TREE_SIZE)
    input_f = @localmem(eltype(flags), TREE_SIZE)
    f = @localmem(eltype(flags), TREE_SIZE)

    # Init tout à 0 comme ça on évite de faire une branche pour le padding:))
    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id
    temp[idx1] = identity; f[idx1] = 0; input_f[idx1] = 0
    temp[idx2] = identity; f[idx2] = 0; input_f[idx2] = 0

    if (2*global_id-1) <= size
        temp[idx1] = values[2*global_id-1]
        f[idx1] = flags[2*global_id-1]
        input_f[idx1] = flags[2*global_id-1]
    end
    if (2*global_id) <= size
        temp[idx2] = values[2*global_id]
        f[idx2] = flags[2*global_id]
        input_f[idx2] = flags[2*global_id]
    end

    @synchronize()

    for shift = 0:(tree_power - 1)
        offset = 1 << shift
        if thread_id <= TREE_SIZE >> (shift+1)
            if f[offset*(thread_id*2)] == 0
                temp[offset*(thread_id*2)] = oplus(temp[offset*(thread_id*2)], temp[offset*(thread_id*2)-offset])
            end
            f[offset*(thread_id*2)] |= f[offset*(thread_id*2)-offset]
        end
        @synchronize()
    end

    if ((2*global_id)-1 <= size)
        partial_values[(2*global_id)-1] = temp[(2*thread_id)-1]
        partial_flags[(2*global_id)-1] = f[(2*thread_id)-1]
    end

    if ((2*global_id) <= size)
        partial_values[(2*global_id)] = temp[(2*thread_id)]
        partial_flags[(2*global_id)] = f[(2*thread_id)]
    end

    if thread_id == 1
        block_values[block_id] = temp[TREE_SIZE]
        block_tree_flags[block_id] = f[TREE_SIZE]
        block_flags[block_id] = input_f[1]
    end

end

#TODO: move sur GPU comme le minmaxReduce.
function segmented_scan_second_level_cpu!(block_values, size::Integer, flags, tree_flags, oplus::Op, identity::T = 0) where {Op, T}

    #TODO: ENLEVER CE TRUC DE CON LÀ ET TOUT GARDER SUR GPU.
    cpu_block_values = Array(block_values)
    cpu_flags        = Array(flags)
    cpu_tree_flags   = Array(tree_flags)

    # Values doit être une puissance de 2
    m_pow = ceil(Int, log2(size))
    n = 1 << m_pow
    temp = fill(identity, n)
    f    = fill(zero(eltype(cpu_tree_flags)), n)
    fi   = fill(one(eltype(cpu_flags)), n)

    for i = 1:size
        temp[i] = cpu_block_values[i]
        f[i]    = cpu_tree_flags[i]
        fi[i]   = cpu_flags[i]
    end


    # Upsweep :)
    for d = 0:(m_pow-1)
        offset = 1 << d
        for k = 0:2*offset:n
            if k+2*offset <= n
                if f[k+2*offset] == 0 # 2^ d+1 alors que k+offset = 2^d
                    temp[k+2*offset] = oplus(temp[k+offset], temp[k+2*offset])
                end
                f[k+2*offset] = f[k+offset] | f[k+2*offset]
            end
        end
    end

    # Downsweep
    temp[n] = identity

    for d = (m_pow-1):-1:0
        offset = 1 << d
        for k = 0:2*offset:n
            if k+2*offset <= n
                t = temp[k+offset]
                temp[k+offset] = temp[k+2*offset]

                if fi[k+offset+1] == 1
                    temp[k+2*offset] = identity
                elseif f[k+offset] == 1
                    temp[k+2*offset] = t
                else
                    temp[k+2*offset] = oplus(temp[k+2*offset], t)
                end
                # Update f
                f[k+offset] = 0
            end
        end
    end

    result = temp[1:size]
    gpu_result = KernelAbstractions.allocate(get_backend(block_values), eltype(block_values), size)
    copyto!(gpu_result, result)
    return gpu_result
end

@kernel function reverse_kernel!(a, @uniform actual_length)
    i = @index(Global)
    if i <= actual_length ÷ 2
        j = actual_length - i + 1
        t = a[i]
        a[i] = a[j]
        a[j] = t
    end
end

@kernel function circshift_kernel(output, input, shift)
    i = @index(Global)
    n = length(input)
    j = mod1(i - shift, n)  # mod1 pour rester dans [1, n]
    output[i] = input[j]
end

function gpu_circshift!(arr, shift, backend)
    tmp = copy(arr)
    circshift_kernel(backend, 16)(arr, tmp, shift, ndrange=length(arr))
    KernelAbstractions.synchronize(backend)
end

@kernel function inclusive_kernell!(array, inital_value, oplus)
    global_id = @index(Global)
    n = length(array)
    if global_id <= n
        array[global_id] = oplus(array[global_id], inital_value[global_id])
    end
end

"""
segmented_scan(backend, values, flags, oplus::Function; backward=false, inclusive=false, identity::Number=0)

Performs a segmented scan, returns the segmented scan array.

# Examples
```jldoctest
julia> segmented_scan()
```
"""

function segmented_scan(backend, mem_arrays::ScanMemoryArrays, values, flags, oplus::Op; backward=false, inclusive=false, identity::Number=0) where {Op}
    if eltype(values) != typeof(identity)
        error("Identity type must be the same as values type. Got ")
        return -1
    end

    values_gpu = values;
    flags_gpu = flags

    if isa(values, Vector)
        #@warn "Got a non GPU vector as values, converting it to GPU (slower). "
        values_gpu = KernelAbstractions.allocate(backend, eltype(values), length(values))
        copyto!(values_gpu, values)
    end

    if isa(flags, Vector)
        #@warn "Got a non GPU vector in flags, converting it to GPU array (slower)."
        flags_gpu = KernelAbstractions.allocate(backend, eltype(flags), length(flags))
        copyto!(flags_gpu, flags)
    end

    n = length(values_gpu)
    nb_threads_per_block = 8
    nb_blocks = cld(n, 2*nb_threads_per_block)
    reverse_kernell = reverse_kernel!(backend, nb_threads_per_block)
    #shift_kernell   = shift_right(backend, nb_threads_per_block) TODO: Pas bon ça !!!!!

    tmp_flags = []
    if backward
        reverse_kernell(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(backend)
        reverse_kernell(flags_gpu, n, ndrange = size(flags_gpu))
        copyto!(mem_arrays.tmp_flags, 1, flags_gpu, 1, n)
        KernelAbstractions.synchronize(backend)

        #shift_kernell(tmp_flags, flags, ndrange = size(flags))
        gpu_circshift!(flags_gpu, 1, backend)
        KernelAbstractions.synchronize(backend)
    end

    final_array           = KernelAbstractions.allocate(backend, eltype(values_gpu), Int(length(values_gpu)))

    # TODO: Only one GPU memory allocation for this
    #=partial_values        = KernelAbstractions.allocate(backend, eltype(values_gpu), Int(length(values_gpu)))
    partial_flags         = KernelAbstractions.zeros(backend, eltype(values_gpu),  Int(length(values_gpu)))

    blocks_last_value     = KernelAbstractions.allocate(backend, eltype(values_gpu), Int(nb_blocks))
    blocks_last_flag      = KernelAbstractions.zeros(backend, eltype(flags_gpu),  Int(nb_blocks))
    blocks_last_tree_flag = KernelAbstractions.zeros(backend, eltype(flags_gpu),  Int(nb_blocks))

    copyto!(partial_values,    fill(identity, Int(length(values_gpu))))
    copyto!(blocks_last_value, fill(identity, Int(nb_blocks)))
    copyto!(final_array,       fill(identity, Int(length(values_gpu)))) =#

    fill!(mem_arrays.partial_flags, 0.0)
    fill!(mem_arrays.blocks_last_flag, 0)
    fill!(mem_arrays.blocks_last_tree_flag, 0)

    fill!(mem_arrays.partial_values, identity)
    fill!(mem_arrays.blocks_last_value, identity)
    fill!(final_array, identity)

    upsweep_kernell = segmented_scan_inner_block_upsweep!(backend, nb_threads_per_block)
    downsweep_kernell = segmented_scan_inner_block_downsweep!(backend, nb_threads_per_block)
    inclusive_kernell = inclusive_kernell!(backend, nb_threads_per_block)

    tree_size = 1 << ceil(Int, log2(nb_threads_per_block * 2))

    upsweep_kernell(mem_arrays.partial_values, mem_arrays.partial_flags, mem_arrays.blocks_last_value, mem_arrays.blocks_last_tree_flag, mem_arrays.blocks_last_flag, values_gpu, length(values_gpu),
    flags_gpu, oplus, Val(tree_size), identity, ndrange = tuple(nb_blocks * nb_threads_per_block))
    KernelAbstractions.synchronize(backend)

    segmented_blocks = segmented_scan_second_level_cpu!(
    mem_arrays.blocks_last_value, nb_blocks, mem_arrays.blocks_last_flag, mem_arrays.blocks_last_tree_flag, oplus, identity)

    downsweep_kernell(final_array, mem_arrays.partial_values, mem_arrays.partial_flags, segmented_blocks, flags_gpu,
    n, oplus, Val(tree_size), identity, ndrange = tuple(nb_blocks * nb_threads_per_block))
    KernelAbstractions.synchronize(backend)

    if inclusive
        inclusive_kernell(final_array, values_gpu, oplus, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(backend)
    end

    if backward
        reverse_kernell(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(backend)

        reverse_kernell(mem_arrays.tmp_flags, n, ndrange = n)
        KernelAbstractions.synchronize(backend)
        reverse_kernell(final_array, n, ndrange = n)
        KernelAbstractions.synchronize(backend)

        copyto!(flags_gpu, 1, mem_arrays.tmp_flags, 1, n)
    end

    return final_array
end

end