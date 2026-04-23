module ScanPrimitive
using KernelAbstractions

export segmented_scan, create_scan_primitive_context

# On définit ici les différents type d'opération qu'on peut faire avec le segmented-scan.
# On les définit comme des struct car OneAPI n'aime pas les "types anonyme" et donc ici notre struct a un type bien précis ...

struct MinOp end
struct MaxOp end
struct AddOp end

(::MinOp)(a, b) = a < b ? a : b
(::MaxOp)(a, b) = a > b ? a : b
(::AddOp)(a, b) = a + b

struct ScanPrimitiveContext{T, F, B, U_kernel, D_kernel, R_kernel, I_kernel, C_kernel, S_kernel}
    pyramid_partial_values::Vector{T}
    pyramid_partial_flags::Vector{F}
    pyramid_blocks_last_value::Vector{T}
    pyramid_blocks_last_flag::Vector{F}
    pyramid_blocks_last_tree_flag::Vector{F}

    tmp_flags::F
    backend::B

    workgroup_size::Int64
    top_level_tree_size::Int64
    nb_block::Int64

    # On stock les Kernels car il doivent être compilé qu'une seule fois (Évite de les compiler à chaque appel de segmented_scan & donc beaucoup plus rapide)
    upsweep_kernell::U_kernel
    downsweep_kernell::D_kernel
    reverse_kernell::R_kernel
    inclusive_kernell::I_kernel
    cyclic_kernell::C_kernel
    second_level_kernell::S_kernel

end

"""
create_scan_primitive_context(backend, val_type, flag_type, workgroup_size, n)

workgroup_size: size of each workgroup, n: the maximal size of the input array.

Create a context for the segmented scan operation.
Returns a `ScanPrimitiveContext` struct containing all the allocated arrays.
"""
function create_scan_primitive_context(backend, val_type, flag_type, workgroup_size, n)
    n_per_block = 2*workgroup_size
    nb_block = cld(n, n_per_block)

    # On stock à chaque fois la taille des étages de la pyramide -> On a une pyramide d'arbre haha
    layer_size = [n]
    while layer_size[end] > n_per_block
        push!(layer_size, cld(layer_size[end], n_per_block))
    end

    if length(layer_size) == 1
        push!(layer_size, 1)
    end

    pyramid_partial_values        = [KernelAbstractions.allocate(backend, val_type, Int(size)) for size in layer_size[1:end-1]]
    pyramid_partial_flags         = [KernelAbstractions.zeros(backend, flag_type,  Int(size)) for size in layer_size[1:end-1]]

    pyramid_blocks_last_value     = [KernelAbstractions.allocate(backend, val_type, Int(size)) for size in layer_size[2:end]]
    pyramid_blocks_last_flag      = [KernelAbstractions.zeros(backend, flag_type,  Int(size))  for size in layer_size[2:end]]
    pyramid_blocks_last_tree_flag = [KernelAbstractions.zeros(backend, flag_type,  Int(size))  for size in layer_size[2:end]]

    top_size = layer_size[end]
    top_level_tree_size = max(2, nextpow(2, top_size))

    tmp_flags             = KernelAbstractions.zeros(backend, flag_type,  Int(n))

    upsweep_kernell = segmented_scan_inner_block_upsweep!(backend, workgroup_size)
    downsweep_kernell = segmented_scan_inner_block_downsweep!(backend, workgroup_size)

    reverse_kernell = reverse_kernel!(backend, workgroup_size)
    inclusive_kernell = inclusive_kernell!(backend, workgroup_size)
    cyclic_kernell = circshift_kernel(backend, workgroup_size)
    second_level_kernell = segmented_scan_second_level_kernell!(backend, top_level_tree_size ÷ 2)

    # Mtn on build la pyramide:


    return ScanPrimitiveContext(pyramid_partial_values, pyramid_partial_flags, pyramid_blocks_last_value,
     pyramid_blocks_last_flag, pyramid_blocks_last_tree_flag, tmp_flags, backend, workgroup_size, top_level_tree_size,
     nb_block, upsweep_kernell, downsweep_kernell, reverse_kernell, inclusive_kernell,
     cyclic_kernell, second_level_kernell)
end

# Hmm à voir si on peut pas utiliser des grid en 3 dim pour rendre le calcul plus vite ?
# TODO: Voir comment CUB fait car il fait d'une manière différente ...
# TODO: être consitant dans le nom des choses !!!
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

@kernel function segmented_scan_second_level_kernell!(block_values, block_flags, block_tree_flags, size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T) where {Op, T, TREE_SIZE}
    thread_id = @index(Local)

    @uniform tree_power = ceil(Int64, log2(TREE_SIZE))

    temp = @localmem(T, TREE_SIZE)
    f    = @localmem(eltype(block_tree_flags), TREE_SIZE)
    fi   = @localmem(eltype(block_flags), TREE_SIZE)

    idx1 = 2 * thread_id - 1
    idx2 = 2 * thread_id

    temp[idx1] = identity; f[idx1] = 0; fi[idx1] = 1
    temp[idx2] = identity; f[idx2] = 0; fi[idx2] = 1

    if idx1 <= size
        temp[idx1] = block_values[idx1]
        f[idx1]    = block_tree_flags[idx1]
        fi[idx1]   = block_flags[idx1]
    end
    if idx2 <= size
        temp[idx2] = block_values[idx2]
        f[idx2]    = block_tree_flags[idx2]
        fi[idx2]   = block_flags[idx2]
    end

    @synchronize()

    for shift = 0:(tree_power - 1)
        offset = 1 << shift
        if thread_id <= (TREE_SIZE >> (shift + 1))
            idx = offset * (thread_id * 2)
            if f[idx] == 0
                temp[idx] = oplus(temp[idx], temp[idx - offset])
            end
            f[idx] |= f[idx - offset]
        end
        @synchronize()
    end

    if thread_id == 1
        temp[TREE_SIZE] = identity
    end
    @synchronize()

    for shift = (tree_power - 1):-1:0
        offset = 1 << shift
        if thread_id <= (TREE_SIZE >> (shift + 1))
            idx = offset * (thread_id * 2)
            t = temp[idx - offset]
            temp[idx - offset] = temp[idx]

            if fi[idx - offset + 1] == 1
                temp[idx] = identity
            elseif f[idx - offset] == 1
                temp[idx] = t
            else
                temp[idx] = oplus(temp[idx], t)
            end
            f[idx - offset] = 0
        end
        @synchronize()
    end

    @synchronize()

    if 2 * thread_id - 1 <= size
        block_values[2 * thread_id - 1] = temp[2 * thread_id - 1]
    end
    if 2 * thread_id <= size
        block_values[2 * thread_id] = temp[2 * thread_id]
    end

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

function segmented_scan(context::ScanPrimitiveContext{T, F, B, U_kernel, D_kernel, R_kernel, I_kernel, C_kernel, S_kernel}, values, flags, oplus::Op;
    backward=false, inclusive=false, identity::eltype(T) = zero(eltype(T)) ) where{Op, T, F, B, U_kernel, D_kernel, R_kernel, I_kernel, C_kernel, S_kernel}

    if eltype(values) != typeof(identity)
        error("Identity type must be the same as values type. Got ")
        return -1
    end

    values_gpu = values;
    flags_gpu = flags

    if isa(values, Vector) && KernelAbstractions.get_backend(values) != context.backend
        @warn "Got a non GPU vector as values, converting it to GPU (slower). "
        values_gpu = KernelAbstractions.allocate(context.backend, eltype(values), length(values))
        copyto!(values_gpu, values)
    end

    if isa(flags, Vector) && KernelAbstractions.get_backend(flags) != context.backend
        @warn "Got a non GPU vector in flags, converting it to GPU array (slower)."
        flags_gpu = KernelAbstractions.allocate(context.backend, eltype(flags), length(flags))
        copyto!(flags_gpu, flags)
    end
    n = length(values_gpu)

    tmp_flags = []
    if backward
        context.reverse_kernell(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)
        context.reverse_kernell(flags_gpu, n, ndrange = size(flags_gpu))
        copyto!(context.tmp_flags, 1, flags_gpu, 1, n)
        KernelAbstractions.synchronize(context.backend)


        tmp = copy(flags_gpu)
        context.cyclic_kernell(flags_gpu, tmp, 1, ndrange=length(flags_gpu))
        KernelAbstractions.synchronize(context.backend)
    end

    final_array           = KernelAbstractions.allocate(context.backend, eltype(values_gpu), Int(length(values_gpu)))

    #=fill!(context.partial_flags, 0)
    fill!(context.blocks_last_flag, 0)
    fill!(context.blocks_last_tree_flag, 0)

    fill!(context.partial_values, identity)
    fill!(context.blocks_last_value, identity)
    fill!(final_array, identity) =#


    tree_size = 1 << ceil(Int, log2(context.workgroup_size * 2))

    elements_per_block = 2 * context.workgroup_size
    layer_size = [n]
    while layer_size[end] > elements_per_block
            push!(layer_size, cld(layer_size[end], elements_per_block))
    end
    if length(layer_size) == 1
        push!(layer_size, 1)
    end

    # On procède au nombre d'étages - 1 de upsweep (car le dernier c'est les block finaux)
    n_layers = length(layer_size) - 1
    current_input_values = values_gpu
    current_input_flags = flags_gpu

    for level in 1:n_layers
        current_layer_size = layer_size[level]
        nb_block = cld(current_layer_size, elements_per_block)

        context.upsweep_kernell(context.pyramid_partial_values[level], context.pyramid_partial_flags[level],
        context.pyramid_blocks_last_value[level], context.pyramid_blocks_last_tree_flag[level],
        context.pyramid_blocks_last_flag[level], current_input_values, current_layer_size,
        current_input_flags, oplus, Val(tree_size), identity, ndrange = tuple(nb_block * context.workgroup_size))

        KernelAbstractions.synchronize(context.backend)
        current_input_values = context.pyramid_partial_values[level]
        current_input_flags = context.pyramid_partial_flags[level]
    end


    context.second_level_kernell(context.pyramid_blocks_last_value[n_layers], context.pyramid_blocks_last_flag[n_layers],
    context.pyramid_blocks_last_tree_flag[n_layers], layer_size[end], oplus, Val(context.top_level_tree_size), identity, ndrange = tuple(context.top_level_tree_size ÷ 2))
    KernelAbstractions.synchronize(context.backend)

    # On propage la pyramide du haut vers le bas
    for level in n_layers:-1:1
        val_output = final_array
        val_flags = flags_gpu
        val_size = n
        if level != 1
            val_output = context.pyramid_blocks_last_value[level-1]
            val_flags = context.pyramid_blocks_last_flag[level-1]
            val_size  = layer_size[level]
        end

        nb_blocks = cld(val_size, elements_per_block)

        context.downsweep_kernell(val_output, context.pyramid_partial_values[level], context.pyramid_partial_flags[level],
        context.pyramid_blocks_last_value[level], val_flags, val_size, oplus, Val(tree_size), identity,
        ndrange = tuple(nb_blocks * context.workgroup_size))

        KernelAbstractions.synchronize(context.backend)

    end

    if inclusive
        context.inclusive_kernell(final_array, values_gpu, oplus, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)
    end

    if backward
        context.reverse_kernell(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)

        context.reverse_kernell(context.tmp_flags, n, ndrange = n)
        KernelAbstractions.synchronize(context.backend)
        context.reverse_kernell(final_array, n, ndrange = n)
        KernelAbstractions.synchronize(context.backend)

        copyto!(flags_gpu, 1, context.tmp_flags, 1, n)
    end

    return final_array
end

end