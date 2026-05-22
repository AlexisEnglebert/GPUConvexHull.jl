module ScanPrimitive
using KernelAbstractions

export segmented_scan, create_scan_primitive_context

struct MinOp end
struct MaxOp end
struct AddOp end

(::MinOp)(a, b) = a < b ? a : b
(::MaxOp)(a, b) = a > b ? a : b
(::AddOp)(a, b) = a + b

struct ScanPrimitiveContext{T, F, B}
    backend::B
    workgroup_size::Int64
end

function create_scan_primitive_context(backend, val_type, flag_type, workgroup_size, n)
    if !ispow2(workgroup_size)
        error("Workgroup size must be a power of two !")
    end 
    return ScanPrimitiveContext{val_type, flag_type, typeof(backend)}(backend, workgroup_size)
end

@kernel function scan_inner_block_downsweep!(out, partial_values, offset_block, size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T = 0) where {Op, T, TREE_SIZE}
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))

    temp = @localmem(eltype(partial_values), TREE_SIZE)

    for i in thread_id:group_size:TREE_SIZE
        temp[i] = identity
    end
    @synchronize()

    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id

    if (2*global_id-1) <= size
        temp[idx1] = partial_values[2*global_id-1]
    end

    if (2*global_id) <= size
        temp[idx2] = partial_values[2*global_id]
    end

    @synchronize()
     if thread_id == 1
         temp[TREE_SIZE] = offset_block[block_id]
     end

    for shift in (tree_power-1):-1:0
        offset = 1 << shift
        if thread_id <= (TREE_SIZE >> (shift + 1))
            t = temp[offset*(thread_id*2)-offset]
            temp[offset*(thread_id*2)-offset] = temp[offset*(thread_id*2)]
            temp[offset*(thread_id*2)] = oplus(temp[offset*(thread_id*2)], t)
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

@kernel function scan_inner_block_upsweep!(partial_values, block_values, values, size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T = 0) where {Op, T, TREE_SIZE}
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))

    temp = @localmem(eltype(values), TREE_SIZE)

    for i in thread_id:group_size:TREE_SIZE
        temp[i] = identity
    end
    @synchronize()

    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id
    if (2*global_id-1) <= size
        temp[idx1] = values[2*global_id-1]
    end
    if (2*global_id) <= size
        temp[idx2] = values[2*global_id]
    end

    @synchronize()

    for shift = 0:(tree_power - 1)
        offset = 1 << shift
        if thread_id <= TREE_SIZE >> (shift+1)
            temp[offset*(thread_id*2)] = oplus(temp[offset*(thread_id*2)], temp[offset*(thread_id*2)-offset])
        end
        @synchronize()
    end

    if ((2*global_id)-1 <= size)
        partial_values[(2*global_id)-1] = temp[(2*thread_id)-1]
    end

    if ((2*global_id) <= size)
        partial_values[(2*global_id)] = temp[(2*thread_id)]
    end

    if thread_id == 1
        block_values[block_id] = temp[TREE_SIZE]
    end
end

@kernel function scan_second_level_kernell!(block_values, size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T) where {Op, T, TREE_SIZE}
    thread_id = @index(Local)

    @uniform tree_power = ceil(Int64, log2(TREE_SIZE))

    temp = @localmem(T, TREE_SIZE)

    idx1 = 2 * thread_id - 1
    idx2 = 2 * thread_id

    temp[idx1] = identity
    temp[idx2] = identity

    if idx1 <= size
        temp[idx1] = block_values[idx1]
    end
    if idx2 <= size
        temp[idx2] = block_values[idx2]
    end

    @synchronize()

    for shift = 0:(tree_power - 1)
        offset = 1 << shift
        if thread_id <= (TREE_SIZE >> (shift + 1))
            idx = offset * (thread_id * 2)
            temp[idx] = oplus(temp[idx], temp[idx - offset])
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
            temp[idx] = oplus(temp[idx], t)
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

@kernel function segmented_scan_inner_block_downsweep!(out, partial_values, partial_flags, offset_block, in_flags,
    size::Integer, oplus::Op, ::Val{TREE_SIZE}, identity::T = 0) where {Op, T, TREE_SIZE}

    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))

    temp = @localmem(eltype(partial_values), TREE_SIZE)
    input_f = @localmem(eltype(in_flags), TREE_SIZE)
    f = @localmem(eltype(in_flags), TREE_SIZE)

    for i in thread_id:group_size:TREE_SIZE
        temp[i] = identity
        f[i] = zero(eltype(in_flags))
        input_f[i] = zero(eltype(in_flags))
    end
    @synchronize()

    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id

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
            t = temp[offset*(thread_id*2)-offset]
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

    temp = @localmem(eltype(values), TREE_SIZE)
    input_f = @localmem(eltype(flags), TREE_SIZE)
    f = @localmem(eltype(flags), TREE_SIZE)

    for i in thread_id:group_size:TREE_SIZE
        temp[i] = identity
        f[i] = zero(eltype(flags))
        input_f[i] = zero(eltype(flags))
    end
    @synchronize()

    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id

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
    if i <= n 
        j = mod1(i - shift, n) 
        output[i] = input[j]
    end
end

@kernel function inclusive_kernell!(array, inital_value, oplus)
    global_id = @index(Global)
    n = length(array)
    if global_id <= n
        array[global_id] = oplus(array[global_id], inital_value[global_id])
    end
end

@kernel function spread_head_flags!(dst, src, stride)
    global_id = @index(Global)
    if global_id <= length(dst)
        src_idx = (global_id - 1) * stride + 1
        if src_idx <= length(src)
            dst[global_id] = src[src_idx]
        end
    end
end

function scan(context::ScanPrimitiveContext{T, F}, values, oplus::Op;
    backward=false, inclusive=false, identity::T = zero(T)) where{T, F, Op}
    return segmented_scan(context, values, nothing, oplus; backward=backward, inclusive=inclusive, identity=identity, segmented=false)
end

function segmented_scan(context::ScanPrimitiveContext{T, F}, values, flags, oplus::Op;
    backward=false, inclusive=false, identity::T = zero(T), segmented=true) where{T, F, Op}

    if eltype(values) != typeof(identity)
        error("Identity type must be the same as values type. Got:" , eltype(values), " instead of: ", typeof(identity))
        return -1
    end

    values_gpu = values

    if isa(values, Vector) && KernelAbstractions.get_backend(values) != context.backend
        @warn "Got a non GPU vector as values, converting it to GPU (slower). "
        values_gpu = KernelAbstractions.allocate(context.backend, eltype(values), length(values))
        copyto!(values_gpu, values)
    end

    n = length(values_gpu)

    local flags_gpu = flags
    if segmented
        if isa(flags, Vector) && KernelAbstractions.get_backend(flags) != context.backend
            @warn "Got a non GPU vector in flags, converting it to GPU array (slower)."
            flags_gpu = KernelAbstractions.allocate(context.backend, F, length(flags))
            copyto!(flags_gpu, flags)
        end
    end

    local tmp_flags
    if backward
        if segmented
            tmp_flags = KernelAbstractions.zeros(context.backend, F, n)
        end
        reverse_kernel!(context.backend, context.workgroup_size)(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)
        if segmented
            reverse_kernel!(context.backend, context.workgroup_size)(flags_gpu, n, ndrange = size(flags_gpu))
            copyto!(tmp_flags, 1, flags_gpu, 1, n)
            KernelAbstractions.synchronize(context.backend)

            circshift_kernel(context.backend, context.workgroup_size)(flags_gpu, tmp_flags, 1, ndrange=length(flags_gpu))
            KernelAbstractions.synchronize(context.backend)
        end
    end

    final_array = KernelAbstractions.allocate(context.backend, eltype(values_gpu), Int(length(values_gpu)))
    tree_size = 1 << ceil(Int, log2(context.workgroup_size * 2))
    elements_per_block = 2 * context.workgroup_size

    layer_size = [n]
    while layer_size[end] > elements_per_block
        push!(layer_size, cld(layer_size[end], elements_per_block))
    end
    if length(layer_size) == 1
        push!(layer_size, 1)
    end

    top_size = layer_size[end]
    top_level_tree_size = max(2, nextpow(2, top_size))

    pyramid_partial_values        = [KernelAbstractions.allocate(context.backend, eltype(values_gpu), Int(size)) for size in layer_size[1:end-1]]
    pyramid_blocks_last_value     = [KernelAbstractions.allocate(context.backend, eltype(values_gpu), Int(size)) for size in layer_size[2:end]]

    local pyramid_partial_flags, pyramid_blocks_last_flag, pyramid_blocks_last_tree_flag
    if segmented
        pyramid_partial_flags         = [KernelAbstractions.zeros(context.backend, F,  Int(size)) for size in layer_size[1:end-1]]
        pyramid_blocks_last_flag      = [KernelAbstractions.zeros(context.backend, F,  Int(size))  for size in layer_size[2:end]]
        pyramid_blocks_last_tree_flag = [KernelAbstractions.zeros(context.backend, F,  Int(size))  for size in layer_size[2:end]]
    end

    n_layers = length(layer_size) - 1
    for level in 1:n_layers
        fill!(pyramid_partial_values[level], identity)
        fill!(pyramid_blocks_last_value[level], identity)
    end

    current_input_values = values_gpu
    local current_input_flags
    if segmented
        current_input_flags = flags_gpu
    end

    for level in 1:n_layers
        current_layer_size = layer_size[level]
        nb_block = cld(current_layer_size, elements_per_block)
        actual_data_size = (level == 1) ? n : current_layer_size
        
        if segmented
            segmented_scan_inner_block_upsweep!(context.backend, context.workgroup_size)(pyramid_partial_values[level], pyramid_partial_flags[level],
            pyramid_blocks_last_value[level], pyramid_blocks_last_tree_flag[level],
            pyramid_blocks_last_flag[level], current_input_values, actual_data_size,
            current_input_flags, oplus, Val(tree_size), identity, ndrange = tuple(nb_block * context.workgroup_size))
        else
            scan_inner_block_upsweep!(context.backend, context.workgroup_size)(pyramid_partial_values[level],
            pyramid_blocks_last_value[level], current_input_values, actual_data_size,
            oplus, Val(tree_size), identity, ndrange = tuple(nb_block * context.workgroup_size))
        end

        KernelAbstractions.synchronize(context.backend)

        if segmented && level > 1
            spread_head_flags!(context.backend, context.workgroup_size)(pyramid_blocks_last_flag[level], pyramid_blocks_last_flag[level-1], elements_per_block, ndrange = nb_block)
            KernelAbstractions.synchronize(context.backend)
        end

        current_input_values = pyramid_blocks_last_value[level]
        if segmented
            current_input_flags = pyramid_blocks_last_tree_flag[level]
        end
    end

    if segmented
        segmented_scan_second_level_kernell!(context.backend, context.workgroup_size)(pyramid_blocks_last_value[n_layers], pyramid_blocks_last_flag[n_layers],
        pyramid_blocks_last_tree_flag[n_layers], layer_size[end], oplus, Val(top_level_tree_size), identity, ndrange = tuple(top_level_tree_size ÷ 2))
    else
        scan_second_level_kernell!(context.backend, context.workgroup_size)(pyramid_blocks_last_value[n_layers], layer_size[end], oplus, Val(top_level_tree_size), identity, ndrange = tuple(top_level_tree_size ÷ 2))
    end
    KernelAbstractions.synchronize(context.backend)

    for level in n_layers:-1:1
        val_output = final_array
        val_size = (level == 1) ? n : layer_size[level]
        local val_flags
        if segmented
            val_flags = flags_gpu
        end

        if level != 1
            val_output = pyramid_blocks_last_value[level-1]
            val_size  = layer_size[level]
            if segmented
                val_flags = pyramid_blocks_last_flag[level-1]
            end
        end

        nb_blocks = cld(val_size, elements_per_block)

        if segmented
            segmented_scan_inner_block_downsweep!(context.backend, context.workgroup_size)(val_output, pyramid_partial_values[level], pyramid_partial_flags[level],
            pyramid_blocks_last_value[level], val_flags, val_size, oplus, Val(tree_size), identity,
            ndrange = tuple(nb_blocks * context.workgroup_size))
        else
            scan_inner_block_downsweep!(context.backend, context.workgroup_size)(val_output, pyramid_partial_values[level],
            pyramid_blocks_last_value[level], val_size, oplus, Val(tree_size), identity,
            ndrange = tuple(nb_blocks * context.workgroup_size))
        end

        KernelAbstractions.synchronize(context.backend)
    end

    if inclusive
        inclusive_kernell!(context.backend, context.workgroup_size)(final_array, values_gpu, oplus, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)
    end

    if backward
        reverse_kernel!(context.backend, context.workgroup_size)(values_gpu, n, ndrange = size(values_gpu))
        KernelAbstractions.synchronize(context.backend)

        if segmented
            reverse_kernel!(context.backend, context.workgroup_size)(tmp_flags, n, ndrange = n)
            KernelAbstractions.synchronize(context.backend)
        end
        
        reverse_kernel!(context.backend, context.workgroup_size)(final_array, n, ndrange = n)
        KernelAbstractions.synchronize(context.backend)

        if segmented
            copyto!(flags_gpu, 1, tmp_flags, 1, n)
        end
    end

    return final_array

end

end