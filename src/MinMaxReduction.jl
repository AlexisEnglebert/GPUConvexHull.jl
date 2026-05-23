module MinMaxReduction
using KernelAbstractions

export min_max_reduce

struct MinMax{T, I} # Type and Index
    min::T
    imin::I
    max::T
    imax::I
end

# On dois définir c'est quoi le zéro de MinMax pour pouvoir gérer ça côté GPU
# HORRIBLE 1H pour trouver ça, même les template en c++ c'est plus simple mdr
Base.zero(::Type{MinMax{T, I}}) where {T, I} = MinMax{T, I}(typemax(T), zero(I), typemin(T), zero(I))

@kernel function min_max_reduce_block_kernel(values, output, @uniform n)
    thread_id = @index(Local)
    block_id  = @index(Group)
    
    global_offset = (block_id - 1) * (2 * first(@groupsize))

    idx1 = global_offset + thread_id
    idx2 = global_offset + thread_id + first(@groupsize)

    smin  = @localmem(eltype(values), first(@groupsize))
    smax  = @localmem(eltype(values), first(@groupsize))
    simin = @localmem(UInt32, first(@groupsize))
    simax = @localmem(UInt32, first(@groupsize))

    if idx1 ≤ n
        val1_min = values[idx1]
        val1_max = values[idx1]
    else
        val1_min = typemax(eltype(values))
        val1_max = typemin(eltype(values))
    end
    
    if idx2 ≤ n
        val2_min = values[idx2]
        val2_max = values[idx2]
    else
        val2_min = typemax(eltype(values))
        val2_max = typemin(eltype(values))
    end
    
    if val1_min < val2_min
        smin[thread_id]  = val1_min
        simin[thread_id] = idx1
    else
        smin[thread_id]  = val2_min
        if idx2 ≤ n
            simin[thread_id] = idx2
        else
            simin[thread_id] = idx1
        end
    end

    if val1_max >= val2_max
        smax[thread_id]  = val1_max
        simax[thread_id] = idx1
    else
        smax[thread_id]  = val2_max
        if idx2 ≤ n 
            simax[thread_id] = idx2
        else
            simax[thread_id] = idx1
        end
    end
    
    @synchronize()

    for steps in 1:ceil(Int64, log2((first(@groupsize()))))
        limit = ceil(Int64, first(@groupsize()) / (2^steps))
        if thread_id ≤ limit 
            if @inbounds smin[thread_id] > smin[thread_id + limit]
                 @inbounds smin[thread_id] = smin[thread_id + limit]
                 @inbounds simin[thread_id] = simin[thread_id + limit]
            end

            if @inbounds smax[thread_id] <= smax[thread_id + limit]
                 @inbounds smax[thread_id] = smax[thread_id + limit]
                 @inbounds simax[thread_id] = simax[thread_id + limit]
            end
        end
        @synchronize()
    end

    if thread_id == 1
        output[block_id] = MinMax{eltype(values), UInt32}(smin[1], simin[1], smax[1], simax[1])
    end
end

@kernel function min_max_reduce_block_kernel_minmax(values, output, @uniform n)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)
    @uniform group_size = first(@groupsize)
    
    smin  = @localmem(eltype(values).parameters[1],  group_size)
    smax  = @localmem(eltype(values).parameters[1],  group_size)
    simin = @localmem(eltype(values).parameters[2], group_size)
    simax = @localmem(eltype(values).parameters[2], group_size)

    if global_id ≤ n
        smin[thread_id]  = values[global_id].min
        smax[thread_id]  = values[global_id].max
        simin[thread_id] = values[global_id].imin
        simax[thread_id] = values[global_id].imax
    else
        smin[thread_id]  = typemax(eltype(values).parameters[1])
        smax[thread_id]  = typemin(eltype(values).parameters[1])
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

            if @inbounds smax[thread_id] <= smax[thread_id + limit]
                 @inbounds smax[thread_id] = smax[thread_id + limit]
                 @inbounds simax[thread_id] = simax[thread_id + limit]
            end
        end
        @synchronize()
    end

    if thread_id == 1
        output[block_id] = MinMax{eltype(values).parameters[1], eltype(values).parameters[2]}(smin[1], simin[1], smax[1], simax[1])
    end
end

"""
min_max_reduce(values, workgroupsSize, backend)

Performs min max reductions on values.

# Examples
```jldoctest
julia> min_max_reduce([1, 2, 3, 4], 4, CPU()) ... 
Main.MinMaxReduction.MinMax{Int64, UInt32}(1, 0x00000001, 4, 0x00000004) 
```
"""
function min_max_reduce(values, workgroupsSize, backend)
    
    n_groups = cld(length(values), workgroupsSize*2)
    partial_minmax_block = KernelAbstractions.zeros(backend, MinMax{eltype(values), UInt32}, n_groups)

    min_max_reduce_block_kernel(backend, workgroupsSize)(values, partial_minmax_block, length(values), ndrange=n_groups*workgroupsSize)
    KernelAbstractions.synchronize(backend)
    
    while(length(partial_minmax_block) > 1)
        n_remainder_groups = cld(length(partial_minmax_block), workgroupsSize)
        # TODO: SPEED UP technique de double buffering
        remainder_output = KernelAbstractions.zeros(backend, MinMax{eltype(values), UInt32}, n_remainder_groups)
        min_max_reduce_block_kernel_minmax(backend, workgroupsSize)(partial_minmax_block, remainder_output, length(partial_minmax_block),
        ndrange=n_remainder_groups*workgroupsSize)

        KernelAbstractions.synchronize(backend)
        partial_minmax_block = remainder_output
    end
    return Array(@view partial_minmax_block[end:end])[1] 
end
end