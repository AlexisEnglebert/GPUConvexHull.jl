module ScanPrimitive
using KernelAbstractions

export segmented_scan

# Hmm à voir si on peut pas utiliser des grid en 3 dim pour rendre le calcul plus vite ? 
# TODO: Regarder la nouvelle méthode "LightScan" qui permetterais de compute les scan primitives plus rappidement.
# TODO: Voir comment CUB fait car il fait d'une manière différente ...  
# TODO: être consitant dans le nom des choses !!!
# TODO: move toutes les primitives de scan dans un autre fichier car là ce n'est plus tenable
# TODO: Faire en sorte d'utiliser l'opérateur d'identité du oplus à la place de 0 pour l'instant...
@kernel function segmented_scan_inner_block_downsweep!(out, partial_values, partial_flags, offset_block, in_flags, size::Integer, oplus::Function)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)
    
    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))
    @uniform tree_size = 1 << tree_power

     # Load all partial array in shared memory
    temp = @localmem(eltype(partial_values), tree_size)
    input_f = @localmem(eltype(in_flags), tree_size) 
    f = @localmem(eltype(in_flags), tree_size) 

    # Init tout à 0 comme ça on évite de faire une branche pour le padding:))
    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id
    temp[idx1] = 0; f[idx1] = 0; input_f[idx1] = 0
    temp[idx2] = 0; f[idx2] = 0; input_f[idx2] = 0
    
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
         temp[tree_size] = offset_block[block_id]
     end
    
    for shift in (tree_power-1):-1:0
        offset = 1 << shift
        if thread_id <= (tree_size >> (shift + 1))
            t = temp[offset*(thread_id*2)-offset] # Partie de gauche
            temp[offset*(thread_id*2)-offset] = temp[offset*(thread_id*2)]

            if input_f[offset*(thread_id*2)-offset+1] == 1
                temp[offset*(thread_id*2)] = 0 
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
    partial_flags, block_values, block_tree_flags, block_flags, values, size::Integer, flags, oplus::Function)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    @uniform group_size = first(@groupsize())
    @uniform tree_power = ceil(Int64, log2(group_size * 2))
    @uniform tree_size = 1 << tree_power

    # Each thread treats 2 values so in shared block memory there is 2 * block_size elements
    temp = @localmem(eltype(values), tree_size)
    input_f = @localmem(eltype(flags), tree_size) 
    f = @localmem(eltype(flags), tree_size) 

    # Init tout à 0 comme ça on évite de faire une branche pour le padding:))
    idx1 = (2*thread_id)-1
    idx2 = 2*thread_id
    temp[idx1] = 0; f[idx1] = 0; input_f[idx1] = 0
    temp[idx2] = 0; f[idx2] = 0; input_f[idx2] = 0

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
        if thread_id <= tree_size >> (shift+1)
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
        block_values[block_id] = temp[tree_size]
        block_tree_flags[block_id] = f[tree_size]
        block_flags[block_id] = input_f[1]
    end 
   
end

function segmented_scan_second_level_cpu!(block_values, size::Integer, flags, tree_flags, oplus::Function)
    
    # Values doit être une puissance de 2
    m_pow = ceil(Int, log2(size))
    n = 1 << m_pow
    
    temp = fill(0, n)
    f    = fill(0, n)
    fi   = fill(1, n)

    for i = 1:size
        temp[i] = block_values[i]
        f[i]    = tree_flags[i]
        fi[i]   = flags[i]
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
    temp[n] = 0

    for d = (m_pow-1):-1:0
        offset = 1 << d
        for k = 0:2*offset:n
            if k+2*offset <= n
                t = temp[k+offset]
                temp[k+offset] = temp[k+2*offset] 

                if fi[k+offset+1] == 1
                    temp[k+2*offset] = 0
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

    return temp[1:size]
end

@kernel function reverse_kernel!(a)
    i = @index(Global)
    n = length(a)
    if i <= n ÷ 2
        j = n - i + 1
        t = a[i]
        a[i] = a[j]
        a[j] = t
    end
end

@kernel function shift_right(input, output)
    global_id = @index(Global)
    @synchronize()
    n = length(input)
    if global_id <= n
        if global_id == 1
            output[global_id] = 0
        else
            output[global_id] = input[global_id-1]
        end
    end
end

@kernel function inclusive_kernell!(array, inital_value, oplus)
    global_id = @index(Global)
    n = length(array)
    if global_id <= n
        array[global_id] = oplus(array[global_id], inital_value[global_id])
    end
end

function segmented_scan(backend, values, flags, oplus::Function, backward=false, inclusive=false)
   n = length(values)
    nb_threads_per_block = 8
    nb_blocks = cld(n, 2*nb_threads_per_block)
    reverse_kernell = reverse_kernel!(backend, nb_threads_per_block)
    #shift_kernell   = shift_right(backend, nb_threads_per_block) TODO: Pas bon ça !!!!! 
    
    tmp_flags = []
    if backward
        reverse_kernell(values, ndrange = size(values))
        synchronize(backend)
        reverse_kernell(flags, ndrange = size(flags))
        tmp_flags = copy(flags) # TODO: move sur le gpu
        synchronize(backend)

        #shift_kernell(tmp_flags, flags, ndrange = size(flags))
        circshift!(flags, 1) # TODO GPU
        synchronize(backend)
    end
    
    # TODO: Only one GPU memory allocation for this
    partial_values        = fill(0, size(values))
    partial_flags         = fill(0, size(values))
    blocks_last_value     = fill(0, nb_blocks)
    blocks_last_flag      = fill(0, nb_blocks)
    blocks_last_tree_flag = fill(0, nb_blocks)
    final_array           = fill(0, size(values))
    
    upsweep_kernell = segmented_scan_inner_block_upsweep!(backend, nb_threads_per_block)    
    downsweep_kernell = segmented_scan_inner_block_downsweep!(backend, nb_threads_per_block)
    inclusive_kernell = inclusive_kernell!(backend, nb_threads_per_block)

    upsweep_kernell(partial_values, partial_flags, blocks_last_value,
    blocks_last_tree_flag, blocks_last_flag, values, length(values), 
    flags, oplus, ndrange = tuple(nb_blocks * nb_threads_per_block))
    
    synchronize(backend)


    segmented_blocks = segmented_scan_second_level_cpu!(
    blocks_last_value, length(blocks_last_value), blocks_last_flag, blocks_last_tree_flag, oplus)


    downsweep_kernell(final_array, partial_values, partial_flags, segmented_blocks, flags, length(partial_values), oplus, ndrange = tuple(nb_blocks * nb_threads_per_block))
    synchronize(backend)

    if inclusive
        inclusive_kernell(final_array, values, oplus, ndrange = size(final_array))
        synchronize(backend)
    end

    if backward
        reverse_kernell(values, ndrange = size(values))
        synchronize(backend)
        reverse_kernell(tmp_flags, ndrange = size(tmp_flags))
        synchronize(backend)
        reverse_kernell(final_array, ndrange = size(values))
        synchronize(backend)

        flags .= tmp_flags # TODO le move en GPU coalescence toussa toussa
    end

    return final_array
end

end