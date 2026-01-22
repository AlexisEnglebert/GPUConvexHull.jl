module qhull_kernel_abstraction
using KernelAbstractions, Polyhedra

export segmented_scan

###### Les paramètres ici sont pour tester, ils devront être setup lorsque la librairie.
backend = CPU()


"""
Computes inclusive segmented scan for a given ⊕ operation using Multi-block segmented scan algorithm
to be more generic. from:

Sengupta, S., Harris, M., Zhang, Y., & Owens, J. D. (2007). Scan primitives for GPU computing.

(Je pense que de base c'est proposé par Blelloch )

1: Perform reduce on all blocks in parallel
2: Save partial sum and partial OR trees to global memory
3: Do second-level segmented scans with final sums
4: Load partial sum and partial OR trees from global mem-
ory to shared memory
5: Set last element of each block to corresponding element
in the output of second-level segmented scan
6: Perform down-sweep on all blocks in parallel

exemple:
    values = [3, 1, 7,  2, 4, 1, 6, 5 ]
    flags  = [1, 0, 0,  1, 0, 1, 0, 0 ]
    output = [3, 4, 11, 2, 6, 1, 7, 12]
"""

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

     # Load all partial array in shared memory
    temp = @localmem(eltype(partial_values), first(@groupsize()) * 2)
    input_f = @localmem(eltype(flags), first(@groupsize()) * 2) 
    f = @localmem(eltype(flags), first(@groupsize()) * 2) 

    # TODO: Meilleur organisation needed mdr
    if ((2*global_id)-1 >= size)
        temp[(2*thread_id)-1] = 0
        f[(2*thread_id)-1] = 0
        input_f[(2*thread_id)-1] = 0
    else
        temp[(2*thread_id)-1] = partial_values[(2*global_id)-1]
        f[(2*thread_id)-1] = partial_flags[(2*global_id)-1]
        input_f[(2*thread_id)-1] = in_flags[(2*global_id)-1]
    end

    if ((2 * global_id) >= size)
        temp[2*thread_id] = 0
        f[(2*thread_id)] = 0
        input_f[(2*thread_id)] = 0
    else
        temp[2*thread_id] = partial_values[2*global_id]
        f[2*thread_id] = partial_flags[2*global_id]
        input_f[2*thread_id] = in_flags[2*global_id]
    end

    temp[(first(@groupsize()) * 2)] = offset_block[block_id]
    
    for shift = floor(Int64, log2((first(@groupsize()) * 2))):-1:0
        offset = 1 << shift
        if thread_id <= ((first(@groupsize()) * 2) >> (shift+1)) 
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

    if thread_id == 1
        @print("Final segmented array is : ", temp, "\n")
    end
end

@kernel function segmented_scan_inner_block_upsweep!(partial_values, 
    partial_flags, block_values, block_tree_flags, block_flags, values, size::Integer, flags, oplus::Function)
    global_id = @index(Global)
    thread_id = @index(Local)
    block_id  = @index(Group)

    # Each thread treats 2 values so in shared block memory there is 2 * block_size elements
    temp = @localmem(eltype(values), first(@groupsize()) * 2)
    input_f = @localmem(eltype(flags), first(@groupsize()) * 2) 
    f = @localmem(eltype(flags), first(@groupsize()) * 2) 

    if ((2*global_id)-1 >= size)
        temp[(2*thread_id)-1] = 0
        f[(2*thread_id)-1] = 0
        input_f[(2*thread_id)-1] = 0
    else
        temp[(2*thread_id)-1] = values[(2*global_id)-1]
        f[(2*thread_id)-1] = flags[(2*global_id)-1]
        input_f[(2*thread_id)-1] = flags[(2*global_id)-1]
    end

    if ((2 * global_id) >= size)
        temp[2*thread_id] = 0
        f[(2*thread_id)] = 0
        input_f[(2*thread_id)] = 0
    else
        temp[2*thread_id] = values[2*global_id]
        f[2*thread_id] = flags[2*global_id]
        input_f[2*thread_id] = flags[2*global_id]
    end
    
    @synchronize()
    
    if thread_id == 1
        @print("Segmented scan block : ",block_id, " got this input", temp, " from :", values, "\n")
    end

    for shift = 0:floor(Int64, log2(first(@groupsize()) * 2))
        offset = 1 << shift
        if thread_id <= ((first(@groupsize()) * 2)  >> (shift+1))
            if f[offset*(thread_id*2)] == 0
                temp[offset*(thread_id*2)] = oplus(temp[offset*(thread_id*2)], temp[offset*(thread_id*2)-offset])
            end
            f[offset*(thread_id*2)] |= f[offset*(thread_id*2)-offset]
        end
        @synchronize()
    end

    if thread_id == 1
        @print("Computed prefix sum in block: ", temp, "\n")
    end

        # Fill the out array
    if ((2*global_id)-1 <= size)
        partial_values[(2*global_id)-1] = temp[(2*thread_id)-1]
        partial_flags[(2*global_id)-1] = f[(2*thread_id)-1]
    end

    if ((2*global_id) <= size)
        partial_values[(2*global_id)] = temp[(2*thread_id)]
        partial_flags[(2*global_id)] = f[(2*thread_id)]
    end

    @synchronize()
    if thread_id == 1
        block_values[block_id] = temp[first(@groupsize()) * 2]
        block_tree_flags[block_id] = f[first(@groupsize()) * 2]
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
    for d = 0:m_pow-1
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
    println("temp : ", temp)

    # Downsweep 
    temp[n] = 0

    for d = m_pow-1:-1:0
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
    
    println("second level: ", temp)
    return temp[1:size]
end

# TODO: n/2 thread
@kernel function reverse_kernell!(array)
    global_id = @index(Global)
    n = length(array)
    if global_id <= n # FAILSAFE, ça ne devrait pas arriver :o
        t = array[global_id]
        array[global_id] = array[n-global_id+1] #PAS BON !!!!!!
        array[n-global_id+1] = t
    end
end

@kernel function shift_right(input, output)
    global_id = @index(Global)
    @synchronize()
    n = length(input)
    if global_id == 1
        @print("wtf : ", input[4], "\n")
    end
    if global_id <= n
        if global_id == 1
            output[global_id] = 0
        else
            output[global_id] = input[global_id-1]
            @print(global_id, " ", output[global_id], " ", input[(global_id-1)], "   ", (global_id-1), "\n")
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

function segmented_scan(values, flags, oplus::Function, backward=false, inclusive=false)
  
    n = length(values)
    nb_threads_per_block = 4
    nb_blocks = cld(n, 2*nb_threads_per_block)
    reverse_kernell = reverse_kernell!(backend, nb_threads_per_block)
    shift_kernell   = shift_right(backend, nb_threads_per_block) 
    tmp_flags = []
    if backward
        reverse_kernell(values, ndrange = size(values))
        synchronize(backend)
        reverse_kernell(flags, ndrange = size(flags))
        tmp_flags = copy(flags) # TODO: move sur le gpu
        synchronize(backend)
        println("Reversed flags before shift: ", flags)
        println("Reversed TMP before shift: ", tmp_flags)

        shift_kernell(tmp_flags, flags, ndrange = size(flags))
        synchronize(backend)

        println("Reversed data: ", values)
        println("Reversed flags: ", flags)
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

    # Ok donc ici on vas faire un "second-level-segmented-scan aussi en mode tree pour pouvoir handle
    # toutes les longueur d'array
    println("Block last sum:", blocks_last_value, " \nBlock last tree flag: ", blocks_last_tree_flag, " \nBlock last flag: ", blocks_last_flag)
    println("Partial prefix sum: ", partial_values)
    println("Partial flags: ", partial_flags)

    # TODO: On le fait sur CPU pour l'instant move sur CUB pour finir.
    segmented_blocks = segmented_scan_second_level_cpu!(
    blocks_last_value, length(blocks_last_value), blocks_last_flag, blocks_last_tree_flag, oplus)

    println("Segmented block prefix sum: ", segmented_blocks)
    # On offset les blocks avec le blocks_last_value
    downsweep_kernell(final_array, partial_values, partial_flags, segmented_blocks, flags, length(partial_values), oplus, ndrange = tuple(nb_blocks * nb_threads_per_block))
    synchronize(backend)

    println("Final array: ", final_array)

    if inclusive
        inclusive_kernell(final_array, values, oplus, ndrange = size(final_array))
        synchronize(backend)
    end

    if backward
        reverse_kernell(values, ndrange = size(values))
        synchronize(backend)
        reverse_kernell(tmp_flags, ndrange = size(tmp_flags))
        synchronize(backend)

        flags = tmp_flags # TODO le move en GPU coalescence toussa toussa
    end
    # On fait le downsweep
    return final_array
end

"""
segment_mask_kernel(b, p, out, identity_val)

Nom un peu fancy du papier, ça permet juste de fill l'array out avec les index de p

# Examples
```jldoctest
julia> a ....
```
"""
@kernel function segment_mask_kernel(b, p, data, out)
    i = @index(Global)
    if i <= length(data)
        if b[i] == 1
            out[p[i]+1] = data[i]
        end
    end
end


"""
compact(b, s, n)


removes elements within segments.

Compact takes in an array of booleans `b` and
discards data from `s`, which are marked as false by the
boolean array

# Examples
```jldoctest
julia> a = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4
```
"""
# TODO: Comprends à quoi sert le `s`
function compact(b, s, n)
    flags = fill(0, length(b))
    p = segmented_scan(b, flags, (a, b) -> a + b) 
    n = p[n]+b[n] # On ajoute le b[n] car on est en exclusive scan =))
    println(p)
    out_array = fill(0, n)
    segment_mask_ker = segment_mask_kernel(backend, 4)
    segment_mask_ker(b, p, s, out_array, ndrange = length(b))

    println("Out array is : ", out_array)
    # TODO: sp ⇐ segmented scan(newSegment)
    #sp = segmented_scan()
end

@kernel function minmax_reduce(values)
    thread_id = @index(Local)
    global_id = @index(Global)
    shared = @localmem(eltype(partial_values), (first(@groupsize(), 2)))
    blockDimX = first(@groupsize())

    # Load to shared memory
    if global_id <= length(values)
        shared[global_id][1] = values[global_id]
        shared[global_id][2] = global_id
    end

    @synchronize()

    # Perform the reduction, returns [(value, index), (value, index)] with(min, max)
    # Todo loop 
    for shift = floor(Int64, log2((first(@groupsize()) * 2))):-1:0
        offset = 1 << shift
        if thread_id <= ((first(@groupsize()) * 2) >> (shift+1)) 
            # Todo avoid if, else for min and max
            shared[thread_id] = min(shared[thread_id][0], shared[thread_id + offset][0])
        end
    end
 

end

function quick_hull(points)
    #presence_flag = fill(1, len(points))
    # Find min and max point in a given dimension
    flags = fill(0, length(points))
    segmented_scan(map((x) -> x[1],points), flags, (a, b) -> min(a, b))

    # Remove from points using compact

    # TODO utiliser KernelAbstractions.zeros, pour créer des listes dans la mémoire du GPU et 
    # donc ne pas devoir transférer les donnés du CPU au GPU à chaque fois qu'on lance un kernell
end

#= values = [3, 1, 7,  2, 4, 1, 6, 5, 1, 1, 1, 1, 1, 1, 1, 1]
flags =  [1, 0, 0,  0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
oplus(a, b) = a + b
segmented_scan(values, flags, oplus)


flag = [0, 0, 0, 1, 0, 0, 0, 0]
data = [1, 2, 3, 4, 5, 6, 7, 8]

segmented_scan(data, flag, oplus)
segmented_scan(data, flag, oplus, true)
segmented_scan(data, flag, oplus, true, true)

compact_bool_data = [1, 0, 1, 0, 1, 1, 1, 0]
compact_data      = [0, 1, 2, 3, 4, 5, 6, 7]    
#compact_kernell = compact(backend, 4)
compact(compact_bool_data, compact_data, length(compact_bool_data))


points = [(0, 2), (-2, 0), (0, -2), (2, 0), (3, 3)]
quick_hull(points)


min_max_values = [10, 3 ,4, 0, 9, 8, 2, 2, 2]
min_max_values_ker = min_max_values(backend, 9)
min_max_values_ker(values, ndrange=length(min_max_values)) =#
end