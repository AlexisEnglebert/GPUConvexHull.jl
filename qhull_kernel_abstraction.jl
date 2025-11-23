using KernelAbstractions, Polyhedra

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
@kernel function segmented_scan_inner_block!(values, size::Integer, flags, oplus::Function)
    global_id = @index(Global)
    thread_id = @index(Local)
    
    # Each thread treats 2 values so in shared block memory there is 2 * block_size elements
    temp = @localmem(eltype(values), first(@groupsize()) * 2)

    if ((2*global_id)-1 > size)
        temp[(2*thread_id)-1] = 0
    else
        temp[(2*thread_id)-1] = values[(2*global_id)-1]
    end

    if ((2 * global_id) > size)
        temp[2*thread_id] = 0
    else
        temp[2*thread_id] = values[2*global_id]
    end

    @synchronize()
    
    for shift = 0:floor(Int64, ilog2(size))
        offset = 1 << shift
        if thread_id < (size >> shift)
            temp[offset*(thread_id*2)] = temp[offset*(thread_id*2)]+temp[offset*(thread_id*2)-offset]
        end
        @synchronize()
    end

     if thread_id == 1
        @print("after upsweep : ", temp, values, " \n")
    end

    # Downsweep (TODO: Fix (last element is not right))
    temp[size] = 0

    for shift = floor(Int64, log2(size)):-1:0 # très perturbant le range au millieu
        offset = 1 << shift
        if thread_id < (size >> shift) 
            t = temp[offset*(thread_id*2)-offset] # Partie de gauche
            temp[offset*(thread_id*2)-offset] = temp[offset*(thread_id*2)]
            temp[offset*(thread_id*2)] += t  
        end
        if thread_id == 1
            @print("ICI : ",temp, values, " \n")
        end
        @synchronize()

    end

    if thread_id == 1
        @print("Final prefix sum : ",temp, values, " \n")
    end
end

backend = CPU()
values = [3, 1, 7,  2, 4, 1, 6, 5 ]
flags = [1, 0, 0,  1, 0, 1, 0, 0 ]
oplus(a, b) = a + b
ev = segmented_scan_inner_block!(backend, 8)
ev(values, size(values)[1], flags, oplus, ndrange = (size(values))) #ndrange is the number of thread in total
synchronize(backend)

#TODO segmented min-max scan

@kernel function quickhull_kernel()
    #First use MIN-MAX segmented scan kernel to find the original simplex
    
end

function gpuquickhull(V::VRepresentation)

end