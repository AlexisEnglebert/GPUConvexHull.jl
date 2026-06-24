# GPUConvexHull.jl
This works is part of my Master thesis "GPU acceleration of convex hull computation".

Still in WIP.


## Example

```code
using GPUConvexHull
using KernelAbstractions

backend = CPU()

data = rand(n_dim, n_points)
gpu_pts = KernelAbstractions.zeros(backend, Float64, (n_dim, n_points))
copyto!(gpu_pts, data)

GPUConvexHull.quick_hull(backend, data)
```