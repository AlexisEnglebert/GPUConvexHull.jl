using Profile, ProfileView
using GPUConvexHull
using KernelAbstractions
using CUDA

backend = CUDABackend()
#Profile.init(delay=0.00001)
data = rand(3, 100_000)
GPUConvexHull.quick_hull(backend, data) 
Profile.clear() 
@profile for _ in 1:100; GPUConvexHull.quick_hull(backend, data); end
ProfileView.view()
