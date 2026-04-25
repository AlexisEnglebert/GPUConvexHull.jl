using Profile
using GPUConvexHull
using KernelAbstractions

Profile.init(delay=0.000001)

backend = CPU()
data = rand(3, 100_000)
GPUConvexHull.quick_hull(backend, data) 
Profile.clear() 

VSCodeServer.@profview GPUConvexHull.quick_hull(backend, data)

