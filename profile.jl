using Profile, ProfileView
using GPUConvexHull
using KernelAbstractions

#Profile.init(delay=0.00001)
data = rand(3, 10_000)
GPUConvexHull.quick_hull(CPU(), data) 
Profile.clear() 
@profile for _ in 1:100; GPUConvexHull.quick_hull(CPU(), data); end
Profile.print(format=:flat, sortedby=:count)
ProfileView.view()
