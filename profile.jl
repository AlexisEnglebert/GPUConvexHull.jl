using Profile, ProfileView

#Profile.init(delay=0.00001)
data = rand(3, 100_000)
GPUConvexHull.quick_hull(CPU(), data) 
Profile.clear() 
@profile for _ in 1:100; GPUConvexHull.quick_hull(CPU(), data); end
ProfileView.view()
