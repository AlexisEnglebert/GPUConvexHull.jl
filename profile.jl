using Profile, ProfileView

data = rand(3, 100_000)
GPUConvexHull.quick_hull(CPU(), data) 
@profile GPUConvexHull.quick_hull(CPU(), data)
ProfileView.view()