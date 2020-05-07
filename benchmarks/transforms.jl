using Flux3D, BenchmarkTools

#ScalePointCloud
t = ScalePointCloud(0.5)
p = PointCloud(rand(1024,3))
@btime t(p) 

#RotatePointCloud
t = RotatePointCloud(rand(3,3))
p = PointCloud(rand(1024,3))
@btime t(p)

#ReAlignPointCloud
t = ReAlignPointCloud(PointCloud(rand(1024,3)))
p = PointCloud(rand(1024,3))
@btime t(p)

#NormalizePointCloud
t = NormalizePointCloud()
p = PointCloud(rand(1024,3))
@btime t(p)

#Compose
t = Compose(RotatePointCloud(rand(3,3)), ScalePointCloud(0.5), 
        ReAlignPointCloud(PointCloud(rand(1024,3))), NormalizePointCloud())
p = PointCloud(rand(1024,3))
@btime t(p)