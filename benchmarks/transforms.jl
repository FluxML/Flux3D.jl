using Flux3D, BenchmarkTools

npoints = 16384

#ScalePointCloud
t = ScalePointCloud(0.5)
p = PointCloud(rand(npoints,3))
println("*"^10,"ScalePointCloud","*"^10)
@btime t(p) 

#RotatePointCloud
t = RotatePointCloud(rand(3,3))
p = PointCloud(rand(npoints,3))
println("*"^10,"RotatePointCloud","*"^10)
@btime t(p)

#ReAlignPointCloud
t = ReAlignPointCloud(PointCloud(rand(npoints,3)))
p = PointCloud(rand(npoints,3))
println("*"^10,"ReAlignPointCloud","*"^10)
@btime t(p)

#NormalizePointCloud
t = NormalizePointCloud()
p = PointCloud(rand(npoints,3))
println("*"^10,"NormalizePointCloud","*"^10)
@btime t(p)

#Compose
t = Compose(ScalePointCloud(0.5), RotatePointCloud(rand(3,3)), 
        ReAlignPointCloud(PointCloud(rand(npoints,3))), NormalizePointCloud())
p = PointCloud(rand(npoints,3))
println("*"^10,"Compose","*"^10)
@btime t(p)

# Google Colab output [runtime: GPU, npoints: 16384] 

# **********ScalePointCloud**********
#   3.909 μs (0 allocations: 0 bytes)
# **********RotatePointCloud**********
#   40.896 μs (2 allocations: 192.08 KiB)
# **********ReAlignPointCloud**********
#   231.779 μs (44 allocations: 193.45 KiB)
# **********NormalizePointCloud**********
#   71.492 μs (5 allocations: 192.36 KiB)
# **********Compose**********
#   383.588 μs (52 allocations: 577.92 KiB)