using Flux3D, BenchmarkTools

function run_benchmarks_cpu(arr, npoints)
	for (t, name) in arr
		println("*"^10," $(name) ","*"^10)
		p = PointCloud(rand(npoints,3))
		@btime $t($p)
	end
end

for npoints in [2^14, 2^16, 2^18, 2^20]

	arr = [(ScalePointCloud(0.5),"ScalePointCloud"),
			(RotatePointCloud(rand(3,3)),"RotatePointCloud"),
			(ReAlignPointCloud(PointCloud(rand(npoints,3))),"ReAlignPointCloud"),
			(NormalizePointCloud(),"NormalizePointCloud"),
			(Compose(ScalePointCloud(0.5), 
					RotatePointCloud(rand(3,3)), 
					ReAlignPointCloud(PointCloud(rand(npoints,3))), 
					NormalizePointCloud()), "Compose")]

    println("*"^10," npoints = $(npoints), device = cpu ","*"^10)
    run_benchmarks_cpu(arr, npoints)
    println()
end

# Google Colab output [runtime: GPU] 

# ********** npoints = 16384, device = cpu **********
# ********** ScalePointCloud **********
#   3.313 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   47.605 μs (2 allocations: 192.08 KiB)
# ********** ReAlignPointCloud **********
#   234.372 μs (58 allocations: 193.75 KiB)
# ********** NormalizePointCloud **********
#   82.026 μs (14 allocations: 192.58 KiB)
# ********** Compose **********
#   380.743 μs (76 allocations: 578.45 KiB)

# ********** npoints = 65536, device = cpu **********
# ********** ScalePointCloud **********
#   13.135 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   193.864 μs (2 allocations: 768.08 KiB)
# ********** ReAlignPointCloud **********
#   961.837 μs (58 allocations: 769.75 KiB)
# ********** NormalizePointCloud **********
#   395.535 μs (14 allocations: 768.58 KiB)
# ********** Compose **********
#   2.471 ms (76 allocations: 2.25 MiB)

# ********** npoints = 262144, device = cpu **********
# ********** ScalePointCloud **********
#   119.710 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   842.307 μs (2 allocations: 3.00 MiB)
# ********** ReAlignPointCloud **********
#   3.790 ms (58 allocations: 3.00 MiB)
# ********** NormalizePointCloud **********
#   1.904 ms (14 allocations: 3.00 MiB)
# ********** Compose **********
#   10.085 ms (76 allocations: 9.00 MiB)

# ********** npoints = 1048576, device = cpu **********
# ********** ScalePointCloud **********
#   500.836 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   3.648 ms (2 allocations: 12.00 MiB)
# ********** ReAlignPointCloud **********
#   15.792 ms (58 allocations: 12.00 MiB)
# ********** NormalizePointCloud **********
#   7.926 ms (14 allocations: 12.00 MiB)
# ********** Compose **********
#   40.644 ms (76 allocations: 36.00 MiB)