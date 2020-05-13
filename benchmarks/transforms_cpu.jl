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
#   3.212 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   34.843 μs (2 allocations: 192.08 KiB)
# ********** ReAlignPointCloud **********
#   171.015 μs (36 allocations: 193.13 KiB)
# ********** NormalizePointCloud **********
#   81.388 μs (14 allocations: 192.58 KiB)
# ********** Compose **********
#   295.170 μs (54 allocations: 577.83 KiB)

# ********** npoints = 65536, device = cpu **********
# ********** ScalePointCloud **********
#   12.812 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   192.015 μs (2 allocations: 768.08 KiB)
# ********** ReAlignPointCloud **********
#   732.088 μs (36 allocations: 769.13 KiB)
# ********** NormalizePointCloud **********
#   393.066 μs (14 allocations: 768.58 KiB)
# ********** Compose **********
#   2.238 ms (54 allocations: 2.25 MiB)

# ********** npoints = 262144, device = cpu **********
# ********** ScalePointCloud **********
#   117.308 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   836.519 μs (2 allocations: 3.00 MiB)
# ********** ReAlignPointCloud **********
#   2.910 ms (36 allocations: 3.00 MiB)
# ********** NormalizePointCloud **********
#   1.863 ms (14 allocations: 3.00 MiB)
# ********** Compose **********
#   9.166 ms (54 allocations: 9.00 MiB)

# ********** npoints = 1048576, device = cpu **********
# ********** ScalePointCloud **********
#   495.236 μs (1 allocation: 16 bytes)
# ********** RotatePointCloud **********
#   3.525 ms (2 allocations: 12.00 MiB)
# ********** ReAlignPointCloud **********
#   11.824 ms (36 allocations: 12.00 MiB)
# ********** NormalizePointCloud **********
#   7.825 ms (14 allocations: 12.00 MiB)
# ********** Compose **********
#   37.393 ms (54 allocations: 36.00 MiB)