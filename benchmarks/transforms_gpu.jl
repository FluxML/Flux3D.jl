using Flux3D, BenchmarkTools
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

function benchmark_t(t, p)
    CuArrays.@sync t(p)
end

function run_benchmarks_gpu(arr, npoints)
    for (t, name) in arr
        println("*"^10," $(name) ","*"^10)
        t = t |> gpu
		p = PointCloud(rand(npoints,3)) |> gpu
		@btime benchmark_t($t, $p)
	end
end

for npoints in [2^14, 2^16, 2^18, 2^20]

	arr = [(ScalePointCloud(0.5),"ScalePointCloud"),
            (RotatePointCloud(rand(3,3)),"RotatePointCloud"),
			(ReAlignPointCloud(PointCloud(rand(npoints,3))),"ReAlignPointCloud"),
			(NormalizePointCloud(),"NormalizePointCloud")]

    println("*"^10," npoints = $(npoints), device = cuda ","*"^10)
    run_benchmarks_gpu(arr, npoints)
    println()
end

# Google Colab output [runtime: GPU]

# ********** npoints = 16384, device = cuda **********
# ********** ScalePointCloud **********
#   35.030 μs (75 allocations: 1.89 KiB)
# ********** RotatePointCloud **********
#   23.688 μs (17 allocations: 496 bytes)
# ********** ReAlignPointCloud **********
#   719.597 μs (910 allocations: 37.50 KiB)
# ********** NormalizePointCloud **********
#   1.303 ms (1572 allocations: 55.98 KiB)
#
# ********** npoints = 65536, device = cuda **********
# ********** ScalePointCloud **********
#   42.299 μs (75 allocations: 1.89 KiB)
# ********** RotatePointCloud **********
#   31.316 μs (17 allocations: 496 bytes)
# ********** ReAlignPointCloud **********
#   708.265 μs (910 allocations: 37.50 KiB)
# ********** NormalizePointCloud **********
#   1.405 ms (1626 allocations: 57.11 KiB)
#
# ********** npoints = 262144, device = cuda **********
# ********** ScalePointCloud **********
#   50.675 μs (76 allocations: 1.91 KiB)
# ********** RotatePointCloud **********
#   111.004 μs (17 allocations: 496 bytes)
# ********** ReAlignPointCloud **********
#   806.898 μs (911 allocations: 37.52 KiB)
# ********** NormalizePointCloud **********
#   1.462 ms (1628 allocations: 57.14 KiB)
#
# ********** npoints = 1048576, device = cuda **********
# ********** ScalePointCloud **********
#   144.797 μs (80 allocations: 1.97 KiB)
# ********** RotatePointCloud **********
#   222.691 μs (17 allocations: 496 bytes)
# ********** ReAlignPointCloud **********
#   1.002 ms (917 allocations: 37.61 KiB)
# ********** NormalizePointCloud **********
#   1.638 ms (1639 allocations: 57.31 KiB)
