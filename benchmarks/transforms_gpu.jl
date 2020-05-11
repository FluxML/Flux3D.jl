using Flux3D, BenchmarkTools
using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

function benchmark_t(t, p)
    CuArrays.@sync begin
        t(p)
    end
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
            # TODO: problem with ReAlignPointCloud (outofmemory)
			# (ReAlignPointCloud(PointCloud(rand(npoints,3))),"ReAlignPointCloud"),
			(NormalizePointCloud(),"NormalizePointCloud")]

    println("*"^10," npoints = $(npoints), device = cuda ","*"^10)
    run_benchmarks_gpu(arr, npoints)
    println()
end

# Google Colab output [runtime: GPU] 

# ********** npoints = 16384, device = cuda **********
# ********** ScalePointCloud **********
#   39.366 μs (63 allocations: 1.70 KiB)
# ********** RotatePointCloud **********
#   29.869 μs (7 allocations: 336 bytes)
# ********** NormalizePointCloud **********
#   972.776 μs (1482 allocations: 54.11 KiB)

# ********** npoints = 65536, device = cuda **********
# ********** ScalePointCloud **********
#   38.538 μs (75 allocations: 1.89 KiB)
# ********** RotatePointCloud **********
#   54.231 μs (17 allocations: 496 bytes)
# ********** NormalizePointCloud **********
#   993.053 μs (1482 allocations: 54.11 KiB)

# ********** npoints = 262144, device = cuda **********
# ********** ScalePointCloud **********
#   151.437 μs (76 allocations: 1.91 KiB)
# ********** RotatePointCloud **********
#   176.519 μs (17 allocations: 496 bytes)
# ********** NormalizePointCloud **********
#   1.123 ms (1484 allocations: 54.14 KiB)

# ********** npoints = 1048576, device = cuda **********
# ********** ScalePointCloud **********
#   252.326 μs (80 allocations: 1.97 KiB)
# ********** RotatePointCloud **********
#   507.770 μs (17 allocations: 496 bytes)
# ********** NormalizePointCloud **********
#   2.645 ms (1503 allocations: 55.44 KiB)
