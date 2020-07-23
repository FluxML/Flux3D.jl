using Flux3D, BenchmarkTools, Printf

function setup_benchmark_record(names)
    benchmarks = Dict{String, Vector{Float64}}()
    for name in names
        benchmarks[name] = []
    end
    return benchmarks
end

function generate_point_cloud(npoints::Int)
    points = ones(3, npoints)
    points = cumsum(points, dims = 2)
    return PointCloud(points / npoints)
end

function run_benchmarks!(benchmarks, x, npoints, benchmark_func, device)
    for (transform, name) in x
        transform = transform |> device
        pc = generate_point_cloud(npoints) |> device
        trial = @benchmark $benchmark_func($transform, $pc)
        time = minimum(trial.times) * 1.0e-6
        println("$name: $time ms")
        push!(benchmarks[name], time)
    end
end

function realign_point_cloud(npoints)
    pc = generate_point_cloud(npoints)
    rot = RotatePointCloud(-ROT_MATRIX)
    return rot(pc)
end

ROT_MATRIX = [1.0 2.0 3.0
	      	  0.2 0.5 0.9
	      	  3.0 2.0 1.0]

npoint_arr = 2 .^ [12, 14, 16, 18, 20]

names = ["Scale", "Rotate", "Realign", "Normalize"] .* "PointCloud"
push!(names, "Chain")

cpu_benchmarks = setup_benchmark_record(names)

println("DEVICE: CPU")
for _npoints in npoint_arr
    arr = [
	(ScalePointCloud(0.5; inplace=false), "ScalePointCloud"),
	(RotatePointCloud(ROT_MATRIX; inplace=false), "RotatePointCloud"),
        (ReAlignPointCloud(realign_point_cloud(_npoints); inplace=false), "RealignPointCloud"),
	(NormalizePointCloud(inplace=false), "NormalizePointCloud"),
        (Chain(
             ScalePointCloud(0.5; inplace=false),
	     RotatePointCloud(ROT_MATRIX; inplace=false),
	     ReAlignPointCloud(realign_point_cloud(_npoints); inplace=false),
	     NormalizePointCloud()), "Chain")
    ]

    println("Running benchmarks for npoints = $_npoints")
    run_benchmarks!(cpu_benchmarks, arr, _npoints, (op, pc) -> op(pc), cpu)
    println()
end

gpu_benchmarks = setup_benchmark_record(names)

using CUDA
if has_cuda()
    println("CUDA is on. Running GPU Benchmarks")
    CUDA.allowscalar(false)
    println("DEVICE: GPU")
    for _npoints in npoint_arr
        arr = [
	    (ScalePointCloud(0.5; inplace=false), "ScalePointCloud"),
            (RotatePointCloud(ROT_MATRIX; inplace=false), "RotatePointCloud"),
            (ReAlignPointCloud(realign_point_cloud(_npoints); inplace=false), "RealignPointCloud"),
	    (NormalizePointCloud(inplace=false), "NormalizePointCloud"),
            (Chain(
                 ScalePointCloud(0.5; inplace=false),
	         RotatePointCloud(ROT_MATRIX; inplace=false),
	         ReAlignPointCloud(realign_point_cloud(_npoints); inplace=false),
	         NormalizePointCloud(inplace=false)), "Chain")
         ]

        println("Running benchmarks for npoints = $_npoints")
        run_benchmarks!(gpu_benchmarks, arr, _npoints, (op, pc) -> (CuArrays.@sync op(pc)), gpu)
        println()
    end
end

function save_bm(fname, cpu_benchmarks, gpu_benchmarks)
	open(fname, "w") do io
		device = "cpu"
		for (key, values) in cpu_benchmarks
			for (p,v) in zip(npoint_arr, values)
				Printf.@printf(io, "%s %s %d %f ms\n",device, key, p, v)
			end
		end

		device = "gpu"
		for (key, values) in gpu_benchmarks
			for (p,v) in zip(npoint_arr, values)
				Printf.@printf(io, "%s %s %d %f ms\n",device, key, p, v)
			end
		end
	end
end

fname = joinpath(@__DIR__, "bm_flux3d.txt")
save_bm(fname, cpu_benchmarks, gpu_benchmarks)
@info "Benchmarks have been saved at $fname"
