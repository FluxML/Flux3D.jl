using Flux3D, BenchmarkTools, BSON

function setup_benchmark_record(names)
    benchmarks = Dict{String, Vector{Float64}}() 
    for name in names
        benchmarks[name] = []
    end
    return benchmarks
end

function generate_point_cloud(npoints::Int)
    points = ones(npoints, 3)
    points = cumsum(points, dims = 1)
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
push!(names, "Compose")

cpu_benchmarks = setup_benchmark_record(names)

@info "DEVICE: CPU"
for npoints in npoint_arr
    arr = [
	(ScalePointCloud(0.5; inplace=false), "ScalePointCloud"),
	(RotatePointCloud(ROT_MATRIX; inplace=false), "RotatePointCloud"),
        (ReAlignPointCloud(realign_point_cloud(npoints); inplace=false), "RealignPointCloud"),
	(NormalizePointCloud(inplace=false), "NormalizePointCloud"),
        (Compose(
             ScalePointCloud(0.5; inplace=false),
	     RotatePointCloud(ROT_MATRIX; inplace=false),
	     ReAlignPointCloud(realign_point_cloud(npoints); inplace=false),
	     NormalizePointCloud()), "Compose")
    ]

    @info "Running benchmarks for npoints = $npoints\n"
    run_benchmarks!(cpu_benchmarks, arr, npoints, (op, pc) -> op(pc), cpu)
    println()
end
   
gpu_benchmarks = setup_benchmark_record(names)

using CUDAapi
if has_cuda()
    @info "CUDA is on. Running GPU Benchmarks"
    import CuArrays
    CuArrays.allowscalar(false)

    @info "DEVICE: GPU"
    for npoints in npoint_arr
        arr = [
	    (ScalePointCloud(0.5; inplace=false), "ScalePointCloud"),
            (RotatePointCloud(ROT_MATRIX; inplace=false), "RotatePointCloud"),
            (ReAlignPointCloud(realign_point_cloud(npoints); inplace=false), "RealignPointCloud"),
	    (NormalizePointCloud(inplace=false), "NormalizePointCloud"),
            (Compose(
                 ScalePointCloud(0.5; inplace=false),
	         RotatePointCloud(ROT_MATRIX; inplace=false),
	         ReAlignPointCloud(realign_point_cloud(npoints); inplace=false),
	         NormalizePointCloud()), "Compose")
         ]

        @info "Running benchmarks for npoints = $npoints\n"
        run_benchmarks!(gpu_benchmarks, arr, npoints, (op, pc) -> CuArrays.@sync op(pc), gpu)
        println()
    end
end

fname = joinpath(@__DIR__, "transform_benchmarks.bson")
BSON.@save fname cpu_benchmarks gpu_benchmarks
@info "Benchmarks have been saved at $fname"
