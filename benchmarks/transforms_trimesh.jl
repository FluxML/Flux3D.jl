using Flux3D, BenchmarkTools, Printf

function setup_benchmark_record(names)
    benchmarks = Dict{String, Vector{Float64}}()
    for name in names
        benchmarks[name] = []
    end
    return benchmarks
end

function run_benchmarks!(benchmarks, x, mesh, benchmark_func, device)
    for (transform, name) in x
        transform = transform |> device
        pc = load_trimesh(mesh) |> device
        trial = @benchmark $benchmark_func($transform, $pc)
        time = minimum(trial.times) * 1.0e-6
        println("$name: $time ms")
        push!(benchmarks[name], time)
    end
end

function realign_trimesh(mesh)
    pc = load_trimesh(mesh)
    rot = RotateTriMesh(-ROT_MATRIX)
    return rot(pc)
end

ROT_MATRIX = [1.0 2.0 3.0
	      	  0.2 0.5 0.9
	      	  3.0 2.0 1.0]

mesh_arr = ["../../teapot.obj", "../../truck_daf.obj"]

names = ["Scale", "Rotate", "Realign", "Normalize", "Translate"] .* "TriMesh"
push!(names, "Chain")

cpu_benchmarks = setup_benchmark_record(names)
println("DEVICE: CPU")
inplace = true
for mesh in mesh_arr
	arr = [
		(ScaleTriMesh(0.5; inplace=inplace), "ScaleTriMesh"),
		(RotateTriMesh(ROT_MATRIX; inplace=inplace), "RotateTriMesh"),
	    (ReAlignTriMesh(realign_trimesh(mesh); inplace=inplace), "RealignTriMesh"),
		(NormalizeTriMesh(inplace=inplace), "NormalizeTriMesh"),
	    (Chain(ScaleTriMesh(0.5; inplace=inplace),
		       RotateTriMesh(ROT_MATRIX; inplace=inplace),
		       ReAlignTriMesh(realign_trimesh(mesh); inplace=inplace),
		       NormalizeTriMesh()), "Chain")
    ]

    println("Running benchmarks for trimesh with no. of verts = $(load_trimesh(mesh).V)")
    run_benchmarks!(cpu_benchmarks, arr, mesh, (op, pc) -> op(pc), cpu)
    println()
end

gpu_benchmarks = setup_benchmark_record(names)
using CUDA
if has_cuda()
    println("CUDA is on. Running GPU Benchmarks")
    println("DEVICE: GPU")
    for mesh in mesh_arr
		arr = [
			(ScaleTriMesh(0.5; inplace=false), "ScaleTriMesh"),
			(RotateTriMesh(ROT_MATRIX; inplace=false), "RotateTriMesh"),
			(ReAlignTriMesh(realign_trimesh(mesh); inplace=false), "RealignTriMesh"),
			(NormalizeTriMesh(inplace=false), "NormalizeTriMesh"),
			(Chain(ScaleTriMesh(0.5; inplace=false),
				   RotateTriMesh(ROT_MATRIX; inplace=false),
				   ReAlignTriMesh(realign_trimesh(mesh); inplace=false),
				   NormalizeTriMesh()), "Chain")
		]

		println("Running benchmarks for trimesh with no. of verts = $(load_trimesh(mesh).V)")
        run_benchmarks!(gpu_benchmarks, arr, mesh, (op, pc) -> (CuArrays.@sync op(pc)), gpu)
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

fname = joinpath(@__DIR__, "bm_flux3d_trimesh.txt")
save_bm(fname, cpu_benchmarks, gpu_benchmarks)
@info "Benchmarks have been saved at $fname"
