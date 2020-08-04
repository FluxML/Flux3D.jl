using Flux3D, Zygote, BenchmarkTools, Printf

function setup_benchmark_record(names)
    benchmarks = Dict{String,Any}()
    for name in names
        benchmarks[name] = []
    end
    return benchmarks
end

function generate_pcloud(npoints::Int)
    points = ones(Float32, 3, npoints)
    points = cumsum(points, dims = 2) / npoints
    return points
end

function generate_trimesh(npoints::Int)
    v = ones(3, npoints)
    v = cumsum(v, dims = 2) / npoints
    f = reshape(repeat(collect(1:npoints), 3), 3, npoints)
    return TriMesh([Float32.(v)], [Int32.(f)])
end

function run_benchmarks!(benchmarks, x, benchmark_func, device)
    for (func, args, name) in x
        args = args .|> device
        trial_f = @benchmark $benchmark_func($func, $args)
        back_args = [x -> func(x...), args]
        trial_t = @benchmark $benchmark_func($gradient, $back_args)
        time_f = minimum(trial_f.times) * 1.0e-6
        time_t = minimum(trial_t.times) * 1.0e-6
        time_b = time_t - time_f
        println("$name (forward): $time_f ms")
        println("$name (back): $time_b ms")
        println("$name (total): $time_t ms")
        push!(benchmarks[name], [time_f, time_b, time_t])
    end
end

npoint_arr = 2 .^ [6, 8, 10, 12, 14]

names = ["sample_points", "chamfer_distance", "edge_loss", "laplacian_loss"]

cpu_bm = setup_benchmark_record(names)
gpu_bm = setup_benchmark_record(names)

println("DEVICE: CPU")
for _npoints in npoint_arr
    arr = [
        # (sample_points, [generate_trimesh(_npoints), _npoints],
        #  "sample_points"),
        (
            chamfer_distance,
            [generate_pcloud(_npoints), generate_pcloud(_npoints)],
            "chamfer_distance",
        ),
        (edge_loss, [generate_trimesh(_npoints)], "edge_loss"),
        (laplacian_loss, [generate_trimesh(_npoints)], "laplacian_loss"),
    ]
    println("Running benchmarks for npoints = $_npoints")
    run_benchmarks!(cpu_bm, arr, (op, pc) -> op(pc...), cpu)
    println()
end

using CUDA
if has_cuda()
    println("CUDA is on. Running GPU Benchmarks")
    println("DEVICE: GPU")
    for _npoints in npoint_arr
        arr = [
            # (sample_points, [generate_trimesh(_npoints), _npoints],
            #  "sample_points"),
            (
                chamfer_distance,
                [generate_pcloud(_npoints), generate_pcloud(_npoints)],
                "chamfer_distance",
            ),
            (edge_loss, [generate_trimesh(_npoints)], "edge_loss"),
            (laplacian_loss, [generate_trimesh(_npoints)], "laplacian_loss"),
        ]
        println("Running benchmarks for npoints = $_npoints")
        run_benchmarks!(gpu_bm, arr, (op, pc) -> (CUDA.@sync op(pc...)), gpu)
        println()
    end
end

function save_bm(fname, rep, cpu_benchmarks, gpu_benchmarks)
    open(fname, "w") do io
        device = "cpu"
        for (key, values) in cpu_benchmarks
            for (p, v) in zip(npoint_arr, values)
                Printf.@printf(
                    io,
                    "Flux3D(Forward) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[1]
                )
                Printf.@printf(
                    io,
                    "Flux3D(Backward) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[2]
                )
                Printf.@printf(
                    io,
                    "Flux3D(Total) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[3]
                )
            end
        end

        device = "gpu"
        for (key, values) in gpu_benchmarks
            for (p, v) in zip(npoint_arr, values)
                Printf.@printf(
                    io,
                    "Flux3D(Forward) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[1]
                )
                Printf.@printf(
                    io,
                    "Flux3D(Backward) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[2]
                )
                Printf.@printf(
                    io,
                    "Flux3D(Total) %s %s %s %d %f ms\n",
                    rep,
                    device,
                    key,
                    p,
                    v[3]
                )
            end
        end
    end
end

fname = joinpath(@__DIR__, "bm_flux3d_metrics.txt")
save_bm(fname, "Metrics", cpu_bm, gpu_bm)
@info "Benchmarks have been saved at $fname"
