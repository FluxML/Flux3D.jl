using Flux3D, BenchmarkTools, Printf

function setup_benchmark_record(names)
    benchmarks = Dict{String,Vector{Float64}}()
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

function generate_trimesh(npoints::Int)
    v = ones(3, npoints)
    v = cumsum(v, dims = 2) / npoints
    f = reshape(collect(1:npoints*3), 3, npoints)
    return TriMesh([v], [f])
end

function run_benchmarks!(benchmarks, x, npoints, generate_func, benchmark_func, device)
    for (transform, name) in x
        transform = transform |> device
        pc = generate_func(npoints) |> device
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

function realign_trimesh(_npoints)
    pc = generate_trimesh(_npoints)
    rot = RotateTriMesh(-ROT_MATRIX)
    return rot(pc)
end

ROT_MATRIX = [
    1.0 2.0 3.0
    0.2 0.5 0.9
    3.0 2.0 1.0
]

npoint_arr = 2 .^ [12, 14, 16, 18, 20]

names = [
    "ScalePointCloud",
    "RotatePointCloud",
    "ReAlignPointCloud",
    "NormalizePointCloud",
    "Chain",
]
cpu_bm_pcloud = setup_benchmark_record(names)
gpu_bm_pcloud = setup_benchmark_record(names)

names = ["ScaleTriMesh", "RotateTriMesh", "ReAlignTriMesh", "NormalizeTriMesh", "Chain"]
cpu_bm_trimesh = setup_benchmark_record(names)
gpu_bm_trimesh = setup_benchmark_record(names)


println("DEVICE: CPU")
for _npoints in npoint_arr
    pcloud_arr = [
        (ScalePointCloud(0.5; inplace = false), "ScalePointCloud"),
        (RotatePointCloud(ROT_MATRIX; inplace = false), "RotatePointCloud"),
        (
            ReAlignPointCloud(realign_point_cloud(_npoints); inplace = false),
            "ReAlignPointCloud",
        ),
        (NormalizePointCloud(inplace = false), "NormalizePointCloud"),
        (
            Chain(
                ScalePointCloud(0.5; inplace = false),
                RotatePointCloud(ROT_MATRIX; inplace = false),
                ReAlignPointCloud(realign_point_cloud(_npoints); inplace = false),
                NormalizePointCloud(),
            ),
            "Chain",
        ),
    ]

    trimesh_arr = [
        (ScaleTriMesh(0.5; inplace = true), "ScaleTriMesh"),
        (RotateTriMesh(ROT_MATRIX; inplace = true), "RotateTriMesh"),
        (ReAlignTriMesh(realign_trimesh(_npoints); inplace = true), "ReAlignTriMesh"),
        (NormalizeTriMesh(inplace = true), "NormalizeTriMesh"),
        (
            Chain(
                ScaleTriMesh(0.5; inplace = true),
                RotateTriMesh(ROT_MATRIX; inplace = true),
                ReAlignTriMesh(realign_trimesh(_npoints); inplace = true),
                NormalizeTriMesh(inplace = true),
            ),
            "Chain",
        ),
    ]
    println("Running benchmarks for npoints = $_npoints")
    run_benchmarks!(
        cpu_bm_pcloud,
        pcloud_arr,
        _npoints,
        generate_point_cloud,
        (op, pc) -> op(pc),
        cpu,
    )
    run_benchmarks!(
        cpu_bm_trimesh,
        trimesh_arr,
        _npoints,
        generate_trimesh,
        (op, pc) -> op(pc),
        cpu,
    )
    println()
end

using CUDA
if has_cuda()
    println("CUDA is on. Running GPU Benchmarks")
    println("DEVICE: GPU")
    for _npoints in npoint_arr
        pcloud_arr = [
            (ScalePointCloud(0.5; inplace = false), "ScalePointCloud"),
            (RotatePointCloud(ROT_MATRIX; inplace = false), "RotatePointCloud"),
            (
                ReAlignPointCloud(realign_point_cloud(_npoints); inplace = false),
                "ReAlignPointCloud",
            ),
            (NormalizePointCloud(inplace = false), "NormalizePointCloud"),
            (
                Chain(
                    ScalePointCloud(0.5; inplace = false),
                    RotatePointCloud(ROT_MATRIX; inplace = false),
                    ReAlignPointCloud(realign_point_cloud(_npoints); inplace = false),
                    NormalizePointCloud(),
                ),
                "Chain",
            ),
        ]

        trimesh_arr = [
            (ScaleTriMesh(0.5; inplace = true), "ScaleTriMesh"),
            (RotateTriMesh(ROT_MATRIX; inplace = true), "RotateTriMesh"),
            (ReAlignTriMesh(realign_trimesh(_npoints); inplace = true), "ReAlignTriMesh"),
            (NormalizeTriMesh(inplace = true), "NormalizeTriMesh"),
            (
                Chain(
                    ScaleTriMesh(0.5; inplace = true),
                    RotateTriMesh(ROT_MATRIX; inplace = true),
                    ReAlignTriMesh(realign_trimesh(_npoints); inplace = true),
                    NormalizeTriMesh(inplace = true),
                ),
                "Chain",
            ),
        ]
        println("Running benchmarks for npoints = $_npoints")
        run_benchmarks!(
            gpu_bm_pcloud,
            pcloud_arr,
            _npoints,
            generate_point_cloud,
            (op, pc) -> (CUDA.@sync op(pc)),
            gpu,
        )
        run_benchmarks!(
            gpu_bm_trimesh,
            trimesh_arr,
            _npoints,
            generate_trimesh,
            (op, pc) -> (CUDA.@sync op(pc)),
            gpu,
        )
        println()
    end
end

function save_bm(fname, rep, cpu_benchmarks, gpu_benchmarks)
    open(fname, "w") do io
        device = "cpu"
        for (key, values) in cpu_benchmarks
            for (p, v) in zip(npoint_arr, values)
                Printf.@printf(io, "%s %s %s %d %f ms\n", rep, device, key, p, v)
            end
        end

        device = "gpu"
        for (key, values) in gpu_benchmarks
            for (p, v) in zip(npoint_arr, values)
                Printf.@printf(io, "%s %s %s %d %f ms\n", rep, device, key, p, v)
            end
        end
    end
end

fname = joinpath(@__DIR__, "bm_flux3d.txt")
save_bm(fname, "PointCloud", cpu_bm_pcloud, gpu_bm_pcloud)
save_bm(fname, "TriMesh", cpu_bm_trimesh, gpu_bm_trimesh)
@info "Benchmarks have been saved at $fname"
