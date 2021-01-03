@testset "PointCoud transforms" begin

    for inplace in [true, false]
        @testset "ScalePointCloud inplace=$(inplace)" begin
            p = rand(Float32, 3, 8, 2) |> gpu
            t1 = ScalePointCloud(2.0; inplace = inplace) |> gpu
            t2 = ScalePointCloud(0.5; inplace = inplace) |> gpu
            pc1 = PointCloud(p) |> gpu
            pc2 = t2(t1(pc1))
            @test pc1.points isa CuArray
            @test pc2.points isa CuArray
            @test all(isapprox.(Array(p), Array(pc2.points), rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "RotatePointCloud inplace=$(inplace)" begin
            p = rand(Float32, 3, 8, 2)
            rotmat = 2 .* one(rand(Float32, 3, 3))
            rotmat_inv = inv(rotmat)

            t1 = RotatePointCloud(rotmat; inplace = inplace) |> gpu
            t2 = RotatePointCloud(rotmat_inv; inplace = inplace) |> gpu
            pc1 = PointCloud(p) |> gpu
            pc2 = t2(t1(pc1)) |> gpu
            @test t1.rotmat isa CuArray
            @test t2.rotmat isa CuArray
            @test pc1.points isa CuArray
            @test pc2.points isa CuArray
            @test all(isapprox.(Array(p), Array(pc2.points), rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "ReAlignPointCloud inplace=$(inplace)" begin
            p1 = rand(Float32, 3, 8, 2)
            p2 = rand(Float32, 3, 8, 1)
            tgt = PointCloud(p2)
            src = PointCloud(p1) |> gpu
            t = ReAlignPointCloud(tgt; inplace = inplace) |> gpu
            src_1 = t(src) |> gpu
            @test t.t_min isa CuArray
            @test t.t_max isa CuArray
            @test src.points isa CuArray
            @test src_1.points isa CuArray

            # Transformed PointCloud should be inside the bounding box defined by `tgt` PointCloud
            @test all(
                reshape(maximum(Array(tgt.points), dims = 2), (:, 1)) .>=
                Array(src_1.points) .>=
                reshape(minimum(Array(tgt.points), dims = 2), (:, 1)),
            )
        end
    end

    for inplace in [true, false]
        @testset "NormalizePointCloud inplace=$(inplace)" begin
            p = rand(Float32, 3, 8, 2)
            t = NormalizePointCloud(; inplace = inplace)
            pc1 = PointCloud(p) |> gpu
            pc2 = t(pc1) |> gpu
            @test pc1.points isa CuArray
            @test pc2.points isa CuArray
            @test all(
                isapprox.(mean(Array(pc2.points); dims = 2), 0, rtol = 1e-5, atol = 1e-5),
            )
            @test all(
                isapprox.(std(Array(pc2.points); dims = 2), 1, rtol = 1e-5, atol = 1e-5),
            )
        end
    end

    @testset "Chain" begin
        p = rand(Float32, 3, 8, 2)
        t =
            Chain(
                ScalePointCloud(0.5),
                RotatePointCloud(rand(3, 3)),
                NormalizePointCloud(),
            ) |> gpu
        pc1 = PointCloud(p) |> gpu
        pc1 = t(pc1)
        @test pc1.points isa CuArray
        @test all(isapprox.(mean(Array(pc1.points); dims = 2), 0, rtol = 1e-5, atol = 1e-5))
        @test all(isapprox.(std(Array(pc1.points); dims = 2), 1, rtol = 1e-5, atol = 1e-4))
    end

    _mesh =
        load_trimesh(
            joinpath(@__DIR__, "../assets/teapot.obj"),
            joinpath(@__DIR__, "../assets/sphere.obj"),
        ) |> gpu

    for inplace in [true, false]
        @testset "ScaleTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = ScaleTriMesh(2.0; inplace = inplace) |> gpu
            t2 = ScaleTriMesh(0.5; inplace = inplace) |> gpu
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32,2}
            @test all(
                isapprox.(
                    cpu(get_verts_packed(_mesh)),
                    cpu(get_verts_packed(m2)),
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
        end
    end

    for inplace in [true, false]
        @testset "RotateTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            rotmat = 2 .* one(rand(Float32, 3, 3))
            rotmat_inv = inv(rotmat)
            t1 = RotateTriMesh(rotmat; inplace = inplace) |> gpu
            t2 = RotateTriMesh(rotmat_inv; inplace = inplace) |> gpu
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32,2}
            @test all(
                isapprox.(
                    cpu(get_verts_packed(_mesh)),
                    cpu(get_verts_packed(m2)),
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
        end
    end

    for inplace in [true, false]
        @testset "ReAlignTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            src = Flux3D.scale(m, 2.0)
            tgt = Flux3D.scale(m, 1.0)
            t = ReAlignTriMesh(tgt, 1; inplace = inplace) |> gpu
            src_1 = t(src)
            if inplace
                @test src === src_1
            else
                @test src !== src_1
            end
            @test get_verts_packed(src_1) isa CuArray{Float32,2}
            @test all(
                maximum(get_verts_list(_mesh)[1]; dims = 2) .>=
                get_verts_list(src_1)[1] .>=
                minimum(get_verts_list(_mesh)[1]; dims = 2),
            )
            @test all(
                maximum(get_verts_list(_mesh)[1]; dims = 2) .>=
                get_verts_list(src_1)[2] .>=
                minimum(get_verts_list(_mesh)[1]; dims = 2),
            )
        end
    end

    for inplace in [true, false]
        @testset "NormalizeTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t = NormalizeTriMesh(; inplace = inplace) |> gpu
            m2 = t(m)
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32,2}
            @test all(
                isapprox.(
                    mean(cpu(get_verts_list(m2))[1]; dims = 2),
                    0.0,
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
            @test all(
                isapprox.(
                    std(cpu(get_verts_list(m2))[1]; dims = 2),
                    1.0,
                    rtol = 1e-4,
                    atol = 1e-5,
                ),
            )
            @test all(
                isapprox.(
                    mean(cpu(get_verts_list(m2))[2]; dims = 2),
                    0.0,
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
            @test all(
                isapprox.(
                    std(cpu(get_verts_list(m2))[2]; dims = 2),
                    1.0,
                    rtol = 1e-4,
                    atol = 1e-5,
                ),
            )
        end
    end

    for inplace in [true, false]
        @testset "TranslateTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = TranslateTriMesh(1.0; inplace = inplace) |> gpu
            t2 = TranslateTriMesh(-1.0; inplace = inplace) |> gpu
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32,2}
            @test all(
                isapprox.(
                    cpu(get_verts_packed(_mesh)),
                    cpu(get_verts_packed(m2)),
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
        end
    end

    for inplace in [true, false]
        @testset "OffsetTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            _offset = ones(size(get_verts_packed(m)))
            t1 = OffsetTriMesh(_offset; inplace = inplace) |> gpu
            t2 = OffsetTriMesh(_offset .* -1; inplace = inplace) |> gpu
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32,2}
            @test all(
                isapprox.(
                    cpu(get_verts_packed(_mesh)),
                    cpu(get_verts_packed(m2)),
                    rtol = 1e-5,
                    atol = 1e-5,
                ),
            )
        end
    end

    _voxels = zeros(Float32, 32, 32, 32, 2) |> gpu
    _voxels[1:15, 2:10, 18:32, :] .= 1

    _v = VoxelGrid(_voxels) |> gpu
    _m =
        load_trimesh(
            joinpath(@__DIR__, "../assets/teapot.obj"),
            joinpath(@__DIR__, "../assets/sphere.obj"),
        ) |> gpu
    _p = PointCloud(sample_points(_m, 1024)) |> gpu

    res = 28
    points = 512
    thresh = 0.9

    @testset "TriMeshToVoxelGrid" begin
        m = deepcopy(_m)
        t = TriMeshToVoxelGrid(res)
        v = t(m)
        @test v isa VoxelGrid{Float32}
        @test v.voxels isa CUDA.CuArray
        @test size(v.voxels) == (res, res, res, 2)
    end

    @testset "PointCloudToVoxelGrid" begin
        p = deepcopy(_p)
        t = PointCloudToVoxelGrid(res)
        v = t(p)
        @test v isa VoxelGrid{Float32}
        @test v.voxels isa CUDA.CuArray
        @test size(v.voxels) == (res, res, res, 2)
    end

    for algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets]
        @testset "VoxelGridToTriMesh algo=$(algo)" begin
            v = deepcopy(_v)
            t = VoxelGridToTriMesh(thresh = thresh, algo = algo)
            m = t(v)
            @test m isa TriMesh{Float32,UInt32,CuArray}
        end
    end

    for algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets]
        @testset "PointCloudToTriMesh algo=$(algo)" begin
            p = deepcopy(_p)
            t = PointCloudToTriMesh(res, algo = algo)
            m = t(p)
            @test m isa TriMesh{Float32,UInt32,CuArray}
        end
    end

    @testset "TriMeshToPointCloud" begin
        m = deepcopy(_m)
        t = TriMeshToPointCloud(points)
        p = t(m)
        @test p isa PointCloud{Float32}
        @test p.points isa CUDA.CuArray
        @test size(p.points) == (3, points, 2)
    end

    for algo in [:Exact, :MarchingCubes, :MarchingTetrahedra]#, :NaiveSurfaceNets]
        @testset "VoxelGridToPointCloud algo=$(algo)" begin
            v = deepcopy(_v)
            t = VoxelGridToPointCloud(points, thresh = thresh, algo = algo)
            p = t(v)
            @test p isa PointCloud{Float32}
            @test p.points isa CUDA.CuArray
            @test size(p.points) == (3, points, 2)
        end
    end
end # Transforms
