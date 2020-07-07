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
                reshape(maximum(
                    Array(tgt.points), dims = 2), (:, 1)) .>=
                    Array(src_1.points) .>=
                    reshape(minimum(Array(tgt.points), dims = 2), (:, 1)))
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
            @test all(isapprox.(mean(Array(pc2.points);dims = 2), 0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(Array(pc2.points);dims = 2), 1, rtol = 1e-5, atol = 1e-5))
        end
    end

    @testset "Chain" begin
        p = rand(Float32, 3, 8, 2)
        t = Chain(ScalePointCloud(0.5), RotatePointCloud(rand(3,3)), NormalizePointCloud()) |> gpu
        pc1 = PointCloud(p) |> gpu
        pc1 = t(pc1)
        @test pc1.points isa CuArray
        @test all(isapprox.(mean(Array(pc1.points);dims = 2), 0, rtol = 1e-5, atol = 1e-5))
        @test all(isapprox.(std(Array(pc1.points);dims = 2), 1, rtol = 1e-5, atol = 1e-4))
    end

    _mesh = load_trimesh([
        joinpath(@__DIR__, "../meshes/teapot.obj"),
        joinpath(@__DIR__, "../meshes/sphere.obj"),
    ]) |> gpu

    for inplace in [true, false]
        @testset "ScaleTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = ScaleTriMesh(2.0; inplace=inplace) |> gpu
            t2 = ScaleTriMesh(0.5; inplace=inplace) |> gpu
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32, 2}
            @test all(isapprox.(
                cpu(get_verts_packed(_mesh)),
                cpu(get_verts_packed(m2)),
                rtol = 1e-5,
                atol = 1e-5,
                ))
        end
    end

    for inplace in [true, false]
        @testset "RotateTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            rotmat = 2 .* one(rand(Float32, 3, 3))
            rotmat_inv = inv(rotmat)
            t1 = RotateTriMesh(rotmat; inplace=inplace) |> gpu
            t2 = RotateTriMesh(rotmat_inv; inplace=inplace) |> gpu
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32, 2}
            @test all(isapprox.(
                cpu(get_verts_packed(_mesh)),
                cpu(get_verts_packed(m2)),
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end

    for inplace in [true, false]
        @testset "ReAlignTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            src = Flux3D.scale(m, 2.0)
            tgt = Flux3D.scale(m, 1.0)
            t = ReAlignTriMesh(tgt, 1; inplace=inplace) |> gpu
            src_1 = t(src)
            if inplace
                @test src === src_1
            else
                @test src !== src_1
            end
            @test get_verts_packed(src_1) isa CuArray{Float32, 2}
            @test all(
                reshape(maximum(get_verts_list(_mesh)[1]; dims = 1), (1, :)) .>=
                get_verts_list(src_1)[1] .>=
                reshape(minimum(get_verts_list(_mesh)[1]; dims = 1), (1, :)),
            )
            @test all(
                reshape(maximum(get_verts_list(_mesh)[1]; dims = 1), (1, :)) .>=
                get_verts_list(src_1)[2] .>=
                reshape(minimum(get_verts_list(_mesh)[1]; dims = 1), (1, :)),
            )
        end
    end

    for inplace in [true, false]
        @testset "NormalizeTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t = NormalizeTriMesh(; inplace=inplace) |> gpu
            m2 = t(m)
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32, 2}
            @test all(isapprox.(mean(cpu(get_verts_list(m2))[1]; dims = 1), 0.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(cpu(get_verts_list(m2))[1]; dims = 1), 1.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(mean(cpu(get_verts_list(m2))[2]; dims = 1), 0.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(cpu(get_verts_list(m2))[2]; dims = 1), 1.0, rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "TranslateTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = TranslateTriMesh(1.0; inplace=inplace) |> gpu
            t2 = TranslateTriMesh(-1.0; inplace=inplace) |> gpu
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32, 2}
            @test all(isapprox.(
                cpu(get_verts_packed(_mesh)),
                cpu(get_verts_packed(m2)),
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end

    for inplace in [true, false]
        @testset "OffsetTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            _offset = ones(size(get_verts_packed(m)))
            t1 = OffsetTriMesh(_offset; inplace=inplace) |> gpu
            t2 = OffsetTriMesh(_offset .* -1; inplace=inplace) |> gpu
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test get_verts_packed(m2) isa CuArray{Float32, 2}
            @test all(isapprox.(
                cpu(get_verts_packed(_mesh)),
                cpu(get_verts_packed(m2)),
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end
end # Transforms
