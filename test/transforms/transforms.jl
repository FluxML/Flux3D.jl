@testset "Testing Transforms..." begin

    for inplace in [true, false]
        @testset "ScalePointCloud inplace=$(inplace)" begin
            p = rand(Float32,3,8,2)
            t1 = ScalePointCloud(2.0; inplace=inplace)
            t2 = ScalePointCloud(0.5; inplace=inplace)
            pc1 = PointCloud(p)
            pc2 = t2(t1(pc1))
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(p, pc2.points, rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "RotatePointCloud inplace=$(inplace)" begin
            p = rand(Float32,3,8,2)
            rotmat = 2 .* one(rand(Float32,3,3))
            rotmat_inv = inv(rotmat)

            t1 = RotatePointCloud(rotmat; inplace=inplace)
            t2 = RotatePointCloud(rotmat_inv; inplace=inplace)
            pc1 = PointCloud(p)
            pc2 = t2(t1(pc1))
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(p, pc2.points, rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "ReAlignPointCloud inplace=$(inplace)" begin
            p1 = rand(Float32,3,8,2)
            p2 = rand(Float32,3,8,1)
            src = PointCloud(p1)
            tgt = PointCloud(p2)
            t = ReAlignPointCloud(tgt; inplace=inplace)
            src_1 = t(src)
            if inplace
                @test src == src_1
            else
                @test src != src_1
            end
            # Transformed PointCloud should be inside the bounding box defined by `tgt` PointCloud
            @test all(
                reshape(maximum(tgt.points, dims=2), (:,1)) .>= src_1.points .>= reshape(minimum(tgt.points, dims=2), (:,1)))
        end
    end

    for inplace in [true, false]
        @testset "NormalizePointCloud inplace=$(inplace)" begin
            p = rand(Float32,3,8,2)
            t = NormalizePointCloud(; inplace=inplace)
            pc1 = PointCloud(p)
            pc2 = t(pc1)
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(mean(pc2.points;dims=2), 0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(pc2.points;dims=2), 1, rtol = 1e-5, atol = 1e-5))
        end
    end

    @testset "Chain" begin
        p = rand(Float32, 3, 8, 2)
        t = Chain(ScalePointCloud(0.5), RotatePointCloud(rand(3,3)), NormalizePointCloud())
        pc1 = PointCloud(p)
        pc1 = t(pc1)
        @test pc1.points isa Array
        @test all(isapprox.(mean(pc1.points;dims=2), 0, rtol = 1e-5, atol = 1e-5))
        @test all(isapprox.(std(pc1.points;dims=2), 1, rtol = 1e-5, atol = 1e-4))
    end

    _mesh = load_trimesh([
        joinpath(@__DIR__, "../meshes/teapot.obj"),
        joinpath(@__DIR__, "../meshes/sphere.obj"),
    ])

    for inplace in [true, false]
        @testset "ScaleTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = ScaleTriMesh(2.0; inplace=inplace)
            t2 = ScaleTriMesh(0.5; inplace=inplace)
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(
                get_verts_packed(_mesh),
                get_verts_packed(m2),
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
            t1 = RotateTriMesh(rotmat; inplace=inplace)
            t2 = RotateTriMesh(rotmat_inv; inplace=inplace)
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(
                get_verts_packed(_mesh),
                get_verts_packed(m2),
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
            t = ReAlignTriMesh(tgt, 1; inplace=inplace)
            src_1 = t(src)
            if inplace
                @test src === src_1
            else
                @test src !== src_1
            end
            src_1 = t(src)
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
            t = NormalizeTriMesh(; inplace=inplace)
            m2 = t(m)
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(mean(get_verts_list(m2)[1]; dims = 1), 0.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(get_verts_list(m2)[1]; dims = 1), 1.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(mean(get_verts_list(m2)[2]; dims = 1), 0.0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(get_verts_list(m2)[2]; dims = 1), 1.0, rtol = 1e-5, atol = 1e-5))
        end
    end

    for inplace in [true, false]
        @testset "TranslateTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            t1 = TranslateTriMesh(1.0; inplace=inplace)
            t2 = TranslateTriMesh(-1.0; inplace=inplace)
            m2 = t2(t1(m))
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(
                get_verts_packed(_mesh),
                get_verts_packed(m2),
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end

    for inplace in [true, false]
        @testset "OffsetTriMesh inplace=$(inplace)" begin
            m = deepcopy(_mesh)
            _offset = ones(size(get_verts_packed(m)))
            t1 = OffsetTriMesh(_offset; inplace=inplace)
            t2 = OffsetTriMesh(_offset .* -1; inplace=inplace)
            m2 = t2(t1(m))

            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(
                get_verts_packed(_mesh),
                get_verts_packed(m2),
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end
end # transforms
