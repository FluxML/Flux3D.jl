@testset "Testing Transforms mesh_func..." begin

    @testset "sample_points" begin
        m = load_trimesh([
            joinpath(@__DIR__, "../meshes/sphere.obj"),
            joinpath(@__DIR__, "../meshes/sphere.obj"),
        ])

        samples = sample_points(m, 1000)
        _radius = sqrt.(sum(samples.^2; dims=2))
        @test all(isapprox.(
            _radius,
            1.0,
            rtol = 1e-2,
            atol = 1e-5,
        ))
    end

    _mesh = load_trimesh([
        joinpath(@__DIR__, "../meshes/teapot.obj"),
        joinpath(@__DIR__, "../meshes/sphere.obj"),
    ])

    for (inplace, FUNC) in [(true, Flux3D.normalize!), (false, Flux3D.normalize)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            m2 = FUNC(m)
            if inplace
                @test m2 === m
            else
                @test m2 !== m
            end
            @test all(isapprox.(
                mean(get_verts_list(m2)[1]; dims = 1),
                0.0,
                rtol = 1e-5,
                atol = 1e-5,
            ))
            @test all(isapprox.(
                std(get_verts_list(m2)[1]; dims = 1),
                1.0,
                rtol = 1e-5,
                atol = 1e-5,
            ))
            @test all(isapprox.(
                mean(get_verts_list(m2)[2]; dims = 1),
                0.0,
                rtol = 1e-5,
                atol = 1e-5,
            ))
            @test all(isapprox.(
                std(get_verts_list(m2)[1]; dims = 1),
                1.0,
                rtol = 1e-5,
                atol = 1e-5,
            ))
        end
    end

    for (inplace, FUNC) in [(true, Flux3D.scale!), (false, Flux3D.scale)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            m2 = FUNC(FUNC(m, 2.0), 0.5)
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

    for (inplace, FUNC) in [(true, Flux3D.rotate!), (false, Flux3D.rotate)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            rotmat = 2 .* one(rand(Float32, 3, 3))
            rotmat_inv = inv(rotmat)
            m2 = FUNC(FUNC(m, rotmat), rotmat_inv)
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

    for (inplace, FUNC) in [(true, Flux3D.realign!), (false, Flux3D.realign)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            src = Flux3D.scale(m, 2.0)
            tgt = Flux3D.scale(m, 1.0)
            src_1 = FUNC(src, tgt, 1)
            if inplace
                @test src === src_1
            else
                @test src !== src_1
            end
            # Transformed PointCloud should be inside the bounding box defined by `tgt` PointCloud
            @test all(
                reshape(maximum(get_verts_list(_mesh)[1]; dims = 1), (1, :)) .>=
                get_verts_list(src_1)[1] .>=
                reshape(minimum(get_verts_list(_mesh)[1]; dims = 1), (1, :)),
            )
            src_1 = FUNC(src, tgt, 2)
            @test all(
                reshape(maximum(get_verts_list(_mesh)[2]; dims = 1), (1, :)) .>=
                get_verts_list(src_1)[2] .>=
                reshape(minimum(get_verts_list(_mesh)[2]; dims = 1), (1, :)),
            )
        end
    end

    for (inplace, FUNC) in [(true, Flux3D.translate!), (false, Flux3D.translate)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            m2 = FUNC(FUNC(m, 1.0), -1.0)
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

    for (inplace, FUNC) in [(true, Flux3D.offset!), (false, Flux3D.offset)]
        @testset "$(FUNC)" begin
            m = deepcopy(_mesh)
            _offset = ones(size(get_verts_packed(m)))
            m2 = FUNC(FUNC(m, _offset), (_offset.*-1))
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
end
