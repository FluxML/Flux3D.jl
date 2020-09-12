@info "Testing pcloud_func..."
@testset "Transforms pcloud_func" begin

    for (inplace, FUNC) in [(true, Flux3D.normalize!), (false, Flux3D.normalize)]
        @testset "$(FUNC) function" begin
            p = rand(Float32, 3, 8, 2)
            pc1 = PointCloud(p)
            pc2 = FUNC(pc1)
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(mean(pc2.points; dims = 2), 0, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(pc2.points; dims = 2), 1, rtol = 1e-5, atol = 1e-5))
        end
    end

    for (inplace, FUNC) in [(true, Flux3D.scale!), (false, Flux3D.scale)]
        @testset "$(FUNC) function" begin
            p = rand(Float32, 3, 8, 2)
            pc1 = PointCloud(p)
            pc2 = FUNC(FUNC(pc1, 2.0), 0.5)
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(p, pc2.points, rtol = 1e-5, atol = 1e-5))
        end
    end

    for (inplace, FUNC) in [(true, Flux3D.rotate!), (false, Flux3D.rotate)]
        @testset "$(FUNC) function" begin
            p = rand(Float32, 3, 8, 2)
            rotmat = 2 .* one(rand(Float32, 3, 3))
            rotmat_inv = inv(rotmat)
            rotmat_b = cat(rotmat, rotmat, dims=3)
            rotmat_inv_b = cat(rotmat_inv, rotmat_inv, dims=3)
            pc1 = PointCloud(p)
            pc2 = FUNC(FUNC(pc1, rotmat), rotmat_inv)
            pc3 = FUNC(FUNC(pc1, rotmat_b), rotmat_inv_b)
            if inplace
                @test pc1 == pc2
                @test pc1 == pc3
            else
                @test pc1 != pc2
                @test pc1 != pc3
            end
            @test all(isapprox.(p, pc2.points, rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(p, pc3.points, rtol = 1e-5, atol = 1e-5))
        end
    end

    for (inplace, FUNC) in [(true, Flux3D.realign!), (false, Flux3D.realign)]
        @testset "$(FUNC) function" begin
            p1 = rand(Float32, 8, 3, 2)
            p2 = rand(Float32, 8, 3, 1)
            src = PointCloud(p1)
            tgt = PointCloud(p2)
            src_1 = FUNC(src, tgt)
            if inplace
                @test src == src_1
            else
                @test src != src_1
            end
            # Transformed PointCloud should be inside the bounding box defined by `tgt` PointCloud
            @test all(
                reshape(maximum(tgt.points, dims = 2), (:, 1)) .>=
                src_1.points .>=
                reshape(minimum(tgt.points, dims = 2), (:, 1)),
            )
        end
    end
end
