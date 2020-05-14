@testset "PointCoud transforms" begin

    for inplace in [true, false]
        @testset "ScalePointCloud inplace=$(inplace)" begin
            p = rand(Float32,32,3)
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
            p = rand(Float32,32,3)
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
            p1 = rand(Float32,32,3)
            p2 = rand(Float32,32,3)
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
                reshape(maximum(tgt.points, dims=1), (1,:)) .>= src_1.points .>= reshape(minimum(tgt.points, dims=1), (1,:)))
        end
    end

    for inplace in [true, false]
        @testset "NormalizePointCloud inplace=$(inplace)" begin
            p = rand(Float32,32,3)
            t = NormalizePointCloud(; inplace=inplace)
            pc1 = PointCloud(p)
            pc2 = t(pc1)
            if inplace
                @test pc1 == pc2
            else
                @test pc1 != pc2
            end
            @test all(isapprox.(mean(pc2.points;dims=1), zeros(Float32, 1, size(p,2)), rtol = 1e-5, atol = 1e-5))
            @test all(isapprox.(std(pc2.points;dims=1), ones(Float32, 1, size(p,2)), rtol = 1e-5, atol = 1e-5))
        end
    end

    @testset "Compose" begin
        p = rand(Float32, 32, 3)
        t = Compose(ScalePointCloud(0.5), RotatePointCloud(rand(3,3)), NormalizePointCloud())
        pc1 = PointCloud(p)
        pc1 = t(pc1)
        @test pc1.points isa Array
        @test all(isapprox.(mean(pc1.points;dims=1), zeros(Float32, 1, size(p,2)), rtol = 1e-5, atol = 1e-5))
        # @test all(isapprox.(std(pc1.points;dims=1), ones(Float32, 1, size(p,2)), rtol = 1e-5, atol = 1e-5))
    end

end # PointCloud transforms