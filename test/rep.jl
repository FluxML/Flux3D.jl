@info "Testing PointCloud..."
@testset "PointCloud" begin
    for T = [Float32,Float64], _N = [true, false]
        points = rand(T,3,16,4)
        _normals = rand(T,3,16,4)
        if _N
            normals = _normals
        else
            normals = nothing
        end
        rep = PointCloud(points, normals)
        if normals == nothing
            @test rep.normals isa Nothing
        else
            @test size(rep.normals) == size(_normals)
            @test rep.normals isa Array{Float32,3}
        end
        @test npoints(rep) == size(points,2)
        @test size(rep.points) == size(points)
        @test rep.points isa Array{Float32,3}

        for i in 1:size(points,3)
            @test rep[i] == points[:,:,i]
        end

        # using other contructor
        rep1 = PointCloud(points=points, normals=normals)
        rep2 = PointCloud(rep)
        @test rep.points == rep1.points
        @test rep.points == rep2.points
        @test rep.normals == rep1.normals
        @test rep.normals == rep2.normals
    end
end
