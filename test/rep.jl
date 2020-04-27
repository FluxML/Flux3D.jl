@info "Testing PointCloud..."
@testset "PointCloud" begin
    for (T) in [
            (Float32),
            (Float64),
            (Int64),
        ]
        points = rand(T,64,3)
        _normals = rand(T,64,3)
        for normals in [nothing, _normals]
            rep = PointCloud(points, normals)
            if normals == nothing
                @test rep.normals isa Nothing
            else
                @test size(rep.normals) == size(_normals)
                @test rep.normals isa Array{Float32,2}
            end
            @test npoints(rep) == size(points,1)
            @test size(rep.points) == size(points)
            @test rep.points isa Array{Float32,2}
        end
    end
end