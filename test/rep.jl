@info "Starting PointCloud tests..."
@testset "PointCloud" begin
    for (T) in [
            (Float32),
            (Float64),
            (Int64),
        ]
        points = rand(T,1024,3)
        rep = PointCloud(points)

        @test rep.npoints == 1024
        @test size(rep.points) == (1024,3)
        @test rep.points isa Array{Float32,2}
    end
end