@testset "Representation" begin
    for T = [Float32,Float64], _N = [true, false]
        points = rand(T,3,16,4)
        _normals = rand(T,3,16,4)
        if _N
            normals = _normals
        else
            normals = nothing
        end
        rep = PointCloud(points, normals)
        crep = gpu(rep)

        if normals === nothing
            @test crep.normals isa Nothing
        else
            @test size(crep.normals) == size(_normals)
            @test crep.normals isa CuArray{Float32,3}
        end
        @test npoints(crep) == size(points,2)
        @test size(crep.points) == size(points)
        @test crep.points isa CuArray{Float32,3}
    end
end # testset Representation
