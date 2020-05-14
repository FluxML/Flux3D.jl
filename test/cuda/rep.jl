using Flux3D
using CuArrays

CuArrays.allowscalar(false)

@testset "Representation" begin

for (T) in [(Float32),(Float64)]
        
    points = rand(T,64,3)
    _normals = rand(T,64,3)

    for normals in [nothing, _normals]

        rep = PointCloud(points, normals)
        crep = gpu(rep)
        
        if normals === nothing
            @test crep.normals isa Nothing
        else
            @test size(crep.normals) == size(_normals)
            @test crep.normals isa CuArray
            @test crep.normals isa CuArray{Float32,2}
        end
        @test npoints(crep) == size(points,1)
        @test size(crep.points) == size(points)
        @test crep.points isa CuArray
        @test crep.points isa CuArray{Float32,2}
    end
end

end # testset Representation