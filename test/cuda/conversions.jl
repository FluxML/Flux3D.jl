@info "Testing Conversions..."
@testset "Conversions Utilities" begin
    _voxels = zeros(Float32, 32, 32, 32, 2) |> gpu
    _voxels[1:15, 2:10, 18:32, :] .= 1

    _v = VoxelGrid(_voxels) |> gpu
    _m =
        load_trimesh(
            joinpath(@__DIR__, "../assets/teapot.obj"),
            joinpath(@__DIR__, "../assets/sphere.obj"),
        ) |> gpu
    _p = PointCloud(sample_points(_m, 1024)) |> gpu

    res = 28
    points = 512
    thresh = 0.9

    for algo in [:Exact, :MarchingCubes, :MarchingTetrahedra]
        m1 = TriMesh(_p, res; algo = algo)
        @test m1 isa TriMesh{Float32,UInt32,CuArray}

        m2 = TriMesh(_v, thresh = thresh, algo = algo)
        @test m2 isa TriMesh{Float32,UInt32,CuArray}

        p1 = PointCloud(_v, points, thresh = thresh, algo = algo)
        @test p1 isa PointCloud{Float32}
        @test p1.points isa CUDA.CuArray
        @test size(p1.points) == (3, points, 2)
    end

    p2 = PointCloud(_m, points)
    @test p2 isa PointCloud{Float32}
    @test size(p2.points) == (3, points, 2)
    @test p2.points isa CUDA.CuArray

    v1 = VoxelGrid(_m, res)
    @test v1 isa VoxelGrid{Float32}
    @test v1.voxels isa CUDA.CuArray
    @test size(v1.voxels) == (res, res, res, 2)

    v2 = VoxelGrid(_p, res)
    @test v2 isa VoxelGrid{Float32}
    @test v2.voxels isa CUDA.CuArray
    @test size(v2.voxels) == (res, res, res, 2)
end
