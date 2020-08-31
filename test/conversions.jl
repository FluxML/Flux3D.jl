@testset "Conversions Utilities" begin
    _voxels = zeros(Float32, 32, 32, 32, 2)
    _voxels[1:15, 2:10, 18:32, :] .= 1

    _v = VoxelGrid(_voxels)
    _m = load_trimesh([
        joinpath(@__DIR__, "./assets/teapot.obj"),
        joinpath(@__DIR__, "./assets/sphere.obj"),
    ])
    _p = PointCloud(sample_points(_m, 1024))

    res = 28
    points = 512
    thresh = 0.9

    for algo in [:Exact, :MarchingCubes, :MarchingTetrahedra]
        m1 = TriMesh(_p, res; algo = algo)
        @test m1 isa TriMesh{Float32,UInt32,Array}

        m2 = TriMesh(_v, thresh = thresh, algo = algo)
        @test m2 isa TriMesh{Float32,UInt32,Array}

        p1 = PointCloud(_v, points, thresh = thresh, algo = algo)
        @test p1 isa PointCloud{Float32}
        @test size(p1.points) == (3, points, 2)
    end

    p2 = PointCloud(_m, points)
    @test p2 isa PointCloud{Float32}
    @test size(p2.points) == (3, points, 2)

    v1 = VoxelGrid(_m, res)
    @test v1 isa VoxelGrid{Float32}
    @test size(v1.voxels) == (res, res, res, 2)

    v2 = VoxelGrid(_p, res)
    @test v2 isa VoxelGrid{Float32}
    @test size(v2.voxels) == (res, res, res, 2)

end

# Validating the geometry of each figures is not possible, so use below code to
# to cross check the dimension and figure of different conversion.

# using Makie
# Makie.AbstractPlotting.inline!(true)
# Makie.AbstractPlotting.set_theme!(show_axis = true)
#
# _voxels = zeros(Float32,32,32,32,2)
# _voxels[1:15,2:10,18:32,:] .= 1
#
# _v = VoxelGrid(_voxels)
# _m = load_trimesh(["assets/teapot.obj","assets/sphere.obj"])
# _p = PointCloud(sample_points(_m,1024))
#
# visualize(_v,1,algo=:Exact)
# visualize(_v,2,algo=:MarchingCubes)
#
# m1 = TriMesh(_v, algo=:MarchingCubes)
# visualize(m1,1)
# visualize(m1,2)
#
# p1 = PointCloud(_v,1000,algo=:MarchingCubes)
# visualize(p1,1,markersize=0.1)
# visualize(p1,2, markersize=0.06)
#
# v2 = VoxelGrid(_m)
# visualize(v2,1,algo=:MarchingCubes)
# visualize(v2,2,algo=:MarchingCubes)
#
# p2 = PointCloud(_m)
# visualize(p2,1,markersize=0.3)
# visualize(p2,2)
#
# m3 = TriMesh(_p)
# visualize(m3,1,color=:green)
# visualize(m3,2)
#
# v3 = VoxelGrid(_p)
# visualize(v3,1)
# visualize(v3,2)
