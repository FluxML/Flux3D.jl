@info "Testing CustomDataset..."
@testset "CustomDataset" begin
    x = rand(100,8)
    getdata(idx) = x[idx,:]
    dset = CustomDataset(size(x,1), getdata)

    @test size(dset) == (size(x,1),)
    @test length(dset) == size(x,1)
    @test firstindex(dset) == 1
    @test lastindex(dset) == size(x,1)
    @test lastindex(dset) == size(x,1)
    @test dset[1] == x[1,:]
    @test dset[3:8] == [x[i,:] for i in 3:8]
    @test dset[:] == [x[i,:] for i in 1:size(x,1)]
end

@info "Testing ModelNet..."
@testset "ModelNet10 PointCloud dataset" begin
    for (split, train) in [
            ("train", true),
            ("test", false)           
        ]
        
        npoints=32
        dset = ModelNet10.dataset(;mode=:pointcloud, train=train, npoints=npoints)
        print(@__DIR__)
        @test dset isa Flux3D.Dataset.AbstractDataset
        @test dset.root == normpath(@__DIR__, "../datasets/modelnet")
        @test dset.path == normpath(@__DIR__, "../datasets/modelnet/modelnet40_normal_resampled")
        @test dset.train == train
        @test dset.npoints == npoints
        @test dset.sampling isa Nothing   #TODO: add appropriate test
        @test dset.transform  isa Nothing
        @test dset.datapaths isa AbstractArray

        if train
            @test dset.length == 3991
            @test size(dset) == (3991,)
            @test length(dset) == 3991
        else
            @test dset.length == 908
            @test size(dset) == (908,)
            @test length(dset) == 908
        end
        
        dpoint1 = dset[1]
        @test dpoint1 isa Flux3D.Dataset.AbstractDataPoint
        @test dpoint1 isa ModelNet10.MN10DataPoint
        @test dpoint1.data isa PointCloud

        t = Compose(ScalePointCloud(2))
        dset = ModelNet10.dataset(;mode=:pointcloud, train=train, npoints=npoints, transform = t)
        @test dset.transform isa Compose
        dpoint2 = dset[1]
        @test dpoint2 isa Flux3D.Dataset.AbstractDataPoint
        @test dpoint2 isa ModelNet10.MN10DataPoint
        @test dpoint2.data isa PointCloud

        @test all(isapprox.(2 .* dpoint1.data.points, dpoint2.data.points, rtol = 1e-5, atol = 1e-5))
    end
end

@testset "ModelNet40 PointCloud dataset" begin
    for (split, train) in [
            ("train", true),
            ("test", false)           
        ]
        
        npoints=32
        dset = ModelNet40.dataset(;mode=:pointcloud, train=train, npoints=npoints)

        @test dset isa Flux3D.Dataset.AbstractDataset
        @test dset.root == normpath(@__DIR__, "../datasets/modelnet")
        @test dset.path == normpath(@__DIR__, "../datasets/modelnet/modelnet40_normal_resampled")
        @test dset.train == train
        @test dset.npoints == npoints
        @test dset.sampling isa Nothing  #TODO: add appropriate test
        @test dset.transform isa Nothing
        @test dset.datapaths isa AbstractArray

        if train
            @test dset.length == 9843
            @test size(dset) == (9843,)
            @test length(dset) == 9843
        else
            @test dset.length == 2468
            @test size(dset) == (2468,)
            @test length(dset) == 2468
        end

        dpoint1 = dset[1]
        @test dpoint1 isa Flux3D.Dataset.AbstractDataPoint
        @test dpoint1 isa ModelNet40.MN40DataPoint
        @test dpoint1.data isa PointCloud

        t = Compose(ScalePointCloud(2))
        dset = ModelNet40.dataset(;mode=:pointcloud, train=train, npoints=npoints, transform = t)
        @test dset.transform isa Compose
        dpoint2 = dset[1]
        @test dpoint2 isa Flux3D.Dataset.AbstractDataPoint
        @test dpoint2 isa ModelNet40.MN40DataPoint
        @test dpoint2.data isa PointCloud

        @test all(isapprox.(2 .* dpoint1.data.points, dpoint2.data.points, rtol = 1e-5, atol = 1e-5))
    end
end