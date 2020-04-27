@info "Testing ModelNet..."
@testset "ModelNet10 PointCloud dataset" begin

    mn10 = ModelNet10(mode="point_cloud")
    for (split, train) in [
            ("train", true),
            ("test", false)           
        ]
        
        dset = mn10(train=train, npoints=1024)

        @test dset isa Flux3D.Dataset.AbstractDataset
        @test dset.root == normpath(@__DIR__, "..", "datasets")
        @test dset.path == normpath(dset.root, "modelnet40_normal_resampled")
        @test dset.train == train
        @test dset.npoints == 1024
        # @test dset.sampling   #TODO add appropriate test
        # @test dset.transform   #TODO add appropriate test
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

        dpoint = dset[1]
        @test dpoint isa Dataset.AbstractDataPoint
        @test dpoint isa Dataset.MN10DataPoint
        @test dpoint.data isa PointCloud
    end
end

@testset "ModelNet40 PointCloud dataset" begin

    mn40 = ModelNet40(mode="point_cloud")
    for (split, train) in [
            ("train", true),
            ("test", false)           
        ]
        
        dset = mn40(train=train, npoints=1024)

        @test dset isa Flux3D.Dataset.AbstractDataset
        @test dset.root == normpath(@__DIR__, "..", "datasets")
        @test dset.path == normpath(dset.root, "modelnet40_normal_resampled")
        @test dset.train == train
        @test dset.npoints == 1024
        # @test dset.sampling   #TODO add appropriate test
        # @test dset.transform   #TODO add appropriate test
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

        dpoint = dset[1]
        @test dpoint isa Dataset.AbstractDataPoint
        @test dpoint isa Dataset.MN40DataPoint
        @test dpoint.data isa PointCloud
    end
end