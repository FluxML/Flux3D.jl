@info "Testing CustomDataset..."
@testset "CustomDataset" begin
    x = rand(100, 8)
    getdata(idx) = x[idx, :]
    dset = CustomDataset(size(x, 1), getdata)

    @test size(dset) == (size(x, 1),)
    @test length(dset) == size(x, 1)
    @test firstindex(dset) == 1
    @test lastindex(dset) == size(x, 1)
    @test lastindex(dset) == size(x, 1)
    @test dset[1] == x[1, :]
    @test dset[3:8] == [x[i, :] for i = 3:8]
    @test dset[:] == [x[i, :] for i = 1:size(x, 1)]
end
