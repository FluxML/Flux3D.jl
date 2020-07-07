@info "Testing Metrics..."
@testset "TriMesh Metrics" begin

    _mesh = load_trimesh([
        joinpath(@__DIR__, "../meshes/teapot.obj"),
        joinpath(@__DIR__, "../meshes/sphere.obj"),
    ]) |> gpu

    @testset "Laplacian loss" begin

        verts1 = [0.1 0.3 0.5;
                  0.5 0.2 0.1;
                  0.6 0.8 0.7]
        verts2 = [0.1 0.3 0.3;
                  0.6 0.7 0.8;
                  0.2 0.3 0.4;
                  0.1 0.5 0.3]
        verts3 = [0.7 0.3 0.6;
                  0.2 0.4 0.8;
                  0.9 0.5 0.2;
                  0.2 0.3 0.4;
                  0.9 0.3 0.8]

        verts_list = [verts1, verts2, verts3]

        faces1 = [1 2 3]
        faces2 = [1 2 3;
                  2 3 4]
        faces3 = [2 3 1;
                  1 2 4;
                  3 4 2;
                  5 4 3;
                  5 1 2;
                  5 4 2;
                  5 3 2]

        faces_list = [faces1, faces2, faces3]
        m = TriMesh(verts_list, faces_list) |> gpu

        _edges = get_edges_packed(m)
        V = size(get_verts_packed(m), 1)
        L = zeros(V,V)
        deg = zeros(V,V)
        for i in 1:size(_edges, 1)
            L[_edges[i,1], _edges[i,2]] = 1
            L[_edges[i,2], _edges[i,1]] = 1
        end
        deg = sum(L; dims = 2)
        deg = map(x -> (x > 0 ? 1 / x : x), deg)
        for i=1:V, j=1:V
            if i==j
                L[i,j] = -1
            elseif L[i,j]==1
                L[i,j] = deg[i]
            end
        end

        verts = get_verts_packed(m)
        L = L * verts
        L = Flux3D._norm(L; dims = 2)
        @test isapprox(mean(L), laplacian_loss(m))
        @test gradient(x->laplacian_loss(x), m) isa Tuple
    end

    @testset "Edge Loss" begin
        m = deepcopy(_mesh)
        verts = get_verts_packed(m)
        edges = get_edges_packed(m)
        v1 = verts[edges[:,1],:]
        v2 = verts[edges[:,2],:]
        loss =  mean((Flux3D._norm(v1-v2; dims=2)) .^ 2)
        @test edge_loss(m) == loss
        @test gradient(x->edge_loss(x), m) isa Tuple
    end

    @testset "sample_points" begin
        m = load_trimesh([
            joinpath(@__DIR__, "../meshes/sphere.obj"),
            joinpath(@__DIR__, "../meshes/sphere.obj"),
        ]) |> gpu
        samples = sample_points(m, 1000)
        _radius = sqrt.(sum(samples.^2; dims=2))
        @test samples isa CuArray{Float32, 3}
        @test all(isapprox.(
            cpu(_radius),
            1.0,
            rtol = 1e-2,
            atol = 1e-5,
        ))
        @test gradient(x->sum(sample_points(x,1000)),m) isa Tuple
    end

    @testset "Chamfer Loss" begin
        m = deepcopy(_mesh)
        loss = chamfer_distance(m,m)
        @test all(isapprox.(loss, 0, rtol = 1e-5, atol = 1e-2))
        @test gradient(x->chamfer_distance(x, x), m) isa Tuple

        # naive chamfer distance
        x = rand(1000,3) |> gpu
        y = rand(500,3) |> gpu
        xx = sum(x .^ 2, dims=2)
        yy = sum(y .^ 2, dims=2)
        zz = x * y'
        rx = reshape(xx, size(xx, 1), :)
        ry = reshape(yy, :, size(yy, 1))
        P = (rx .+ ry) .- (2 .* zz)
        distA = minimum(P; dims=2)
        distB = minimum(P; dims=1)
        loss = mean(distA) + mean(distB)
        @test isapprox(chamfer_distance(x,y), loss)
    end
end
