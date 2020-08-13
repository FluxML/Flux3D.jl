@testset "TriMesh Metrics" begin

    _mesh = load_trimesh([
        joinpath(@__DIR__, "meshes/teapot.obj"),
        joinpath(@__DIR__, "meshes/sphere.obj"),
    ])

    @testset "Laplacian loss" begin

        verts1 = [
            0.1 0.3 0.5
            0.5 0.2 0.1
            0.6 0.8 0.7
        ]
        verts2 = [
            0.1 0.3 0.3
            0.6 0.7 0.8
            0.2 0.3 0.4
            0.1 0.5 0.3
        ]
        verts3 = [
            0.7 0.3 0.6
            0.2 0.4 0.8
            0.9 0.5 0.2
            0.2 0.3 0.4
            0.9 0.3 0.8
        ]


        faces1 = [1 2 3]
        faces2 = [
            1 2 3
            2 3 4
        ]
        faces3 = [
            2 3 1
            1 2 4
            3 4 2
            5 4 3
            5 1 2
            5 4 2
            5 3 2
        ]

        T, R = Float32, UInt32
        verts_list = [T.(verts1'), T.(verts2'), T.(verts3')]
        faces_list = [R.(faces1'), R.(faces2'), R.(faces3')]
        m = TriMesh(verts_list, faces_list)

        _edges = get_edges_packed(m)
        V = size(get_verts_packed(m), 2)
        L = zeros(V, V)
        deg = zeros(V, V)
        for i = 1:size(_edges, 1)
            L[_edges[i, 1], _edges[i, 2]] = 1
            L[_edges[i, 2], _edges[i, 1]] = 1
        end
        deg = sum(L; dims = 2)
        deg = map(x -> (x > 0 ? 1 / x : x), deg)
        for i = 1:V, j = 1:V
            if i == j
                L[i, j] = -1
            elseif L[i, j] == 1
                L[i, j] = deg[i]
            end
        end

        verts = get_verts_packed(m)
        L = L * transpose(verts)
        L = Flux3D._norm(L; dims = 2)
        @test isapprox(mean(L), laplacian_loss(m))
        @test gradient(x -> laplacian_loss(x), m) isa Tuple
    end

    @testset "Edge Loss" begin
        m = deepcopy(_mesh)
        verts = get_verts_packed(m)
        edges = get_edges_packed(m)
        v1 = verts[:, edges[:, 1]]
        v2 = verts[:, edges[:, 2]]
        loss = mean((Flux3D._norm(v1 - v2; dims = 1)) .^ 2)
        @test edge_loss(m) == loss
        @test gradient(x -> edge_loss(x), m) isa Tuple
    end

    @testset "Chamfer Loss" begin
        m = deepcopy(_mesh)
        loss = chamfer_distance(m, m)
        @test all(isapprox.(loss, 0, rtol = 1e-5, atol = 1e-2))
        @test gradient(x -> chamfer_distance(x, x), m) isa Tuple

        # naive chamfer distance

        function naive_chamfer(x,y)
            xx = sum(x .^ 2, dims = 1)
            yy = sum(y .^ 2, dims = 1)
            zz = Flux3D.Flux.batched_mul(permutedims(x, (2, 1, 3)), y)
            rx = reshape(xx, size(xx, 2), 1, :)
            ry = reshape(yy, 1, size(yy, 2), :)
            P = (rx .+ ry) .- (2 .* zz)
            nn_for_x = minimum(P; dims = 2)
            nn_for_y = minimum(P; dims = 1)
            nn_for_x = mean(nn_for_x; dims = 2)
            nn_for_y = mean(nn_for_y; dims = 2)
            loss = mean(nn_for_x) + mean(nn_for_y)
            return loss
        end

        x = rand(Float32, 3, 1000, 2)
        y = rand(Float32, 3, 500, 2)
        @test isapprox(chamfer_distance(x, y), naive_chamfer(x,y))
        grad1 = gradient(x -> chamfer_distance(x, y), x)[1]
        grad2 = gradient(x -> naive_chamfer(x, y), x)[1]
        @test isapprox(grad1,grad2,atol=1e-2,rtol=1e-3)
    end
end
