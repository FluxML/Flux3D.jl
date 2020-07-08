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
    @test all(isapprox.(L, Array(get_laplacian_packed(m)), rtol = 1e-5, atol = 1e-5))

    # verts_normals, faces_normals, faces_areas
    verts1 = [0.1 0.3 0.0;
             0.5 0.2 0.0;
             0.6 0.8 0.0;
             0.0 0.3 0.2;
             0.0 0.2 0.5;
             0.0 0.8 0.7;
             0.5 0.0 0.2;
             0.6 0.0 0.5;
             0.8 0.0 0.7;
             0.0 0.0 0.0;
             0.0 0.0 0.0;
             0.0 0.0 0.0]

    verts2 = [0.1 0.3 0.0;
              0.5 0.2 0.0;
              0.6 0.8 0.0;
              0.0 0.3 0.2;
              0.0 0.2 0.5;
              0.0 0.8 0.7]

    verts2 = [0.1 0.3 0.0;
              0.5 0.2 0.0;
              0.0 0.3 0.2;
              0.0 0.2 0.5;
              0.0 0.8 0.7]


    faces1 = [1 2 3;
             4 5 6;
             7 8 9;
             10 11 12]

    faces2 = [1 2 3;
              3 4 5]


    _normal1 = [0.0 0.0 1.0;
                0.0 0.0 1.0;
                0.0 0.0 1.0;
                -1.0 0.0 0.0;
                -1.0 0.0 0.0;
                -1.0 0.0 0.0;
                0.0 1.0 0.0;
                0.0 1.0 0.0;
                0.0 1.0 0.0;
                0.0 0.0 0.0;
                0.0 0.0 0.0;
                0.0 0.0 0.0]

    _normal2 = [-0.2408 -0.9631 -0.1204;
                -0.2408 -0.9631 -0.1204;
                -0.9389 -0.3414 -0.0427;
                -1.0000 0.0000 0.0000;
                -1.0000 0.0000 0.0000]

    _fnormal1 = [-0.0 0.0 1.0;
                 -1.0 0.0 0.0;
                 0.0 1.0 0.0;
                 0.0 0.0 0.0]

    _fnormal2 = [-0.2408 -0.9631 -0.1204;
                 -1.0000 0.0000 0.0000;]

    _farea1 = [0.125, 0.1, 0.02, 0.0]
    _farea2 = [0.0415, 0.1]

    _normal1 = _normal1'
    _normal2 = _normal2'
    _fnormal1 = _fnormal1'
    _fnormal2 = _fnormal2'
    _farea1 = _farea1'
    _farea2 = _farea2'

    #verts_normals, faces_normals, faces_areas
    for T in [Float32], R in [Int64, UInt32]
        verts = [T.(verts1'), T.(verts2')]
        faces = [R.(faces1'), R.(faces2')]
        m = TriMesh(verts, faces)
        gm = m |> gpu

        normals_packed = cpu(compute_verts_normals_packed(gm))
        normals_padded = cpu(compute_verts_normals_padded(gm))
        normals_list = cpu(compute_verts_normals_list(gm))
        @test compute_verts_normals_packed(gm) isa CuArray{T,2}
        @test compute_verts_normals_padded(gm) isa CuArray{T,3}
        @test compute_verts_normals_list(gm) isa Array{<:CuArray{T,2},1}
        @test all(isapprox.(cat(_normal1,_normal2; dims=2), normals_packed, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.([_normal1, _normal2], normals_list, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_normal1, normals_padded[:,1:size(_normal1,2),1], rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_normal2, normals_padded[:,1:size(_normal2,2),2], rtol = 1e-4, atol = 1e-4))
        @test all(normals_padded[:,size(_normal1,2)+1:end,1] .== 0)
        @test all(normals_padded[:,size(_normal2,2)+1:end,2] .== 0)

        normals_packed = cpu(compute_faces_normals_packed(gm))
        normals_padded = cpu(compute_faces_normals_padded(gm))
        normals_list = cpu(compute_faces_normals_list(gm))
        @test compute_faces_normals_packed(gm) isa CuArray{T,2}
        @test compute_faces_normals_padded(gm) isa CuArray{T,3}
        @test compute_faces_normals_list(gm) isa Array{<:CuArray{T,2},1}
        @test all(isapprox.(cat(_fnormal1,_fnormal2; dims=2), normals_packed, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.([_fnormal1, _fnormal2], normals_list, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_fnormal1, normals_padded[:,1:size(_fnormal1,2),1], rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_fnormal2, normals_padded[:,1:size(_fnormal2,2),2], rtol = 1e-4, atol = 1e-4))
        @test all(normals_padded[:,size(_fnormal1,2)+1:end,1] .== 0)
        @test all(normals_padded[:,size(_fnormal2,2)+1:end,2] .== 0)

        areas_packed = cpu(compute_faces_areas_packed(gm))
        areas_padded = cpu(compute_faces_areas_padded(gm))
        areas_list = cpu(compute_faces_areas_list(gm))
        @test compute_faces_areas_packed(gm) isa CuArray{T,1}
        @test compute_faces_areas_padded(gm) isa CuArray{T,3}
        @test compute_faces_areas_list(gm) isa Array{<:CuArray{T,2},1}
        @test all(isapprox.(reshape(cat(_farea1,_farea2; dims=2),:), areas_packed, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.([_farea1, _farea2], areas_list, rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_farea1, areas_padded[:,1:size(_farea1,2),1], rtol = 1e-4, atol = 1e-4))
        @test all(isapprox.(_farea2, areas_padded[:,1:size(_farea2,2),2], rtol = 1e-4, atol = 1e-4))
        @test all(areas_padded[:,size(_farea1,2)+1:end,1] .== 0)
        @test all(areas_padded[:,size(_farea2,2)+1:end,2] .== 0)
    end
end # testset TriMesh

@info "Testing Rep Utils..."
@testset "Rep Utils" begin
    for T in [Float32]
        a1 = [1 2 3;
              4 5 6;
              7 8 9;
              10 11 12]

        a2 = [13 14 15;
              16 17 18]

        a3 = [19 20 21;
              22 23 24;
              25 26 27]

        a1 = T.(a1')
        a2 = T.(a2')
        a3 = T.(a3')
        _list = [a1,a2,a3] |> gpu

        _packed = [1 2 3;
                   4 5 6;
                   7 8 9;
                   10 11 12;
                   13 14 15;
                   16 17 18;
                   19 20 21;
                   22 23 24;
                   25 26 27]
        _packed = T.(_packed') |> gpu

        _padded = zeros(3, 4, 3)
        _padded[:, 1:size(a1,2),1] = a1
        _padded[:, 1:size(a2,2),2] = a2
        _padded[:,1:size(a3,2),3] = a3
        _padded = T.(_padded) |> gpu

        items_len, packed_first_idx, packed_to_list_idx =
            Flux3D._auxiliary_mesh(_list)

        @test items_len == [4,2,3]
        @test packed_first_idx == [1,5,7]
        @test all(packed_to_list_idx[1:4] .== 1)
        @test all(packed_to_list_idx[5:6] .== 2)
        @test all(packed_to_list_idx[7:9] .== 3)

        @test Flux3D._list_to_padded(_list, 0) == _padded
        @test Flux3D._list_to_padded(_list, 0) isa CuArray{T,3}
        @test gradient(x-> 0.5 .* sum(Flux3D._list_to_padded(x, 0) .^ 2), _list)[1] == _list

        @test Flux3D._list_to_packed(_list) == _packed
        @test Flux3D._list_to_packed(_list) isa CuArray{T,2}
        @test gradient(x->.5 .* sum(Flux3D._list_to_packed(x) .^ 2), _list)[1] == Tuple(_list)

        @test Flux3D._packed_to_padded(_packed, items_len, 0) == _padded
        @test Flux3D._packed_to_padded(_packed, items_len, 0) isa CuArray{T,3}
        @test gradient(x-> .5 .* sum(Flux3D._packed_to_padded(x, items_len, 0) .^ 2), _packed)[1] == _packed

        @test Flux3D._packed_to_list(_packed, items_len) == _list
        @test Flux3D._packed_to_list(_packed, items_len) isa Array{<:CuArray{T,2},1}
        gs = gradient(_packed) do x
                loss = 0
                for x in Flux3D._packed_to_list(x, items_len)
                    loss += sum(x .^ 2)
                end
                return 0.5*loss
            end
        @test gs[1] == _packed

        @test Flux3D._padded_to_list(_padded, items_len) == _list
        @test Flux3D._padded_to_list(_padded, items_len) isa Array{<:CuArray{T,2},1}
        gs = gradient(_padded) do x
                loss = 0
                for x in Flux3D._padded_to_list(x, items_len)
                    loss += sum(x .^ 2)
                end
                return 0.5*loss
            end
        @test gs[1] == _padded

        @test Flux3D._padded_to_packed(_padded, items_len) == _packed
        @test Flux3D._padded_to_packed(_padded, items_len) isa CuArray{T,2}
        @test gradient(x->sum(.5 .* sum(Flux3D._padded_to_packed(x, items_len) .^2 )), _padded)[1] == _padded
    end
end
