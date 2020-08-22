function TriMesh(p::PointCloud, res::Int=32; algo=:MarchingCubes)
    voxel = pointcloud_to_voxel(p, res)
    v = VoxelGrid(voxel)
    return TriMesh(v; algo=algo)
end

function TriMesh(v::VoxelGrid; thresh::Number=0.5f0, algo=:MarchingCubes)
    verts,faces = voxel_to_trimesh(v,Float32(thresh),algo)
    return TriMesh(verts,faces)
end

function PointCloud(m::TriMesh, npoints::Int=1000)
    p = sample_points(m, npoints)
    return PointCloud(p)
end

function PointCloud(v::VoxelGrid, npoints::Int=1000; thresh::Number=0.5f0, algo=:MarchingCubes)
    m = TriMesh(v, thresh=thresh, algo=algo)
    points = sample_points(m, npoints)
    return PointCloud(points)
end

function VoxelGrid(m::TriMesh,res::Int=32)
    v = trimesh_to_voxel(m,res)
    return VoxelGrid(v)
end

function VoxelGrid(p::PointCloud, res::Int=32)
    v = pointcloud_to_voxel(p, res)
    return VoxelGrid(v)
end

function pointcloud_to_voxel(
    pcloud::PointCloud,
    resolution::Int = 32,
)
    p = cpu(pcloud.points)
    _, _N, _B = size(p)
    verts_max = maximum(maximum(p, dims = 1), dims = 2)
    verts_min = minimum(minimum(p, dims = 1), dims = 2)
    cloud = ((p .- verts_min) ./ (verts_max - verts_min))

    ind1 = Int[]
    ind2 = Int[]
    ind3 = Int[]

    for x = 1:resolution, y = 1:resolution, z = 1:resolution
        push!(ind1, x)
        push!(ind2, y)
        push!(ind3, z)
    end
    ind1 = reshape(ind1, 1, :)
    ind2 = reshape(ind2, 1, :)
    ind3 = reshape(ind3, 1, :)

    grid_points = cat(ind1, ind2, ind3, dims = 1)
    grid_points = ((grid_points .+ 0.5) ./ resolution)

    nn_idx = cat(
        [
            CartesianIndex.(
                reduce(vcat, knn(KDTree(cloud[:, :, i]), grid_points, 1)[1]),
                i,
            ) for i = 1:_B
        ]...,
        dims = 2,
    )

    dists_vec = (grid_points .- cloud[:, nn_idx])
    dists = sum((dists_vec .^ 2), dims = 1)
    dists = reshape(dists, resolution, resolution, resolution, _B)
    voxels = typeof(similar(pcloud.points,1,1,1,1))(dists .<= (0.6 / (resolution * resolution)))
    return voxels
end

function trimesh_to_voxel(m::TriMesh{T,R,S},res::Int=32)    where {T,R,S}
    verts_list = get_verts_list(m)
    faces_list = get_faces_list(m)

    _B = length(verts_list)
    voxels = S{T,4}(undef,res,res,res,_B)
    # print(typeof(verts_list))
    for (i,v,f) in zip(1:_B,verts_list, faces_list)
        voxels[:,:,:,i] = _voxelize(cpu(v),f,res)
    end
    return voxels
end

function _voxelize(
    v::Array{<:AbstractFloat,2},
    f::Array{<:Integer,2},
    resolution::Int = 32,
)
    verts_max = maximum(v)
    verts_min = minimum(v)
    verts = ((v .- verts_min) ./ (verts_max - verts_min))

    points = verts
    smallest_side = (1.0 / resolution)^2

    v1 = verts[:, f[1, :]]
    v2 = verts[:, f[2, :]]
    v3 = verts[:, f[3, :]]

    while true
        side_1 = sum((v1 .- v2) .^ 2, dims = 1)
        side_2 = sum((v2 .- v3) .^ 2, dims = 1)
        side_3 = sum((v3 .- v1) .^ 2, dims = 1)
        sides = cat(side_1, side_2, side_3, dims = 1)
        sides = reshape(maximum(sides, dims = 1), :)

        keep = sides .> smallest_side
        if !(any(keep))
            break
        end

        v1 = v1[:, keep]
        v2 = v2[:, keep]
        v3 = v3[:, keep]

        v4 = (v1 + v3) / 2
        v5 = (v1 + v2) / 2
        v6 = (v2 + v3) / 2

        points = cat(points, v4, v5, v6, dims = 2)
        vertex_set = [v1, v2, v3, v4, v5, v6]
        new_traingles = [
            1 4 5
            5 2 6
            5 4 6
            4 3 6
        ]
        new_verts = []
        for i = 1:4
            for j = 1:3
                if i == 1
                    push!(new_verts, vertex_set[new_traingles[i, j]])
                else
                    new_verts[j] = cat(
                        new_verts[j],
                        vertex_set[new_traingles[i, j]],
                        dims = 2,
                    )
                end
            end
        end

        v1, v2, v3 = new_verts
    end

    voxels = zeros(resolution, resolution, resolution)
    idx =
        (
            (
                round.(Int, points .* (resolution - 1), RoundToZero) .+
                resolution
            ) .% resolution
        ) .+ 1
    voxels[CartesianIndex.(idx[1, :], idx[2, :], idx[3, :])] .= 1
    return voxels
end

function voxel_to_trimesh(v::VoxelGrid, thresh, algo)
    algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets] ||
        error("given algo: $(algo) is not supported. Accepted algo are
              {:Exact,:MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets}.")

    _assert_voxel(v) || error("invalid VoxelGrid, found element which is not between [0,1].")
    voxel = v.voxels
    T = typeof(similar(voxel,1,1))
    R = Array{UInt32,2}
    verts = Vector{T}(undef,0)
    faces = Vector{R}(undef,0)
    method = algo==:Exact ? _voxel_exact : _voxel_algo

    for i in 1:size(voxel,4)
        v,f = method(cpu(voxel[:,:,:,i]),thresh,algo)
        v = T(v)
        v = v ./ maximum(v)
        f = R(f)
        push!(verts, v)
        push!(faces, f)
    end
    return verts, faces
end

function _voxel_algo(v, thresh, algo)
    res = size(v,1)
    voxel = zeros(eltype(v),res+2,res+2,res+2)
    voxel[2:end-1,2:end-1,2:end-1] .= v
    algo = eval(algo)
    voxel = voxel .>= thresh
    v, f = isosurface(voxel, algo(iso = 0, insidepositive = true))
    verts = reduce(hcat, v)
    faces = reduce(hcat, f)
    return verts, faces
end

function _voxel_exact(voxel, thresh, algo)
    voxel = voxel .>= thresh
    res = size(voxel, 1)
    if res >= 3
        voxel[2:res-1, 2:res-1, 2:res-1] =
            voxel[2:res-1, 2:res-1, 2:res-1] .& .!(
                voxel[1:res-2, 2:res-1, 2:res-1] .&
                voxel[3:res, 2:res-1, 2:res-1] .&
                voxel[2:res-1, 1:res-2, 2:res-1] .&
                voxel[2:res-1, 3:res, 2:res-1] .&
                voxel[2:res-1, 2:res-1, 1:res-2] .&
                voxel[2:res-1, 2:res-1, 3:res],
            )
    end

    v = Float32[]
    f = UInt32[]

    for i in CartesianIndices(voxel)
        if voxel[i]
            _add_face!(i, v, f)
        end
    end

    verts = reshape(v, 3, :)
    faces = reshape(f, 3, :)

    return verts, faces
end

function _add_face!(i, v, f)

    x, y, z = i[1], i[2], i[3]

    cube_verts = [
        x-1,
        y-1,
        z-1,
        x-1,
        y-1,
        z,
        x-1,
        y,
        z-1,
        x-1,
        y,
        z,
        x,
        y-1,
        z-1,
        x,
        y-1,
        z,
        x,
        y,
        z-1,
        x,
        y,
        z,
    ]

    cube_faces = [
        1,
        7,
        5,
        1,
        3,
        7,
        1,
        4,
        3,
        1,
        2,
        4,
        3,
        8,
        7,
        3,
        4,
        8,
        5,
        7,
        8,
        5,
        8,
        6,
        1,
        5,
        6,
        1,
        6,
        2,
        2,
        6,
        8,
        2,
        8,
        4,
    ]

    offset = length(v) / 3
    append!(v, cube_verts)
    append!(f, cube_faces .+ offset)
end
