export TriMesh, GBMesh, gbmeshes, load_trimesh, save_trimesh, get_verts_list,
       get_verts_packed, get_verts_padded, get_faces_list, get_faces_packed,
       get_faces_padded, get_edges_packed, get_edges_to_key,
       get_faces_to_edges_packed, get_laplacian_packed, get_laplacian_sparse,
       compute_verts_normals_list, compute_verts_normals_packed,
       compute_verts_normals_padded, compute_faces_normals_list,
       compute_faces_normals_packed, compute_faces_normals_padded,
       compute_faces_areas_list, compute_faces_areas_packed,
       compute_faces_areas_padded

import GeometryBasics, Printf
import GeometryBasics:
    Point3f0, GLTriangleFace, NgonFace, convert_simplex, meta, triangle_mesh

import Zygote: @ignore

"""
    TriMesh

Initialize Triangle Mesh representation.

### Fields:

- `N`                       - Batch size of TriMesh.
- `V`                       - Maximum vertices per mesh in TriMesh.
- `F`                       - Maximum faces per mesh in TriMesh.
- `equalised`               - Bool, indicates all mesh have same verts and faces size.
- `valid`                   -
- `offset`                  - Offset indicating number to be added to migrate to 0-indexed system.

- `_verts_len`              - Number of vertices in each mesh of TriMesh.
- `_verts_list`             - Vertices in list format.
- `_verts_packed`           - Vertices in packed format.
- `_verts_padded`           - Vertices in padded format.

- `_faces_len`              - Number of faces in each mesh of TriMesh.
- `_faces_list`             - Vertices in list format.
- `_faces_packed`           - Vertices in packed format.
- `_faces_padded`           - Vertices in padded format.

- `_edges_packed`           - Edges in packed format (according to packed vertices).
- `_faces_to_edges_packed`  - Faces formed by edges in packed format (according to packed edges).
- `_laplacian_packed`       - Laplacian sparce matrix in packed format.
- `_edges_to_key`           - Dict mapping edges tuple to unique key.

### Available Contructor:

- `TriMesh(verts_list, faces_list; offset::Number = -1)`
- `TriMesh(m::Vector{<:GeometryBasics.Mesh})`
- `TriMesh(m::GeometryBasics.Mesh)`

"""
mutable struct TriMesh{T<:AbstractFloat,R<:Integer} <: AbstractMesh{T,R}
    N::Int64
    V::Int64
    F::Int64
    equalised::Bool
    valid::AbstractArray{Bool, 1}
    offset::Int8
    _verts_len::AbstractArray{Int,1}
    _faces_len::AbstractArray{Int,1}

    _verts_packed::Union{Nothing, AbstractArray{T,2}}
    _verts_padded::Union{Nothing, AbstractArray{T,3}}
    _verts_list::Union{Nothing, AbstractArray{<:AbstractArray{T,2},1}}

    _faces_packed::Union{Nothing, AbstractArray{R,2}}
    _faces_padded::Union{Nothing, AbstractArray{R,3}}
    _faces_list::Union{Nothing, AbstractArray{<:AbstractArray{R,2},1}}

    _edges_packed::Union{Nothing, AbstractArray{R,2}}
    _faces_to_edges_packed::Union{Nothing, AbstractArray{R,2}}
    _laplacian_packed::Union{Nothing,AbstractSparseArray{T,R,2}}

    _edges_to_key::Union{Nothing,Dict{Tuple{R,R},R}}
end

TriMesh(verts::AbstractArray{<:AbstractArray{T,2},1},
        faces::AbstractArray{<:AbstractArray{R,2},1};
        offset::Number = -1) where {T,R} =
        TriMesh(Float32.(verts), UInt32.(faces); offset=offset)

function TriMesh(
    verts::AbstractArray{<:AbstractArray{T,2},1},
    faces::AbstractArray{<:AbstractArray{R,2},1};
    offset::Number = -1
)   where {T<:AbstractFloat,R<:Integer}

    length(verts) == length(faces) || error("batch size of verts and faces should match, $(length(verts)) != $(length(faces))")
    _verts_len = size.(verts, 1)
    _faces_len = size.(faces, 1)
    N = length(verts)
    V = maximum(_verts_len)
    F = maximum(_faces_len)
    equalised = all(_verts_len .== V) && all(_faces_len .== F)
    valid = _faces_len .> 0
    offset = Int8(offset)

    _verts_list = verts
    _faces_list = faces

    return TriMesh(N,V,F,equalised,valid,offset,_verts_len,_faces_len,
                   nothing, nothing,_verts_list, nothing, nothing,_faces_list,
                   nothing, nothing, nothing, nothing)
end

function TriMesh(ms::Vector{<:GeometryBasics.Mesh})
    verts_list = []
    faces_list = []
    for (i,m) = enumerate(ms)
        (verts, faces) = _load_meta(m)
        push!(verts_list, verts)
        push!(faces_list, faces)
    end
    return TriMesh([verts_list], [faces_list])
end

TriMesh(m::GeometryBasics.Mesh) = TriMesh([m])

# converts (verts, faces) to GeometryBasics Mesh
function GBMesh(verts::AbstractArray{T,2}, faces::AbstractArray{R,2}) where {T,R}
    points = Point3f0[
        GeometryBasics.Point{3,Float32}(verts[i, :]) for i = 1:size(verts, 1)
    ]
    verts_dim = size(faces, 2)
    poly_face = NgonFace{verts_dim,UInt32}[
        NgonFace{verts_dim,UInt32}(faces[i, :]) for i = 1:size(faces, 1)
    ]
    # faces = convert_simplex.(GLTriangleFace, poly_face)
    faces = GLTriangleFace.(poly_face)
    return GeometryBasics.Mesh(meta(points), faces)
end

GBMesh(m::TriMesh; index::Int = 1) = GBMesh(m[index]...)

function gbmeshes(m::TriMesh)
    gbmeshes = [ GBMesh(m; index=i) for i in 1:m.N ]
    return gbmeshes
end

function _load_meta(m::GeometryBasics.Mesh)
    if !(m isa GeometryBasics.Mesh{3,Float32,<:GeometryBasics.Triangle})
        m = triangle_mesh(m)
    end
    vs = m.position
    vertices = [reshape(Array(v), 1, :) for v in vs]
    vertices = reduce(vcat, vertices)
    fs = getfield(getfield(m, :simplices), :faces)
    faces = [reshape(Array(UInt32.(f)), 1, :) for f in fs]
    faces = reduce(vcat, faces)
    return (vertices, faces)
end

function load_trimesh(fn::String; elements_types...)
    mesh = load(fn; elements_types...)
    verts, faces = _load_meta(mesh)
    # # offset = _get_offset(x)   #offset is always -1 when loaded from MeshIO
    return TriMesh([verts], [faces])
end

function load_trimesh(fns::Vector{String}; elements_types...)
    verts_list = Vector{<:AbstractArray{<:AbstractFloat,2}}[]
    faces_list = Vector{<:AbstractArray{<:Integer,2}}[]
    for (i,fn) in enumerate(fns)
        mesh = load(fn; elements_types...)
        verts, faces = _load_meta(mesh)
        push!(verts_list, verts)
        push!(faces_list, faces)
    end
    return TriMesh(verts_list, faces_list)
end

save_trimesh(fn::String, mesh::TriMesh, index::Int=1) =
    save_trimesh(fn, mesh[index]...)

function save_trimesh(fn::String, verts::AbstractArray{<:AbstractFloat,2}, faces::AbstractArray{<:Integer,2})
    verts = Float32.(verts)
    faces = UInt32.(faces)
    mesh = GBMesh(verts, faces)
    save(fn, mesh)
end

# MeshIO doesn't have support for saving obj file
function MeshIO.save(str::Stream{format"OBJ"}, msh::GeometryBasics.AbstractMesh)
    io = stream(str)
    vts = GeometryBasics.coordinates(msh)
    fcs = GeometryBasics.faces(msh)

    # write header
    write(io, "# Flux3D.jl v0.1.0 OBJ File: \n")
    write(io, "# www.github.com/nirmal-suthar/Flux3D.jl \n")

    # write vertices data
    for v in vts
        Printf.@printf(io, "v %.6f %.6f %.6f\n", v...)
    end

    # write faces data
    for f in fcs
        Printf.@printf(io, "f %d %d %d\n", f...)
    end

    close(io)
end

function Base.setproperty!(m::TriMesh, f::Symbol, v)
    if (f==:_verts_packed) || (f==:_verts_padded) || (f==:_verts_list)
        if (f == :_verts_packed) && (getproperty(m,f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_padded, nothing)
            _compute_verts_list(m, true)
        elseif (f == :_verts_padded) && (getproperty(m,f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_packed, nothing)
            _compute_verts_list(m, true)
        elseif (f == :_verts_list) && (getproperty(m,f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_packed, nothing)
            setfield!(m, :_verts_padded, nothing)
        end
    else
        setfield!(m, f, convert(fieldtype(typeof(m), f), v))
    end
end

Base.getindex(m::TriMesh, inds...) = (m._verts_list[inds...], m._faces_list[inds...])

function get_verts_packed(m::TriMesh; refresh::Bool = false)
    _compute_verts_packed(m, refresh)
    return m._verts_packed
end

function get_verts_padded(m::TriMesh; refresh::Bool = false)
    _compute_verts_padded(m, refresh)
    return m._verts_padded
end

function get_verts_list(m::TriMesh; refresh::Bool = false)
    return m._verts_list
end

function get_faces_packed(m::TriMesh; refresh::Bool = false)
    @ignore _compute_faces_packed(m, refresh)
    return m._faces_packed
end

function get_faces_padded(m::TriMesh; refresh::Bool = false)
    @ignore _compute_faces_padded(m, refresh)
    return m._faces_padded
end

function get_faces_list(m::TriMesh; refresh::Bool = false)
    return m._faces_list
end

function get_edges_packed(m::TriMesh; refresh::Bool = false)
    @ignore _compute_edges_packed(m, refresh)
    return m._edges_packed
end

function get_edges_to_key(m::TriMesh; refresh::Bool = false)
    @ignore _compute_edges_packed(m, refresh)
    return m._edges_to_key
end

function get_faces_to_edges_packed(m::TriMesh; refresh::Bool = false)
    @ignore _compute_edges_packed(m, refresh)
    return m._faces_to_edges_packed
end

function get_laplacian_packed(m::TriMesh; refresh::Bool = false)
    _compute_laplacian_packed(m, refresh)
    return m._laplacian_packed
end

get_laplacian_sparse(m::TriMesh; refresh::Bool = false) = get_laplacian_packed(m; refresh=refresh)

function compute_verts_normals_packed(m::TriMesh)

    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[faces, :]
    vertex_normals = Zygote.bufferfrom(zeros(Float32, size(verts)...))

    vertex_normals[faces[:, 1], :] += _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    vertex_normals[faces[:, 2], :] += _lg_cross(
        vert_faces[:, 3, :] - vert_faces[:, 2, :],
        vert_faces[:, 1, :] - vert_faces[:, 2, :],
    )
    vertex_normals[faces[:, 3], :] += _lg_cross(
        vert_faces[:, 1, :] - vert_faces[:, 3, :],
        vert_faces[:, 2, :] - vert_faces[:, 3, :],
    )

    return _normalize(copy(vertex_normals), dims = 2)
end

function compute_verts_normals_padded(m::TriMesh)
    normals_packed = compute_verts_normals_packed(m)
    normals_padded = _packed_to_padded(normals_packed, m._verts_len, 0.0)
    return normals_padded
end

function compute_verts_normals_list(m::TriMesh)
    normals_packed = compute_verts_normals_packed(m)
    normals_list = _packed_to_list(normals_packed, m._verts_len)
    return normals_list
end

function compute_faces_normals_packed(m::TriMesh)
    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[faces, :]
    face_normals = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    return _normalize(face_normals, dims = 2)
end

function compute_faces_normals_padded(m::TriMesh)
    normals_packed = compute_faces_normals_packed(m)
    normals_padded = _packed_to_padded(normals_packed, m._faces_len, 0.0)
    return normals_padded
end

function compute_faces_normals_list(m::TriMesh)
    normals_packed = compute_faces_normals_packed(m)
    normals_list = _packed_to_list(normals_packed, m._faces_len)
    return normals_list
end

function compute_faces_areas_packed(m::TriMesh; eps::Number = 1e-6)
    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[faces, :]
    face_normals_vec = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    face_norm = sqrt.(sum(face_normals_vec .^ 2, dims = 2))
    face_areas = dropdims(face_norm ./ 2; dims = 2)
    return face_areas
end

function compute_faces_areas_padded(m::TriMesh; eps::Number = 1e-6)
    areas_packed = compute_faces_areas_packed(m; eps=eps)
    areas_packed = reshape(areas_packed,:,1)
    areas_padded = _packed_to_padded(areas_packed, m._faces_len, 0.0)
    return areas_padded
end

function compute_faces_areas_list(m::TriMesh; eps::Number = 1e-6)
    areas_packed = compute_faces_areas_packed(m; eps=eps)
    areas_list = _packed_to_list(areas_packed, m._faces_len)
    return areas_list
end

function _compute_verts_packed(m::TriMesh, refresh::Bool = false)
    if refresh || (m._verts_packed isa Nothing)
        verts_packed = _list_to_packed(m._verts_list)
        setfield!(m, :_verts_packed, verts_packed)
    end
end

function _compute_verts_padded(m::TriMesh, refresh::Bool = false)
    if refresh || (m._verts_padded isa Nothing)
        verts_padded = _list_to_padded(m._verts_list, 0, (m.V, 3))
        setfield!(m, :_verts_padded, verts_padded)
    end
end

function _compute_verts_list(m::TriMesh, refresh::Bool = false)
    if refresh || (m._verts_list isa Nothing)
        if m._verts_packed !== nothing
            verts_list = _packed_to_list(m._verts_packed, m._verts_len)
        elseif m._verts_padded !== nothing
            verts_list = _padded_to_list(m._verts_padded, m._verts_len)
        else
            error("not possible to contruct list without padded and packed")
        end
        setfield!(m, :_verts_list, verts_list)
    end
end

function _compute_faces_packed(m::TriMesh{T,R}, refresh::Bool = false) where {T,R}
    if refresh || (m._faces_packed isa Nothing)
        faces_packed = _list_to_packed(m._faces_list)
        _,verts_packed_first_idx,_ = _auxiliary_mesh(m._verts_list)
        _,_,faces_packed_list_idx = _auxiliary_mesh(m._faces_list)
        faces_packed_offset = verts_packed_first_idx[faces_packed_list_idx] .- 1
        faces_packed = faces_packed .+ faces_packed_offset
        setfield!(m, :_faces_packed, R.(faces_packed))
    end
end

function _compute_faces_padded(m::TriMesh, refresh::Bool = false)
    if refresh || (m._faces_padded isa Nothing)
        faces_padded = _list_to_padded(m._faces_list, 0, (m.F, 3))
        setfield!(m, :_faces_padded, faces_padded)
    end
end

function _compute_edges_packed(m::TriMesh, refresh::Bool = false)
    if refresh || (any([m._edges_packed, m._edges_to_key, m._faces_to_edges_packed] .=== nothing))

        faces = get_faces_packed(m)
        verts = get_verts_packed(m)

        e12 = cat(faces[:, 1], faces[:, 2], dims = 2)
        e23 = cat(faces[:, 2], faces[:, 3], dims = 2)
        e31 = cat(faces[:, 3], faces[:, 1], dims = 2)

        # Sort edges (v0, v1) such that v0 <= v1
        e12 = sort(e12; dims = 2)
        e23 = sort(e23; dims = 2)
        e31 = sort(e31; dims = 2)

        # Edges including duplicates
        edges = cat(e12, e23, e31, dims = 1)

        # Converting edge (v0, v1) into integer hash, ie. (V+1)*v0 + v1.
        # There will be no collision, which is asserted by (V+1), as 1<=v0<=V.
        V_hash = size(verts, 1) + 1
        edges_hash = (V_hash .* edges[:, 1]) .+ edges[:, 2]

        # Sort and remove duplicate edges_hash
        sort!(edges_hash)
        unique!(edges_hash)

        # Convert edges_hash to edges
        edges = cat((edges_hash .÷ V_hash), (edges_hash .% V_hash); dims = 2)

        # Edges to key
        edges_to_key = Dict{Tuple{UInt32,UInt32},UInt32}([
            (Tuple(edges[i, :]), i) for i = 1:size(edges, 1)
        ])

        # e12 -> tuple -> get
        e12_tup = [Tuple(e12[i, :]) for i = 1:size(e12, 1)]
        e23_tup = [Tuple(e23[i, :]) for i = 1:size(e23, 1)]
        e31_tup = [Tuple(e31[i, :]) for i = 1:size(e31, 1)]
        faces_to_edges_tuple = cat(e23_tup, e31_tup, e12_tup; dims = 2)

        faces_to_edges = map(x -> get(edges_to_key, x, -1), faces_to_edges_tuple)

        m._edges_packed = edges
        m._edges_to_key = edges_to_key
        m._faces_to_edges_packed = faces_to_edges
    end
end
"""

"""
function _compute_laplacian_packed(m::TriMesh{T,R}, refresh::Bool = false) where {T,R}
    if refresh || (m._laplacian_packed isa Nothing)
        # laplacian sparse `L`:
        # where i, j are vertices indices, then
        # L[i, j] =    -1       , if i == j
        # L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        # L[i, j] =    0        , otherwise
        verts = get_verts_packed(m)
        edges = get_edges_packed(m)

        # computing all i, j of edges (i,j), as (j,i) is also an edge
        e1 = edges[:, 1]
        e2 = edges[:, 2]

        # computing all edges, considering (i,j) and (j,i)
        idx12 = cat(e1, e2, dims = 2)
        idx21 = cat(e2, e1, dims = 2)
        idx = cat(idx12, idx21, dims = 1)

        A = sparse(
            idx[:, 1],
            idx[:, 2],
            ones(Float32, size(idx, 1)),
            size(verts, 1),
            size(verts, 1),
        )

        # computing degree of each vertices
        deg = sum(A, dims = 2)

        # if degree >= 1, then there exist edge with that vertex,
        # so computing 1/deg[i]
        deg1 = map(x -> T.(x > 0 ? 1 / x : x), deg[e1])
        deg2 = map(x -> T.(x > 0 ? 1 / x : x), deg[e2])
        diag = fill(T.(-1), size(verts, 1))

        # assuming (i,j) is an edge, so we computing 1/deg[i],
        # and store at L[i,j] and same for (j,i).
        # We also store -1 for L[i,i] for all vertices.
        Is = cat(e1, e2, R.(1:size(verts, 1)); dims = 1)
        Js = cat(e2, e1, R.(1:size(verts, 1)); dims = 1)
        Vs = cat(deg1, deg2, diag; dims = 1)
        m._laplacian_packed = sparse(Is, Js, Vs, size(verts, 1), size(verts, 1))
    end
end
