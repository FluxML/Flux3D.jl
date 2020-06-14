export TriMesh,
    load_trimesh,
    compute_vertex_normals,
    compute_face_normals,
    compute_face_normals,
    compute_face_areas,
    get_edges,
    get_laplacian_sparse,
    get_faces_to_edges,
    get_edges_to_key

import GeometryBasics
import GeometryBasics:
    Point3f0, GLTriangleFace, NgonFace, convert_simplex, Mesh, meta, triangle_mesh

# TODO: add texture fields
mutable struct TriMesh{T<:Float32,R<:UInt32} <: AbstractMesh
    vertices::AbstractArray{T,2}
    faces::AbstractArray{R,2}
    offset::Int8
    edges::Union{Nothing,AbstractArray{UInt32,2}}
    laplacian_sparse::Union{Nothing,AbstractSparseMatrix{Float32,UInt32}}
    edges_to_key::Union{Nothing,Dict{Tuple{UInt32,UInt32},UInt32}}
    faces_to_edges::Union{Nothing,AbstractArray{UInt32,2}}
end

# TODO: Add contructor according to batched format
function TriMesh(
    vertices::AbstractArray{<:Number,2},
    faces::AbstractArray{<:Number,2};
    offset::Integer = -1,
)
    vertices = Float32.(vertices)
    faces = UInt32.(faces)
    offset = Int8(offset)
    return TriMesh(vertices, faces, offset, nothing, nothing, nothing, nothing)
end

TriMesh(m::GeometryBasics.Mesh) = TriMesh(_load_meta(m::GeometryBasics.Mesh)...)

# covert TriMesh to GeometryBasics Mesh
function GBMesh(m::TriMesh)
    points = Point3f0[
        GeometryBasics.Point{3,Float32}(m.vertices[i, :]) for i = 1:size(m.vertices, 1)
    ]
    vert_len = size(m.faces, 2)
    poly_face = NgonFace{vert_len,UInt32}[
        NgonFace{vert_len,UInt32}(m.faces[i, :]) for i = 1:size(m.faces, 1)
    ]
    # faces = convert_simplex.(GLTriangleFace, poly_face)
    faces = GLTriangleFace.(poly_face)
    return Mesh(meta(points), faces)
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

_get_offset(x::GeometryBasics.OffsetInteger{o,T}) where {o,T} = o

function load_trimesh(fn; elements_types...)
    mesh = load(fn; elements_types...)
    vertices, faces = _load_meta(mesh)
    # # offset = _get_offset(x)   #offset is always -1 when loaded from MeshIO
    return TriMesh(vertices, faces)
end

# TODO: extend save function for obj file (as it is not implemented yet) and use that function here
function save_trimesh(file_name, mesh::TriMesh)
    error("Not implemented")
end

function compute_vertex_normals(m::TriMesh)
    vert_faces = m.vertices[m.faces, :]
    vertex_normals = Zygote.bufferfrom(zeros(Float32, size(m.vertices)...))

    vertex_normals[m.faces[:, 1], :] += _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    vertex_normals[m.faces[:, 2], :] += _lg_cross(
        vert_faces[:, 3, :] - vert_faces[:, 2, :],
        vert_faces[:, 1, :] - vert_faces[:, 2, :],
    )
    vertex_normals[m.faces[:, 3], :] += _lg_cross(
        vert_faces[:, 1, :] - vert_faces[:, 3, :],
        vert_faces[:, 2, :] - vert_faces[:, 3, :],
    )

    return _normalize(copy(vertex_normals), dims = 2)
end

function compute_face_normals(m::TriMesh)
    vert_faces = m.vertices[m.faces, :]
    face_normals = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    return _normalize(face_normals, dims = 2)
end

function compute_face_areas(m::TriMesh; compute_normals::Bool = true, eps::Number = 1e-6)
    vert_faces = m.vertices[m.faces, :]
    face_normals_vec = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    face_norm = sqrt.(sum(face_normals_vec .^ 2, dims = 2))
    face_areas = dropdims(face_norm ./ 2; dims = 2)
    if compute_normals
        face_normals = face_normals_vec ./ max.(face_norm, eps)
    else
        face_normals = nothing
    end
    return (face_areas, face_normals)
end

function get_edges(m::TriMesh, refresh::Bool = false)
    _compute_edges(m, refresh)
    return m.edges
end

function get_edges_to_key(m::TriMesh, refresh::Bool = false)
    _compute_edges(m, refresh)
    return m.edges_to_key
end

function get_faces_to_edges(m::TriMesh, refresh::Bool = false)
    _compute_edges(m, refresh)
    return m.faces_to_edges
end

function get_laplacian_sparse(m::TriMesh, refresh::Bool = false)
    _compute_laplacian_sparse(m, refresh)
    return m.laplacian_sparse
end

function _compute_edges(m::TriMesh, refresh::Bool = false)
    if refresh || (m.edges isa Nothing)
        e12 = cat(m.faces[:, 1], m.faces[:, 2], dims = 2)
        e23 = cat(m.faces[:, 2], m.faces[:, 3], dims = 2)
        e31 = cat(m.faces[:, 3], m.faces[:, 1], dims = 2)

        # Sort edges (v0, v1) such that v0 <= v1
        e12 = sort(e12; dims = 2)
        e23 = sort(e23; dims = 2)
        e31 = sort(e31; dims = 2)

        # Edges including duplicates
        edges = cat(e12, e23, e31, dims = 1)

        # Converting edge (v0, v1) into integer hash, ie. (V+1)*v0 + v1.
        # There will be no collision, which is asserted by (V+1), as 1<=v0<=V.
        V_hash = size(m.vertices, 1) + 1
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

        m.edges = edges
        m.edges_to_key = edges_to_key
        m.faces_to_edges = faces_to_edges
    end
end

function _old_compute_edges(m::TriMesh)
    e12 = cat(m.faces[:, 1], m.faces[:, 2], dims = 2)
    e23 = cat(m.faces[:, 2], m.faces[:, 3], dims = 2)
    e31 = cat(m.faces[:, 3], m.faces[:, 1], dims = 2)

    # Edges including duplicates
    edges = cat(e12, e23, e31, dims = 1)

    # Sort edges (v0, v1) such that v0 <= v1
    edges = sort(edges, dims = 2)

    # Remove duplicate edges
    edges = unique(edges; dims = 1)
    return edges
end

function _compute_laplacian_sparse(m::TriMesh, refresh::Bool = false)
    if refresh || (m.laplacian_sparse isa Nothing)
        edges = get_edges(m, refresh)

        e1 = edges[:, 1]
        e2 = edges[:, 2]

        idx12 = cat(e1, e2, dims = 2)
        idx21 = cat(e2, e1, dims = 2)
        idx = cat(idx12, idx21, dims = 1)

        A = sparse(
            idx[:, 1],
            idx[:, 2],
            ones(Float32, size(idx, 1)),
            size(m.vertices, 1),
            size(m.vertices, 1),
        )

        deg = Array{Float32}(sum(A, dims = 2))  # TODO: will be problematic for GPU

        deg1 = map(x -> (x > 0 ? 1 / x : x), deg[e1])
        deg2 = map(x -> (x > 0 ? 1 / x : x), deg[e2])
        diag = fill(-1.0f0, size(m.vertices, 1))

        Is = cat(e1, e2, UInt32.(1:size(m.vertices, 1)); dims = 1)
        Js = cat(e2, e1, UInt32.(1:size(m.vertices, 1)); dims = 1)
        Vs = cat(deg1, deg2, diag; dims = 1)
        m.laplacian_sparse = sparse(Is, Js, Vs, size(m.vertices, 1), size(m.vertices, 1))
    end
end
