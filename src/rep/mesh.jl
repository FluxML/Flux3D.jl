# export TriMesh
import GeometryBasics

# TODO: add texture fields
mutable struct TriMesh{T<:Float32,R<:UInt32} <: AbstractMesh
    vertices::AbstractArray{T,2}
    faces::AbstractArray{R,2}
    offset::Int8
    edges::Union{Nothing, AbstractArray{Float32,2}}
    laplacian_sparse::Union{Nothing, AbstractSparseMatrix{UInt32, Float32}}
end

# TODO: Add contructor according to batched format
function TriMesh(vertices::AbstractArray{<:Number,2}, faces::AbstractArray{<:Number,2}; offset::Number=-1)
    vertices = Float32.(vertices)
    faces = UInt32.(faces)
    offset = Int8(offset)
    TriMesh(vertices, faces, offset, nothing, nothing)
end

function load_trimesh(fn; elements_types...)
    mesh = load(fn; elements_types...)
    v = mesh.position
    vertices = cat([reshape(Array.(pts), 1, :) for pts in v]..., dims=1)
    names = propertynames(getfield(mesh, :simplices))
    f = getfield(getfield(mesh, :simplices), :faces)
    offset = _get_offset(f[1][1])
    faces = cat([reshape(Array(UInt32.(fi)),1,:) for fi in f]..., dims=1)
    return TriMesh(vertices, faces; offset=offset)
end

function vertex_normals(m::TriMesh)
    vert_faces = m.vertices[m.faces, :]
    vertex_normals = Zygote.bufferfrom(zeros(Float32, size(m.vertices)...))

    vertex_normals[m.faces[:,1],:] += _lg_cross(vert_faces[:,2,:] - vert_faces[:,1,:],vert_faces[:,3,:] - vert_faces[:,1,:])
    vertex_normals[m.faces[:,2],:] += _lg_cross(vert_faces[:,3,:] - vert_faces[:,2,:],vert_faces[:,1,:] - vert_faces[:,2,:])
    vertex_normals[m.faces[:,3],:] += _lg_cross(vert_faces[:,1,:] - vert_faces[:,3,:],vert_faces[:,2,:] - vert_faces[:,3,:])

    _normalize(copy(vertex_normals), dims=2)
end

function face_normals(m::TriMesh)
    vert_faces = m.vertices[m.faces, :]
    face_normals = _lg_cross(vert_faces[:,2,:] - vert_faces[:,1,:],vert_faces[:,3,:] - vert_faces[:,1,:])
    _normalize(face_normals, dims=2)
end

function face_areas(m::TriMesh; compute_normals::Bool=true, eps::Number=1e-6)
    vert_faces = m.vertices[m.faces, :]
    face_normals_vec = _lg_cross(vert_faces[:,2,:] - vert_faces[:,1,:],vert_faces[:,3,:] - vert_faces[:,1,:])
    face_norm = sqrt.(sum(face_normals_vec .^ 2, dims=2))
    face_areas = face_norm ./ 2
    if compute_normals
        face_normals = face_normals_vec ./ max.(face_norm, eps)
    else
        face_normals = nothing
    end
    return (face_areas, face_normals)
end

function get_edges(m::TriMesh)
    _compute_edge(m)
    m.edges
end

function get_laplacian_sparse(m::TriMesh)
    _compute_laplacian_sparse(m)
    m.laplacian_sparse
end

function laplacian_loss(m::TriMesh)
    # TODO: There will be some changes when migrating to batched format
    L = get_laplacian_sparse(m)
    L = L * m.vertices
    L = _norm(L; dims=2)
    loss = sum(L)
    loss
end

# TODO: extend save function for obj file (as it is not implemented yet) and use that function here
function save_trimesh(file_name, mesh::TriMesh)
    error("Not implemented")
end

function _compute_edge(m::TriMesh)
    if m.edges isa Nothing
        e12 = cat(m.vertices[:,1], m.vertices[:,2], dims=2)
        e23 = cat(m.vertices[:,2], m.vertices[:,3], dims=2)
        e31 = cat(m.vertices[:,3], m.vertices[:,1], dims=2)

        # Edges including duplicates
        edges = cat(e12, e23, e31, dims=1)
        
        # Sort edges (v0, v1) such that v0 <= v1
        edges = sort(edges, dims=2)
        # Remove duplicate edges
        edges = unique(edges, dims=1)
        m.edges = edges
    end
end

function _compute_laplacian_sparse(m::TriMesh)
    if m.laplacian_sparse isa Nothing
        _compute_edge(m)
        
        e1 = m.edges[:,1]
        e2 = m.edges[:,2]

        idx12 = cat(m.edges[:,1], m.edges[:,2],dims=2)
        idx21 = cat(m.edges[:,1], m.edges[:,2],dims=2)
        idx = cat(idx12,idx21, dims=1)

        A = sparse(idx[:,1], idx[:,2], ones(Float32, size(idx,1)), size(m.vertices,1), size(m.vertices,1))

        deg = Array(sum(A,dims=2))  # TODO: will be problematic for GPU

        deg1 = 1 ./ map(x-> (x>0 ? 1/x : x), deg[e1])
        deg2 = 1 ./ map(x-> (x>0 ? 1/x : x), deg[e2])
        diag = fill(-1.0f0, size(m.vertices, 1))

        Is = cat(e1, e2, 1:size(m.vertices, 1); dims=1)
        Js = cat(e2, e1, 1:size(m.vertices, 1); dims=1)
        Vs = cat(deg1, deg2, diag; dims=1)
        m.laplacian_sparse = sparse(Is, Js, Vs, size(m.vertices, 1), size(m.vertices, 1))
    end
end


_get_offset(x::GeometryBasics.OffsetInteger{o,T}) where {o, T} = o

function _lg_cross(A::AbstractArray, B::AbstractArray)
    if !(size(A,2) == size(B,2) == 3)
        throw(DimensionMismatch("cross product is only defined for AbstractArray of dimension 3 at dims 2"))
    end
    a1, a2, a3 = A[:,1], A[:,2], A[:,3] 
    b1, b2, b3 = B[:,1], B[:,2], B[:,3]
    cat((a2 .* b3) - (a3 .* b2), (a3 .* b1) - (a1 .* b3), (a1 .* b2) - (a2 .* b1); dims=2)
end

_normalize(A::AbstractArray; eps::Number=1e-6, dims::Int=2) = A ./ max.(sqrt.(sum(A .^ 2; dims=dims)),eps) 

_norm(A::AbstractArray; dims::Int=2) = sqrt.(sum(A .^ 2; dims=dims))