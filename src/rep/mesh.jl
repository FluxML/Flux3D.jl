# export TriMesh
import GeometryBasics

# TODO: add texture fields
mutable struct TriMesh{T<:Float32,R<:UInt32} <: AbstractMesh
    vertices::AbstractArray{T,2}
    faces::AbstractArray{R,2}
    offset::Int8
end

# TODO: Add contructor according to batched format
function TriMesh(vertices::AbstractArray{<:Number,2}, faces::AbstractArray{<:Number,2}; offset::Number=-1)
    vertices = Float32.(vertices)
    faces = UInt32.(faces)
    offset = Int8(offset)
    TriMesh(vertices, faces, offset)
end

function load_trimesh(fn; elements_types...)
    mesh = load(fn; elements_types...)
    v = mesh.position
    vertices = cat([reshape(Array.(pts), 1, :) for pts in v]..., dims=1)
    names = propertynames(getfield(mesh, :simplices))
    f = getfield(getfield(mesh, :simplices), :faces)
    offset = _get_offset(f[1][1])
    faces = cat([reshape(Array(UInt32.(fi)),1,:) for fi in f]..., dims=1)
    return TriMesh(vertices, faces, offset)
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


# TODO: extend save function for obj file (as it is not implemented yet) and use that function here
function save_trimesh(file_name, mesh::TriMesh)
    error("Not implemented")
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

_normalize(A::AbstractArray; eps::Number=1e-6, dims::Int=2) = A ./ max.(sqrt.(sum(A .^ 2,dims=dims)),eps) 