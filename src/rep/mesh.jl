# export TriMesh

mutable struct TriMesh{T<:Float32,R<:UInt32} <: AbstractMesh
    vertices::AbstractArray{2,T}
    faces::AbstractArray{2,R}
    offset::Int8
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

# TODO: Implement two variants, face_normal and vertex_normal
function get_normal(m::TriMesh) 
    error("Not implemented")
end

# TODO: extend save function for obj file (as it is not implemented yet) and use that function here
function save_trimesh(file_name, mesh::TriMesh)
    error("Not implemented")
end

_get_offset(x::GeometryBasics.OffsetInteger{o,T}) where {o, T} = o