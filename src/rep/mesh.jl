# export TriMesh

# TODO: discuss type for faces (use simple array or use GeometryBasics faces)
mutable struct TriMesh{T<:Float32} <: AbstractMesh
    vertices::AbstractArray{2,T}
    faces::GeometryBasics.NgonFace{3,GeometryBasics.OffsetInteger{-1,UInt32}}
end

function load_trimesh(fn; elements_types...)
    mesh = load(fn; elements_types...)
    v = mesh.position
    v = cat([reshape(Array.(pts), 1, :) for pts in v]..., dims=1)
    names = propertynames(getfield(mesh, :simplices))
    f = getfield(getfield(mesh, :simplices), :faces)
    return TriMesh(v,f)
end

# TODO: Implement two variants, face_normal and vertex_normal
function get_normal(m::TriMesh) 
    error("Not implemented")
end

# TODO: extend save function for obj file (as it is not implemented yet) and use that function here
function save_trimesh(file_name, mesh::TriMesh)
    error("Not implemented")
end
