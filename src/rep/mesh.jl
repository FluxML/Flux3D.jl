export TriMesh,
    GBMesh,
    gbmeshes,
    load_trimesh,
    save_trimesh,
    get_verts_list,
    get_verts_packed,
    get_verts_padded,
    get_faces_list,
    get_faces_packed,
    get_faces_padded,
    get_edges_packed,
    get_edges_to_key,
    get_faces_to_edges_packed,
    get_laplacian_packed,
    get_laplacian_sparse,
    compute_verts_normals_list,
    compute_verts_normals_packed,
    compute_verts_normals_padded,
    compute_faces_normals_list,
    compute_faces_normals_packed,
    compute_faces_normals_padded,
    compute_faces_areas_list,
    compute_faces_areas_packed,
    compute_faces_areas_padded

import GeometryBasics, Printf, MeshIO
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
- `TriMesh(m::TriMesh)`

"""
mutable struct TriMesh{T<:AbstractFloat,R<:Integer,S} <: AbstractObject
    N::Int64
    V::Int64
    F::Int64
    equalised::Bool
    valid::BitArray{1}
    offset::Int8
    _verts_len::S
    _faces_len::S

    _verts_packed::S
    _verts_padded::S
    _verts_list::Vector{<:S}
    _verts_packed_valid::Bool
    _verts_padded_valid::Bool
    _verts_list_valid::Bool

    _faces_packed::Array{R,2}
    _faces_padded::Array{R,3}
    _faces_list::Vector{Array{R,2}}
    _faces_packed_valid::Bool
    _faces_padded_valid::Bool

    _edges_packed::Union{Nothing,Array{R,2}}
    _faces_to_edges_packed::Union{Nothing,Array{R,2}}
    _laplacian_packed::Union{Nothing,AbstractSparseArray{T,R,2}}

    _edges_to_key::Union{Nothing,Dict{Tuple{R,R},R}}
end

# TriMesh(
#     verts::Vector{<:AbstractArray{T,2}},
#     faces::Vector{<:AbstractArray{R,2}};
#     offset::Number = -1,
# ) where {T<:Number,R<:Number} =
#     TriMesh([Float32.(v) for v in verts], [UInt32.(f) for f in faces]; offset = offset)

TriMesh(
    verts::Vector{<:CuArray{T,2}},
    faces::Vector{<:AbstractArray{R,2}};
    offset::Number = -1,
) where {T<:AbstractFloat,R<:Integer} = TriMesh(CuArray, verts, faces; offset = offset)

TriMesh(
    verts::Vector{<:Array{T,2}},
    faces::Vector{<:AbstractArray{R,2}};
    offset::Number = -1,
) where {T<:AbstractFloat,R<:Integer} = TriMesh(Array, verts, faces; offset = offset)

function TriMesh(
    ::Type{S},
    verts::Vector{<:AbstractArray{T,2}},
    faces::Vector{<:AbstractArray{R,2}};
    offset::Number = -1,
) where {S,T<:AbstractFloat,R<:Integer}

    length(verts) == length(faces) ||
        error("batch size of verts and faces should match, $(length(verts)) != $(length(faces))")

    # To remove lazy wrappers on verts and faces list
    verts = [T.(v) for v in verts]
    faces = [R.(f) for f in faces]

    _verts_len = S(size.(verts, 2))
    _faces_len = S(size.(faces, 2))

    N = length(verts)
    V = maximum(_verts_len)
    F = maximum(_faces_len)
    equalised = all(_verts_len .== V) && all(_faces_len .== F)
    valid = _faces_len .> 0
    offset = Int8(offset)

    _verts_list = verts::Vector{<:S{T,2}}
    _faces_list = faces::Vector{Array{R,2}}

    return TriMesh{T,R,S}(
        N,
        V,
        F,
        equalised,
        valid,
        offset,
        _verts_len,
        _faces_len,
        S{T,2}(undef, 3, sum(_verts_len)),
        S{T,3}(undef, 3, V, N),
        _verts_list,
        false,
        false,
        true,
        Array{R,2}(undef, 3, sum(_faces_len)),
        Array{R,3}(undef, 3, F, N),
        _faces_list,
        false,
        false,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function TriMesh(ms::Vector{<:GeometryBasics.Mesh})
    verts_list = Array{Float32,2}[]
    faces_list = Array{UInt32,2}[]
    for (i, m) in enumerate(ms)
        (verts, faces) = _load_meta(m)
        push!(verts_list, verts)
        push!(faces_list, faces)
    end
    return TriMesh(verts_list, faces_list)
end

TriMesh(m::GeometryBasics.Mesh) = TriMesh([m])
TriMesh(m::TriMesh) = TriMesh(get_verts_list(m), get_faces_list(m))

# @functor TriMesh
functor(x::TriMesh) = (_verts_list = x._verts_list,),
xs -> TriMesh(xs._verts_list, x._faces_list; offset = x.offset)

function Base.show(io::IO, m::TriMesh{T,R,S}) where {T,R,S}
    print(
        io,
        "TriMesh{$(T), $(R), $(S)} Structure:\n    Batch size: ",
        m.N,
        "\n    Max verts: ",
        m.V,
        "\n    Max faces: ",
        m.F,
        "\n    offset: ",
        m.offset,
        "\n    Storage type: ",
        S,
    )
end

function Base.setproperty!(m::TriMesh, f::Symbol, v)
    if (f == :_verts_packed) || (f == :_verts_padded) || (f == :_verts_list)
        # only attempt to change if given object is not same as object already
        # present inside struct, which can be checked with !== operator.
        # List is always valid, so to make sure that we refresh list always
        if (f == :_verts_packed) && (getproperty(m, f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_padded_valid, false)
            setfield!(m, :_verts_list_valid, false)
            # _compute_verts_list(m, true)
        elseif (f == :_verts_padded) && (getproperty(m, f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_packed_valid, false)
            setfield!(m, :_verts_list_valid, false)
            # _compute_verts_list(m, true)
        elseif (f == :_verts_list) && (getproperty(m, f) !== v)
            setfield!(m, f, convert(fieldtype(typeof(m), f), v))
            setfield!(m, :_verts_packed_valid, false)
            setfield!(m, :_verts_padded_valid, false)
        end
    else
        setfield!(m, f, convert(fieldtype(typeof(m), f), v))
    end
end

Base.getindex(m::TriMesh, inds...) = (m._verts_list[inds...], m._faces_list[inds...])

"""
    GBMesh(m::TriMesh; index::Int = 1)
    GBMesh(verts::AbstractArray{T,2}, faces::AbstractArray{R,2}) where {T,R}

Initialize GeometryBasics.Mesh from triangle mesh in TriMesh `m` at `index`.

See also: [`gbmeshes`](@ref)
"""
function GBMesh(verts::AbstractArray{T,2}, faces::AbstractArray{R,2}) where {T,R}
    points = Point3f0[GeometryBasics.Point{3,Float32}(verts[:, i]) for i = 1:size(verts, 2)]
    verts_dim = size(faces, 1)
    poly_face = NgonFace{verts_dim,UInt32}[
        NgonFace{verts_dim,UInt32}(faces[:, i]) for i = 1:size(faces, 2)
    ]
    # faces = convert_simplex.(GLTriangleFace, poly_face)
    faces = GLTriangleFace.(poly_face)
    return GeometryBasics.Mesh(meta(points), faces)
end

GBMesh(m::TriMesh, index::Int = 1) = GBMesh(m[index]...)

"""
    gbmeshes(m::TriMesh)

Initialize list of GeometryBasics.Mesh from TriMesh `m`

See also: [`gbmeshes`](@ref)
"""
function gbmeshes(m::TriMesh)
    gbmeshes = [GBMesh(m, i) for i = 1:m.N]
    return gbmeshes
end

"""
    _load_meta(m::GeometryBasics.Mesh)

Returns `vertices` and `faces` in Array format.

"""
function _load_meta(m::GeometryBasics.Mesh)
    if !(m isa GeometryBasics.Mesh{3,Float32,<:GeometryBasics.Triangle})
        m = triangle_mesh(m)
    end
    vs = m.position
    vertices = [Array(v) for v in vs]
    vertices = reduce(hcat, vertices)
    fs = getfield(getfield(m, :simplices), :faces)
    faces = [Array(UInt32.(f)) for f in fs]
    faces = reduce(hcat, faces)
    return (vertices, faces)
end

"""
    load_trimesh(fn::String)
    load_trimesh(fns::Vector{String})

Load TriMesh from file(s).
It will load TriMesh with multiple meshes, if list of files `fns` is given.

Supported formats are `obj`, `stl`, `ply`, `off` and `2DM`.

"""
function load_trimesh(fn::String; elements_types...)
    mesh = load(fn; elements_types...)
    verts, faces = _load_meta(mesh)
    # offset is always -1 when loaded from MeshIO
    return TriMesh([verts], [faces])
end

function load_trimesh(fns::Vector{String}; elements_types...)
    verts_list = Array{Float32,2}[]
    faces_list = Array{UInt32,2}[]
    for (i, fn) in enumerate(fns)
        mesh = load(fn; elements_types...)
        verts, faces = _load_meta(mesh)
        push!(verts_list, verts)
        push!(faces_list, faces)
    end
    return TriMesh(verts_list, faces_list)
end

"""
    save_trimesh(fn::String, mesh::TriMesh, index::Int = 1)
    save_trimesh(fn::String, mesh::GeometryBasics.Mesh)

Save mesh in given `fn`.
`index` is an optional argument specifing the index of mesh,
incase of multiple meshes in TriMesh `mesh`.

Supported formats are `obj`, `stl`, `ply`, `off` and `2DM`.

"""
save_trimesh(fn::String, mesh::TriMesh, index::Int = 1) =
    save_trimesh(fn, GBMesh(mesh, index))

function save_trimesh(fn::String, mesh::GeometryBasics.Mesh)
    save(fn, mesh)
end

# overloading MeshIO.save function to support saving obj file.
function MeshIO.save(str::Stream{format"OBJ"}, msh::GeometryBasics.AbstractMesh)
    io = stream(str)
    vts = GeometryBasics.coordinates(msh)
    fcs = GeometryBasics.faces(msh)

    # write header
    write(io, "# Flux3D.jl v0.1.0 OBJ File: \n")
    write(io, "# www.github.com/FluxML/Flux3D.jl \n")

    # write vertices data
    for v in vts
        Printf.@printf(io, "v %.6f %.6f %.6f\n", v...)
    end

    # write faces data
    for f in fcs
        Printf.@printf(io, "f %d %d %d\n", Int.(f)...)
    end

    close(io)
end

"""
    get_verts_packed(m::TriMesh; refresh::Bool = false)

Returns vertices of TriMesh `m` in packed format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_verts_padded`](@ref), [`get_verts_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_verts_packed(m)
```
"""
function get_verts_packed(m::TriMesh{T,R,S}; refresh::Bool = false)::S{T,2} where {T,R,S}
    _compute_verts_packed(m, refresh)
    return m._verts_packed
end

"""
    get_verts_padded(m::TriMesh; refresh::Bool = false)

Returns vertices of TriMesh `m` in padded format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_verts_packed`](@ref), [`get_verts_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_verts_padded(m)
```
"""
function get_verts_padded(m::TriMesh{T,R,S}; refresh::Bool = false)::S{T,3} where {T,R,S}
    _compute_verts_padded(m, refresh)
    return m._verts_padded
end

"""
    get_verts_list(m::TriMesh; refresh::Bool = false)

Returns vertices of TriMesh `m` in list format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_verts_padded`](@ref), [`get_verts_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_verts_list(m)
```
"""
function get_verts_list(
    m::TriMesh{T,R,S};
    refresh::Bool = false,
)::Vector{<:S{T,2}} where {T,R,S}
    _compute_verts_list(m, refresh)
    return m._verts_list
end

"""
    get_faces_packed(m::TriMesh; refresh::Bool = false)

Returns faces of TriMesh `m` in packed format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_faces_padded`](@ref), [`get_faces_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_faces_packed(m)
```
"""
function get_faces_packed(m::TriMesh{T,R}; refresh::Bool = false)::Array{R,2} where {T,R}
    @ignore _compute_faces_packed(m, refresh)
    return m._faces_packed
end

"""
    get_faces_padded(m::TriMesh; refresh::Bool = false)

Returns faces of TriMesh `m` in padded format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_faces_padded`](@ref), [`get_faces_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_faces_padded(m)
```
"""
function get_faces_padded(m::TriMesh{T,R}; refresh::Bool = false)::Array{R,3} where {T,R}
    @ignore _compute_faces_padded(m, refresh)
    return m._faces_padded
end

"""
    get_faces_list(m::TriMesh; refresh::Bool = false)

Returns faces of TriMesh `m` in list format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_faces_padded`](@ref), [`get_faces_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_faces_list(m)
```
"""
function get_faces_list(
    m::TriMesh{T,R};
    refresh::Bool = false,
)::Vector{Array{R,2}} where {T,R}
    return m._faces_list
end

"""
    get_edges_packed(m::TriMesh; refresh::Bool = false)

Returns edges of TriMesh `m` in packed format. Edges are according
to the indices of corresponding vertices in `get_verts_packed(m)`

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_verts_packed`](@ref), [`get_faces_to_edges_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_edges_packed(m)
```
"""
function get_edges_packed(m::TriMesh{T,R}; refresh::Bool = false)::Array{R,2} where {T,R}
    @ignore _compute_edges_packed(m, refresh)
    return m._edges_packed
end

"""
    get_edges_to_key(m::TriMesh; refresh::Bool = false)

Returns dict mapping edges (tuple) of TriMesh `m` to unique key.
Edges are according to the indices of corresponding vertices in`get_verts_packed(m)`

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`get_verts_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_edges_to_key(m)
```
"""
function get_edges_to_key(
    m::TriMesh{T,R};
    refresh::Bool = false,
)::Dict{Tuple{R,R},R} where {T,R}
    @ignore _compute_edges_packed(m, refresh)
    return m._edges_to_key
end

"""
    get_faces_to_edges_packed(m::TriMesh; refresh::Bool = false)

Returns faces of TriMesh `m` in form of edges.
Each edge corresponds to the indices of `get_edges_packed(m)`.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`. Assuming face f consists of
(v1,v2,v3) vertices, and e1 = {v2,v3}, e2 = {v3,v1}, e3 = {v1,v2},
so face f in form of edges would be (e1,e2,e3).

See also: [`get_edges_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_faces_to_edges_packed(m)
```
"""
function get_faces_to_edges_packed(
    m::TriMesh{T,R};
    refresh::Bool = false,
)::Array{R,2} where {T,R}
    @ignore _compute_edges_packed(m, refresh)
    return m._faces_to_edges_packed
end

"""
    get_laplacian_packed(m::TriMesh; refresh::Bool = false)
    get_laplacian_sparse(m::TriMesh; refresh::Bool = false)

Returns Laplacian sparce matrix of TriMesh `m`.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`laplacian_loss`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> get_laplacian_packed(m)
```
"""
function get_laplacian_packed(
    m::TriMesh{T,R};
    refresh::Bool = false,
)::SparseMatrixCSC{T,R} where {T,R}
    _compute_laplacian_packed(m, refresh)
    return m._laplacian_packed
end

const get_laplacian_sparse = get_laplacian_packed

"""
    compute_verts_normals_packed(m::TriMesh)

Computes Unit normal of vertices of TriMesh `m` in packed format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.
Normal vec of each vertex is weighted sum of normals
of adjacent faces, weighted by corresponding faces areas
and then normalize to unit vector.

See also: [`compute_verts_normals_padded`](@ref), [`compute_verts_normals_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_verts_normals_packed(m)
```
"""
function compute_verts_normals_packed(m::TriMesh{T,R,S})::S{T,2} where {T,R,S}

    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[:, faces]
    vertex_normals = similar(verts, size(verts))
    vertex_normals = @ignore fill!(vertex_normals, 0.0)
    vertex_normals = Zygote.bufferfrom(vertex_normals)

    # normal of each vertex is sum of normals of its faces weighted by
    # corresponding areas, ie. (A1 .* fn1) + (A2 .* fn2) + (A2 .* fn2)
    # where fn1, fn2, fn3 are normals of faces shared by that vertex
    # and A1, A2, A3 are corresponding areas.

    vertex_normals[:, faces[1, :]] += _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    vertex_normals[:, faces[2, :]] += _lg_cross(
        vert_faces[:, 3, :] - vert_faces[:, 2, :],
        vert_faces[:, 1, :] - vert_faces[:, 2, :],
    )
    vertex_normals[:, faces[3, :]] += _lg_cross(
        vert_faces[:, 1, :] - vert_faces[:, 3, :],
        vert_faces[:, 2, :] - vert_faces[:, 3, :],
    )

    return _normalize(copy(vertex_normals), dims = 1)
end

"""
    compute_verts_normals_padded(m::TriMesh)

Computes Unit normal of vertices of TriMesh `m` in padded format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.
Normal vec of each vertex is weighted sum of normals
of adjacent faces, weighted by corresponding faces areas
and then normalize to unit vector.

See also: [`compute_verts_normals_packed`](@ref), [`compute_verts_normals_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_verts_normals_padded(m)
```
"""
function compute_verts_normals_padded(m::TriMesh{T,R,S})::S{T,3} where {T,R,S}
    normals_packed = compute_verts_normals_packed(m)
    normals_padded = _packed_to_padded(normals_packed, m._verts_len, 0.0)
    return normals_padded
end

"""
    compute_verts_normals_list(m::TriMesh)

Computes Unit normal of vertices of TriMesh `m` in list format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.
Normal vec of each vertex is weighted sum of normals
of adjacent faces, weighted by corresponding faces areas
and then normalize to unit vector.

See also: [`compute_verts_normals_padded`](@ref), [`compute_verts_normals_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_verts_normals_list(m)
```
"""
function compute_verts_normals_list(m::TriMesh{T,R,S})::Vector{<:S{T,2}} where {T,R,S}
    normals_packed = compute_verts_normals_packed(m)
    normals_list = _packed_to_list(normals_packed, m._verts_len)
    return normals_list
end

"""
    compute_faces_normals_packed(m::TriMesh)

Computes Unit normal of faces of TriMesh `m` in packed format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_normals_padded`](@ref), [`compute_faces_normals_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_normals_packed(m)
```
"""
function compute_faces_normals_packed(m::TriMesh{T,R,S})::S{T,2} where {T,R,S}
    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[:, faces]
    # normal vec of face [f1, f2, f3] is (f2-f1)X(f3-f1)
    face_normals = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    return _normalize(face_normals, dims = 1)
end

"""
    compute_faces_normals_padded(m::TriMesh)

Computes Unit normal of faces of TriMesh `m` in padded format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_normals_packed`](@ref), [`compute_faces_normals_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_normals_padded(m)
```
"""
function compute_faces_normals_padded(m::TriMesh{T,R,S})::S{T,3} where {T,R,S}
    normals_packed = compute_faces_normals_packed(m)
    normals_padded = _packed_to_padded(normals_packed, m._faces_len, 0.0)
    return normals_padded
end

"""
    compute_faces_normals_list(m::TriMesh)

Computes Unit normal of faces of TriMesh `m` in list format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_normals_padded`](@ref), [`compute_faces_normals_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_normals_list(m)
```
"""
function compute_faces_normals_list(m::TriMesh{T,R,S})::Vector{<:S{T,2}} where {T,R,S}
    normals_packed = compute_faces_normals_packed(m)
    normals_list = _packed_to_list(normals_packed, m._faces_len)
    return normals_list
end

"""
    compute_faces_areas_packed(m::TriMesh)

Computes area of faces of TriMesh `m` in packed format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_areas_padded`](@ref), [`compute_faces_areas_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_areas_packed(m)
```
"""
function compute_faces_areas_packed(
    m::TriMesh{T,R,S};
    eps::Number = 1e-6,
)::S{T,1} where {T,R,S}
    verts = get_verts_packed(m)
    faces = get_faces_packed(m)

    vert_faces = verts[:, faces]
    face_normals_vec = _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    )
    face_norm = _norm(face_normals_vec; dims = 1)
    face_areas = dropdims(face_norm ./ 2; dims = 1)
    return face_areas
end

"""
    compute_faces_areas_padded(m::TriMesh)

Computes area of faces of TriMesh `m` in padded format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_areas_packed`](@ref), [`compute_faces_areas_list`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_areas_padded(m)
```
"""
function compute_faces_areas_padded(
    m::TriMesh{T,R,S};
    eps::Number = 1e-6,
)::S{T,3} where {T,R,S}
    areas_packed = compute_faces_areas_packed(m; eps = eps)
    # increasing ndims to 2, to support _packed_to_list
    areas_packed = reshape(areas_packed, 1, :)
    areas_padded = _packed_to_padded(areas_packed, m._faces_len, 0.0)
    return areas_padded
end

"""
    compute_faces_areas_list(m::TriMesh)

Computes area of faces of TriMesh `m` in list format.

`refresh` is an optional keyword argument to
recompute this format for TriMesh `m`.

See also: [`compute_faces_areas_padded`](@ref), [`compute_faces_areas_packed`](@ref)

### Examples:

```jldoctest
julia> m = load_trimesh("teapot.obj")
julia> compute_faces_areas_list(m)
```
"""
function compute_faces_areas_list(
    m::TriMesh{T,R,S};
    eps::Number = 1e-6,
)::Vector{<:S{T,2}} where {T,R,S}
    areas_packed = compute_faces_areas_packed(m; eps = eps)
    # increasing ndims to 2, to support _packed_to_list
    areas_packed = reshape(areas_packed, 1, :)
    areas_list = _packed_to_list(areas_packed, m._faces_len)
    return areas_list
end

function _compute_verts_packed(m::TriMesh, refresh::Bool = false)
    if refresh || !(m._verts_packed_valid)
        verts_packed = _list_to_packed(m._verts_list)
        # avoiding setproperty!, as we are building packed
        # from list and list is always valid
        setfield!(m, :_verts_packed, verts_packed)
        setfield!(m, :_verts_packed_valid, true)
        return nothing
    end
end

function _compute_verts_padded(m::TriMesh, refresh::Bool = false)
    if refresh || !(m._verts_padded_valid)
        _list_to_padded!(m._verts_padded, m._verts_list, 0, (3, m.V))
        # avoiding setproperty!, as we are building padded
        # from list and list is always valid
        # setfield!(m, :_verts_padded, verts_padded)
        setfield!(m, :_verts_padded_valid, true)
        return nothing
    end
end

function _compute_verts_list(m::TriMesh, refresh::Bool = false)
    if refresh || !(m._verts_list_valid)
        if m._verts_packed !== nothing
            verts_list = _packed_to_list(m._verts_packed, m._verts_len)
        elseif m._verts_padded !== nothing
            verts_list = _padded_to_list(m._verts_padded, m._verts_len)
        else
            error("not possible to contruct list without padded and packed")
        end
        # avoiding setproperty, cause verts_list is always valid.
        setfield!(m, :_verts_list, verts_list)
        setfield!(m, :_verts_list_valid, true)
        return nothing
    end
end

function _compute_faces_packed(m::TriMesh{T,R}, refresh::Bool = false) where {T,R}
    if refresh || !(m._faces_packed_valid)
        faces_packed = _list_to_packed(m._faces_list)
        _, verts_packed_first_idx, _ = _auxiliary_mesh(m._verts_list)
        _, _, faces_packed_list_idx = _auxiliary_mesh(m._faces_list)
        # offset will be equal to first idx in verts_packed of corresponding faces_idx-1
        faces_packed_offset = verts_packed_first_idx[faces_packed_list_idx] .- 1
        faces_packed = faces_packed .+ reshape(faces_packed_offset, 1, :)
        setfield!(m, :_faces_packed, R.(faces_packed))
        setfield!(m, :_faces_packed_valid, true)
        return nothing
    end
end

function _compute_faces_padded(m::TriMesh, refresh::Bool = false)
    if refresh || !(m._faces_padded_valid)
        _list_to_padded!(m._faces_padded, m._faces_list, 0, (3, m.F))
        # setfield!(m, :_faces_padded, faces_padded)
        setfield!(m, :_faces_padded_valid, true)
        return nothing
    end
end

function _compute_edges_packed(m::TriMesh{T,R}, refresh::Bool = false) where {T,R}
    if refresh ||
       (any([m._edges_packed, m._edges_to_key, m._faces_to_edges_packed] .=== nothing))

        faces = get_faces_packed(m)
        verts = get_verts_packed(m)

        e12 = cat(faces[1, :], faces[2, :], dims = 2)
        e23 = cat(faces[2, :], faces[3, :], dims = 2)
        e31 = cat(faces[3, :], faces[1, :], dims = 2)

        # Sort edges (v0, v1) such that v0 <= v1
        e12 = sort(e12; dims = 2)
        e23 = sort(e23; dims = 2)
        e31 = sort(e31; dims = 2)

        # Edges including duplicates
        edges = cat(e12, e23, e31, dims = 1)

        # Converting edge (v0, v1) into integer hash, ie. (V+1)*v0 + v1.
        # There will be no collision, which is asserted by (V+1), as 1<=v0<=V.
        V_hash = size(verts, 2) + 1
        edges_hash = (V_hash .* edges[:, 1]) .+ edges[:, 2]

        # Sort and remove duplicate edges_hash
        sort!(edges_hash)
        unique!(edges_hash)

        # Convert edges_hash to edges
        edges = cat((edges_hash .÷ V_hash), (edges_hash .% V_hash); dims = 2)

        # Edges to key
        edges_to_key =
            Dict{Tuple{R,R},R}([(Tuple(edges[i, :]), i) for i = 1:size(edges, 1)])

        # e12 -> tuple -> get
        e12_tup = [Tuple(e12[i, :]) for i = 1:size(e12, 1)]
        e23_tup = [Tuple(e23[i, :]) for i = 1:size(e23, 1)]
        e31_tup = [Tuple(e31[i, :]) for i = 1:size(e31, 1)]
        faces_to_edges_tuple = cat(e23_tup, e31_tup, e12_tup; dims = 2)

        faces_to_edges = map(x -> get(edges_to_key, x, -1), faces_to_edges_tuple)

        m._edges_packed = edges
        m._edges_to_key = edges_to_key
        m._faces_to_edges_packed = faces_to_edges
        return nothing
    end
end

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
            fill(1, size(idx, 1)),
            size(verts, 2),
            size(verts, 2),
        )

        # computing degree of each vertices
        deg = sum(A, dims = 2)

        # if degree >= 1, then there exist edge with that vertex,
        # so computing 1/deg[i]
        deg1 = map(x -> T.(x > 0 ? 1 / x : x), deg[e1])
        deg2 = map(x -> T.(x > 0 ? 1 / x : x), deg[e2])
        diag = fill(T.(-1.0), size(verts, 2))

        # assuming (i,j) is an edge, so we computing 1/deg[i],
        # and store at L[i,j] and same for (j,i).
        # We also store -1 for L[i,i] for all vertices.
        Is = cat(e1, e2, R.(1:size(verts, 2)); dims = 1)
        Js = cat(e2, e1, R.(1:size(verts, 2)); dims = 1)
        Vs = cat(deg1, deg2, diag; dims = 1)
        m._laplacian_packed = sparse(Is, Js, Vs, size(verts, 2), size(verts, 2))
        return nothing
    end
end
