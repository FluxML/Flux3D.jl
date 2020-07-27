export sample_points

import Distributions

"""
    sample_points(m::TriMesh, num_samples::Int=5000; returns_normals::Bool=false, eps::Number = 1e-6)

Uniformly samples `num_samples` points from the surface of TriMesh `m`.

`returns_normals` is optional keyword argument, to returns normals
from respective faces of samples. `eps` is optional keyword argument for
a small number to prevent division by zero for small surface areas.

### Examples:
julia> m = load_trimesh("teapot.obj")
julia> points = sample_points(m, 5000)
julia> points, normals = sample_points(m, 5000; returns_normals=true)

"""
function sample_points(
    m::TriMesh{T,R,S},
    num_samples::Int = 5000;
    eps::Number = EPS,
)::S{T,3} where {T,R,S}
    verts_padded = get_verts_padded(m)
    faces_padded = get_faces_padded(m)
    faces_areas_padded = compute_faces_areas_padded(m)

    #(Fi,B)
    #TODO: condition for probvec fails in Float32
    faces_areas_prob = Zygote.ignore() do
        faces_areas_padded = Float64.(faces_areas_padded)
        faces_areas_prob = faces_areas_padded ./ max.(sum(faces_areas_padded; dims = 2), eps)
        faces_areas_prob[:,end,:] += reshape(1 .- sum(faces_areas_prob; dims=2),1,:)
        return faces_areas_prob
    end
    samples = @ignore similar(verts_padded, 3, num_samples, m.N)
    samples = Zygote.bufferfrom(samples)

    for (i, _len) in enumerate(m._faces_len)
        probvec = faces_areas_prob[1, 1:_len, i]
        dist = Distributions.Categorical(probvec)
        sample_faces_idx = @ignore rand(dist, num_samples)
        sample_faces = faces_padded[:, sample_faces_idx, i]
        samples[:, :, i] =
            _sample_points(S,verts_padded[:,1:m._verts_len[i],i], sample_faces, num_samples)
    end

    return copy(samples)
end

function _sample_points(
    ::Type{S},
    verts::AbstractArray{T,2},
    sample_faces::AbstractArray{R,2},
    num_samples::Int,
) where {S,T,R}

    v1 = verts[:, sample_faces[1, :]]
    v2 = verts[:, sample_faces[2, :]]
    v3 = verts[:, sample_faces[3, :]]
    (w1, w2, w3) = S.(_rand_barycentric_coords(num_samples))
    samples = (w1 .* v1) + (w2 .* v2) + (w3 .* v3)
    return samples
end

function _rand_barycentric_coords(num_samples::Int)
    u = sqrt.(rand(Float32, 1, num_samples))
    v = rand(Float32, 1, num_samples)
    w1 = 1.0f0 .- u
    w2 = u .* (1.0f0 .- v)
    w3 = u .* v
    return (w1, w2, w3)
end

"""
    normalize!(m::TriMesh)

Normalize each mesh in TriMesh `m` with mean centered at origin and unit standard deviation
and overwrite the `m` with normalized TriMesh.

See also: [`normalize`](@ref)

### Examples:

```julia
julia> m = load_trimesh("teapot.obj")
julia> normalize!(m)
```
"""
function normalize!(m::TriMesh)
    verts_padded = get_verts_padded(m)
    _len = reshape(m._verts_len, 1, 1, :)
    _centroid = sum(verts_padded; dims = 2) ./ _len
    _correction = ((_centroid .^ 2) .* (m.V .- _len))
    _std =
        sqrt.(
            (sum((verts_padded .- _centroid) .^ 2; dims = 2) - _correction) ./ (_len .- 1),
        )
    verts_padded = (verts_padded .- _centroid) ./ max.(_std, EPS)
    m._verts_list = _padded_to_list(verts_padded, m._verts_len)
    return m
end

"""
    normalize(m::TriMesh)

Normalize each mesh in TriMesh `m` with mean centered at origin and unit standard deviation

See also: [`normalize!`](@ref)

### Examples:

```julia
julia> m = load_trimesh("teapot.obj")
julia> m = normalize(m)
```
"""
function normalize(m::TriMesh)
    m = deepcopy(m)
    normalize!(m)
    return m
end

"""
    scale!(m::TriMesh, factor::Number)
    scale!(m::TriMesh, factor::AbstractArray{<:Number})

Scale the TriMesh `m` by scaling factor `factor`
and overwrite `m` with scaled TriMesh. If `factor`
is array of size `(3, )`, then TriMesh will be scale
by scaling factor of respective dimension.

Scaling factor `factor` (each element in case of array)
should be strictly greater than `0.0`.

See also: [`scale`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> scale!(m, 1.0)
julia> scale!(m, [1.0, 1.0, 1.0])
```
"""
function scale!(m::TriMesh, factor::Float32)
    (factor > 0.0) || error("factor must be greater than 0.0")
    verts_packed = get_verts_packed(m)
    lmul!(factor, verts_packed)
    m._verts_packed = verts_packed
    return m
end

function scale!(m::TriMesh, factor::AbstractArray{Float32})
    (size(factor) == (3,)) ||
        error("factor must be (3, ), but instead got $(size(factor)) array")
    (factor .> 0.0) || error("factor must be greater than 0.0")
    verts_packed = get_verts_packed(m) .* reshape(factor, :, 1)
    m._verts_packed = verts_packed
    return m
end

scale!(m::TriMesh, factor) = scale!(m, Float32.(factor))

"""
    scale(m::TriMesh, factor::Number)
    scale(m::TriMesh, factor::AbstractArray{<:Number})

Scale the TriMesh `m` by scaling factor `factor`.
If `factor` is array of size `(3, )`, then TriMesh
will be scaleby scaling factor of respective dimension.

Scaling factor `factor` (each element in case of array)
should be strictly greater than `0.0`.

See also: [`scale!`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> m = scale(m, 1.0)
julia> m = scale!(m, [1.0, 1.0, 1.0])
"""
function scale(m::TriMesh, factor::Union{Float32,AbstractArray{Float32}})
    m = deepcopy(m)
    scale!(m, factor)
    return m
end

scale(m::TriMesh, factor) = scale(m, Float32.(factor))

"""
    rotate!(m::TriMesh, rotmat::AbstractArray{<:Number,2})

Rotate the TriMesh `m` by rotation matrix `rotmat`
and overwrite `m` with rotated TriMesh.

Rotation matrix `rotmat` should be of size `(3,3)`

See also: [`rotate`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> rotmat = rand(3,3)
julia> rotate!(m, rotmat)
```
"""
function rotate!(m::TriMesh, rotmat::AbstractArray{Float32,2})
    size(rotmat) == (3, 3) ||
        error("rotmat must be (3, 3) array, but instead got $(size(rotmat)) array")
    verts_packed = transpose(rotmat) * get_verts_packed(m)
    m._verts_packed = verts_packed
    return m
end

rotate!(m::TriMesh, rotmat::AbstractArray{<:Number,2}) = rotate!(m, Float32.(rotmat))

"""
    rotate(m::TriMesh, rotmat::AbstractArray{<:Number,2})

Rotate the TriMesh `m` by rotation matrix `rotmat`.

Rotation matrix `rotmat` should be of size `(3,3)`

See also: [`rotate!`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> rotmat = rand(3,3)
julia> m = rotate(m, rotmat)
```
"""
function rotate(m::TriMesh, rotmat::AbstractArray{<:Number,2})
    m = deepcopy(m)
    rotate!(m, rotmat)
    return m
end

"""
    realign!(src::TriMesh, tgt::TriMesh)
    realign!(src::TriMesh, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})

Re-Align the TriMesh `src` with the axis aligned bounding box of mesh at `index` in TriMesh `tgt`
and overwrite `src` with re-aligned TriMesh.

See also: [`realign`](@ref)

### Examples:
```julia
julia> src = load_trimesh("teapot.obj")
julia> tgt = scale(src, 2.0)
julia> realign!(src, tgt)
```
"""
function realign!(
    src::TriMesh,
    tgt_min::AbstractArray{Float32,2},
    tgt_max::AbstractArray{Float32,2},
)
    verts_padded = get_verts_padded(src)
    src_min = minimum(verts_padded, dims = 2)
    src_max = maximum(verts_padded, dims = 2)
    verts_padded =
        ((verts_padded .- src_min) ./ (src_max - src_min .+ EPS)) .* (tgt_max - tgt_min) .+
        tgt_min
    src._verts_packed = _padded_to_packed(verts_padded, src._verts_len)
    return src
end

realign!(
    src::TriMesh,
    tgt_min::AbstractArray{<:Number,2},
    tgt_max::AbstractArray{<:Number,2},
) = realign!(src, Float32.(tgt_min), Float32.(tgt_max))

realign!(src::TriMesh, tgt::AbstractArray{<:Number,2}) = realign!(src, Float32.(tgt))

function realign!(src::TriMesh, tgt::AbstractArray{Float32,2})
    tgt_min = minimum(tgt, dims = 2)
    tgt_max = maximum(tgt, dims = 2)
    realign!(src, tgt_min, tgt_max)
    return src
end

realign!(src::TriMesh, tgt::TriMesh, index::Integer = 1) =
    realign!(src, Float32.(get_verts_list(tgt)[index]))

"""
    realign(src::TriMesh, tgt::TriMesh, index::Integer=1)
    realign(src::TriMesh, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})

Re-Align the TriMesh `src` with the axis aligned bounding box of mesh at `index` in TriMesh `tgt`.

See also: [`realign`](@ref)

### Examples:
```julia
julia> src = load_trimesh("teapot.obj")
julia> tgt = scale(src, 2.0)
julia> src = realign(src, tgt)
```
"""
function realign(
    src::TriMesh,
    tgt_min::AbstractArray{<:Number,2},
    tgt_max::AbstractArray{<:Number,2},
)
    src = deepcopy(src)
    realign!(src, tgt_min, tgt_max)
    return src
end

function realign(src::TriMesh, tgt::TriMesh, index::Integer)
    src = deepcopy(src)
    realign!(src, tgt, index)
    return src
end

"""
    translate!(m::TriMesh, vector::Number)
    translate!(m::TriMesh, vector::AbstractArray{<:Number})

Translate the TriMesh `m` by translating vector `vector`
and overwrite `m` with translated TriMesh. If `vector`
is a number, then TriMesh will be translated by same number
in all dimension.

See also: [`translate`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> translate!(m, 0.0)
julia> translate!(m, [0.0, 0.0, 0.0])

"""
translate!(m::TriMesh, vector::Float32) = translate!(m, fill(vector, (3,)))

function translate!(m::TriMesh, vector::AbstractArray{Float32})
    (size(vector) == (3,)) ||
        error("vector must be (3, ), but instead got $(size(vector)) array")
    verts_packed = get_verts_packed(m)
    verts_packed = verts_packed .+ reshape(vector, :, 1)
    m._verts_packed = verts_packed
    return m
end

translate!(m::TriMesh, vector) = translate!(m, Float32.(vector))

"""
    translate(m::TriMesh, vector::Number)
    translate(m::TriMesh, vector::AbstractArray{<:Number})

Translate the TriMesh `m` by translating vector `vector`.
If `vector` is a number, then TriMesh will be translated
by same number in all dimension.

See also: [`translate!`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> m = translate(m, 0.0)
julia> m = translate(m, [0.0, 0.0, 0.0])

"""
function translate(m::TriMesh, args...)
    m = deepcopy(m)
    translate!(m, args...)
end

"""
    offset!(m::TriMesh, offset_verts::AbstractArray{<:Number,2})

Add offset to the vertices of the TriMesh `m` by offset vertices `offset_verts_packed`
and overwrite `m` with updated TriMesh.

See also: [`offset`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> offset_verts = ones(get_verts_packed(m))
julia> offset!(m, offset_verts)

"""

function offset!(m::TriMesh, offset_verts_packed::AbstractArray{Float32,2})
    verts_packed = get_verts_packed(m)
    (size(offset_verts_packed) == size(verts_packed)) ||
        error("mesh and offset_verts size mismatch")
    verts_packed += offset_verts_packed
    m._verts_packed = verts_packed
    return m
end

offset!(m::TriMesh, offset_verts_packed::AbstractArray{<:Number,2}) =
    offset!(m, Float32.(offset_verts_packed))

"""
    offset(m::TriMesh, offset_verts_packed::AbstractArray{<:Number,2})

Add offset to the vertices of the TriMesh `m` by offset vertices `offset_verts_packed`.

See also: [`offset!`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> offset_verts = ones(get_verts_packed(m))
julia> m = offset(m, offset_verts)

"""
function offset(m::TriMesh, offset_verts_packed::AbstractArray{<:Number,2})
    m = deepcopy(m)
    offset!(m, offset_verts_packed)
end
