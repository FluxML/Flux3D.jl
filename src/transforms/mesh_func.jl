export sample_points

import Distributions

"""
    sample_points(m::TriMesh, num_samples::Int=5000; returns_normals::Bool=false, eps::Number = 1e-6)

Uniformly samples `num_samples` points from the surface of TriMesh `m`.

`returns_normals` is optional keyword argument, to make returns normals
from respective faces of samples. `eps` is optional keyword argument for
a small number to prevent division by zero for small surface areas. 

### Examples:
julia> m = load_trimesh("teapot.obj")
julia> points = sample_points(m, 5000)
julia> points, normals = sample_points(m, 5000; returns_normals=true)

"""
function sample_points(
    m::TriMesh,
    num_samples::Int = 5000;
    returns_normals::Bool = false,
    eps::Number = EPS,
)
    face_areas, face_normals = compute_face_areas(m; compute_normals = returns_normals)
    face_areas_prob = Float64.(face_areas) ./ sum(Float64.(face_areas))
    #TODO: condition for probvec fails in Float32

    # face_areas_prob = face_areas ./ sum(face_areas)
    # face_areas_prob = face_areas ./ max(sum(face_areas), eps)
    dist = Distributions.Categorical(face_areas_prob)
    sample_faces_idx = my_rand(dist, num_samples)
    sample_faces_idx = Zygote.nograd(sample_faces_idx)
    sample_faces = m.faces[sample_faces_idx, :]
    v1 = m.vertices[sample_faces[:, 1], :]
    v2 = m.vertices[sample_faces[:, 2], :]
    v3 = m.vertices[sample_faces[:, 3], :]
    (w1, w2, w3) = _rand_barycentric_coords(num_samples)
    samples = (w1 .* v1) + (w2 .* v2) + (w3 .* v3)

    if returns_normals
        samples_normals = face_normals[sample_faces_idx, :]
        return (samples, samples_normals)
    end
    return samples
end

function _rand_barycentric_coords(num_samples::Int)
    u = sqrt.(my_rand(Float32, num_samples))
    v = my_rand(Float32, num_samples)
    w1 = 1.0f0 .- u
    w2 = u .* (1.0f0 .- v)
    w3 = u .* v
    return (w1, w2, w3)
end

"""
    normalize!(m::TriMesh)

Normalize the TriMesh `m` with mean centered at origin and unit standard deviation
and overwrite the m with normalized TriMesh.

See also: [`normalize`](@ref)

### Examples:

```julia
julia> m = load_trimesh("teapot.obj")
julia> normalize!(m)
```
"""
function normalize!(m::TriMesh)
    centroid = mean(m.vertices; dims = 1)
    m.vertices = (m.vertices .- centroid) ./ (std(m.vertices, mean = centroid, dims = 1) .+ EPS)
    return m
end

"""
    normalize(m::TriMesh)

Normalize the TriMesh `m` with mean centered at origin and unit standard deviation

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
    lmul!(factor, m.vertices)
    return m
end

function scale!(m::TriMesh, factor::AbstractArray{Float32})
    (size(factor) == (3,)) || error("factor must be (3, ), but instead got $(size(factor)) array")  
    (factor .> 0.0) || error("factor must be greater than 0.0")
    m.vertices = m.vertices .* reshape(factor, 1, 3)
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
function scale(m::TriMesh, factor::Union{Float32, AbstractArray{Float32}})
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
    size(rotmat) == (3, 3) || error("rotmat must be (3, 3) array, but instead got $(size(rotmat)) array")
    m.vertices = m.vertices * rotmat
    return m
end

rotate!(m::TriMesh, rotmat::AbstractArray{<:Number,2}) =
    rotate!(m, Float32.(rotmat))

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

Re-Align the TriMesh `src` with the axis aligned bounding box of TriMesh `tgt`
and overwrite `src` with re-aligned TriMesh.

See also: [`realign`](@ref)

### Examples:
```julia
julia> src = load_trimesh("teapot.obj")
julia> tgt = scale(src, 2.0)
julia> realign!(src, tgt)
```
"""
function realign!(src::TriMesh, tgt_min::AbstractArray{Float32,2}, tgt_max::AbstractArray{Float32,2})
    src_min = reshape(minimum(src.vertices, dims = 1), (1, :))
    src_max = reshape(maximum(src.vertices, dims = 1), (1, :))
    src.vertices = ((src.vertices .- src_min) ./ (src_max - src_min .+ EPS)) .* (tgt_max - tgt_min) .+ tgt_min
    return src
end

realign!(src::TriMesh, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2}) =
    realign!(src, Float32.(tgt_min), Float32.(tgt_max))

function realign!(src::TriMesh, tgt::TriMesh)
    tgt_min = reshape(minimum(tgt.vertices, dims = 1), (1, :))
    tgt_max = reshape(maximum(tgt.vertices, dims = 1), (1, :))
    realign!(src, tgt_min, tgt_max)
    return src
end

"""
    realign(src::TriMesh, tgt::TriMesh)
    realign(src::TriMesh, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})

Re-Align the TriMesh `src` with the axis aligned bounding box of TriMesh `tgt`.

See also: [`realign`](@ref)

### Examples:
```julia
julia> src = load_trimesh("teapot.obj")
julia> tgt = scale(src, 2.0)
julia> src = realign(src, tgt)
```
"""
function realign(src::TriMesh, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})
    src = deepcopy(src)
    realign!(src, tgt_min, tgt_max)
    return src
end

function realign(src::TriMesh, tgt::TriMesh)
    src = deepcopy(src)
    realign!(src, tgt)
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
function translate!(m::TriMesh, vector::Float32)
    m.vertices = m.vertices .+ vector
    return m
end

function translate!(m::TriMesh, vector::AbstractArray{Float32})
    (size(vector) == (3,)) || error("vector must be (3, ), but instead got $(size(vector)) array")  
    m.vertices = m.vertices .+ reshape(vector, 1, 3)
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
function translate(m::TriMesh, vector::Union{Number, AbstractArray{<:Number}})
    m = deepcopy(m)
    translate!(m, vector)
end

"""
    offset!(m::TriMesh, offset_verts::AbstractArray{<:Number,2})

Add offset to the vertices of the TriMesh `m` by offset vertices `offset_verts`
and overwrite `m` with updated TriMesh.

See also: [`offset`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> offset_verts = ones(m.vertices)
julia> offset!(m, offset_verts)

"""

function offset!(m::TriMesh, offset_verts::AbstractArray{Float32, 2})
    (size(offset_verts) == size(m.vertices)) || error("mesh and offset_verts size mismatch")
    m.vertices = m.vertices + offset_verts
    return m
end

offset!(m::TriMesh, offset_verts::AbstractArray{<:Number, 2}) = 
    offset!(m, Float32.(offset_verts))
    
"""
    offset(m::TriMesh, offset_verts::AbstractArray{<:Number,2})

Add offset to the vertices of the TriMesh `m` by offset vertices `offset_verts`.

See also: [`offset!`](@ref)

### Examples:
```julia
julia> m = load_trimesh("teapot.obj")
julia> offset_verts = ones(m.vertices)
julia> m = offset(m, offset_verts)

"""
function offset(m::TriMesh, offset_verts::AbstractArray{<:Number, 2})
    m = deepcopy(m)
    offset!(m, offset_verts)
end