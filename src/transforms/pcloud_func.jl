"""
    normalize!(pcloud::PointCloud)

Normalize the PointCloud `pcloud` with mean centered at origin and unit standard deviation
and overwrite the pcloud with normalized PointCloud.

See also: [`normalize`](@ref)

### Examples:

```julia
julia> p = PointCloud(rand(1024,3))
julia> normalize!(p)
```
"""
function normalize!(pcloud::PointCloud)
    centroid = mean(pcloud.points, dims = 2)
    pcloud.points = (pcloud.points .- centroid) ./ (std(pcloud.points, mean = centroid, dims = 2) .+ EPS)
    return pcloud
end

"""
    normalize(pcloud::PointCloud)

Normalize the PointCloud `pcloud` with mean centered at origin and unit standard deviation

See also: [`normalize!`](@ref)

### Examples:

```julia
julia> p = PointCloud(rand(1024,3))
julia> p = normalize(p)
```
"""
function normalize(pcloud::PointCloud)
    p = deepcopy(pcloud)
    normalize!(p)
    return p
end

"""
    scale!(pcloud::PointCloud, factor::Number)

Scale the PointCloud `pcloud` by scaling factor `factor`
and overwrite `pcloud` with scaled PointCloud.

Scaling factor `factor` should be strictly greater than `0.0`.

See also: [`scale`](@ref)

### Examples:
```julia
julia> p = PointCloud(rand(1024,3))
julia> scale!(p, 1.0)
```
"""
function scale!(pcloud::PointCloud, factor::Float32)
    (factor > 0.0) || error("factor must be greater than 0.0")
    lmul!(factor, pcloud.points)
    return pcloud
end

scale!(pcloud::PointCloud, factor::Number) = scale!(pcloud, Float32(factor))

"""
    scale(pcloud::PointCloud, factor::Number)

Scale the PointCloud `pcloud` by scaling factor `factor`.

Scaling factor `factor` should be strictly greater than `0.0`.

See also: [`scale!`](@ref)

### Examples:
```julia
julia> p = PointCloud(rand(1024,3))
julia> p = scale(1.0)
```
"""
function scale(pcloud::PointCloud, factor::Float32)
    p = deepcopy(pcloud)
    scale!(p, factor)
    return p
end

scale(pcloud::PointCloud, factor::Number) = scale(pcloud, Float32(factor))

"""
    rotate!(pcloud::PointCloud, rotmat::AbstractArray{Number,2})

Rotate the PointCloud `pcloud` by rotation matrix `rotmat`
and overwrite `pcloud` with rotated PointCloud.

Rotation matrix `rotmat` should be of size `(3,3)`

See also: [`rotate`](@ref)

### Examples:
```julia
julia> p = PointCloud(rand(1024,3))
julia> rotmat = rand(3,3)
julia> rotate!(p, rotmat)
```
"""
function rotate!(pcloud::PointCloud, rotmat::AbstractArray{Float32,2})
    size(rotmat) == (3, 3) || error("rotmat must be (3, 3) array, but instead got $(size(rotmat)) array")
    size(pcloud.points, 1) == 3 || error("dimension of points in PointCloud must be 3")
    points_packed = transpose(rotmat) * reshape(pcloud.points, 3, :)
    pcloud.points = reshape(points_packed, size(pcloud.points)...)
    return pcloud
end

rotate!(pcloud::PointCloud, rotmat::AbstractArray{<:Number,2}) =
    rotate!(pcloud, Float32.(rotmat))

"""
    rotate(pcloud::PointCloud, rotmat::Array{Number,2})

Rotate the PointCloud `pcloud` by rotation matrix `rotmat`.

Rotation matrix `rotmat` should be of size `(3,3)`

See also: [`rotate!`](@ref)

### Examples:
```julia
julia> p = PointCloud(rand(1024,3))
julia> rotmat = rand(3,3)
julia> p = rotate(p, rotmat)
```
"""
function rotate(pcloud::PointCloud, rotmat::AbstractArray{Float32,2})
    p = deepcopy(pcloud)
    rotate!(p, rotmat)
    return p
end

rotate(pcloud::PointCloud, rotmat::AbstractArray{Number,2}) = rotate(pcloud, Float32.(rotmat))

"""
    realign!(src::PointCloud, tgt::PointCloud)
    realign!(src::PointCloud, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})

Re-Align the PointCloud `src` with the axis aligned bounding box of PointCloud `tgt`
and overwrite `pcloud` with re-aligned PointCloud.

PointCloud `src` and `tgt` should be of same dimension.

See also: [`realign`](@ref)

### Examples:
```julia
julia> src = PointCloud(rand(1024,3))
julia> tgt = PointCloud(rand(1024,3))
julia> realign!(src, tgt)
```
"""
function realign!(src::PointCloud, tgt_min::AbstractArray{Float32,2}, tgt_max::AbstractArray{Float32,2})
    size(src.points, 1) == size(tgt_max, 1) || error("source and target pointcloud dimension mismatch")
    src_min = minimum(src.points, dims = 2)
    src_max = maximum(src.points, dims = 2)
    src.points = ((src.points .- src_min) ./ (src_max - src_min .+ EPS)) .* (tgt_max - tgt_min) .+ tgt_min
    return src
end

realign!(src::PointCloud, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2}) =
    realign!(src, Float32.(tgt_min), Float32.(tgt_max))

function realign!(src::PointCloud, tgt::PointCloud, index::Number=1)
    points = tgt[index]
    tgt_min = minimum(points, dims = 2)
    tgt_max = maximum(points, dims = 2)
    realign!(src, tgt_min, tgt_max)
    return src
end

"""
    realign(src::PointCloud, tgt::PointCloud)
    realign(src::PointCloud, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})

Re-Align the PointCloud `src` with the axis aligned bounding box of PointCloud `tgt`.

PointCloud `src` and `tgt` should be of same dimension.

See also: [`realign!`](@ref)

### Examples:
```julia
julia> src = PointCloud(rand(1024,3))
julia> tgt = PointCloud(rand(1024,3))
julia> src = realign!(src, tgt)
```
"""
function realign(src::PointCloud, tgt_min::AbstractArray{<:Number,2}, tgt_max::AbstractArray{<:Number,2})
    p = deepcopy(src)
    realign!(p, tgt_min, tgt_max)
    return p
end

function realign(src::PointCloud, tgt::PointCloud, index::Number=1)
    p = deepcopy(src)
    realign!(p, tgt, index)
    return p
end
