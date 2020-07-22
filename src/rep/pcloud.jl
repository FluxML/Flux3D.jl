export PointCloud, npoints

"""
    PointCloud

Initialize PointCloud representation.

`points` should be Array of size `(D, N, B)` where `N` is the number of points, `D` is
dimensionality of each points (i.e. `D`=2 or `D`=3) and `B` is the batch size of PointCloud.
`normals` is optional field, if given should be Array of size `(D, N, B)` where `N` and `B`
should match with the `N` and `B` of `points` and `D`=2 or `D`=3
(i.e. normals for 2D and 3D PointCloud respectively).

### Fields:

- `points`      - Points that makes up whole PointCloud.
- `normals`     - Normals of each points in PointCloud.

### Available Contructor:

- `PointCloud(points, normals=nothing)`
- `PointCloud(;points, normals=nothing)`
- `PointCloud(pcloud::PointCloud)`
"""
mutable struct PointCloud{T<:Float32} <: AbstractObject
    points::AbstractArray{T,3}
    normals::Union{AbstractArray{T,3}, Nothing}
end

function PointCloud(points::AbstractArray{Float32,2}, normals::Union{AbstractArray{Float32,2}, Nothing}=nothing)
    points = reshape(points, size(points)...,1)

    if !(normals isa Nothing)
        size(points,2) == size(normals,2) || error("number of points and normals must match in PointCloud.")
        normals = reshape(normals, size(normals)...,1);
    end

    return PointCloud(points, normals)
end

function PointCloud(points::AbstractArray, normals::Union{AbstractArray, Nothing}=nothing)
    points = Float32.(points)
    if normals !== nothing
        normals = Float32.(normals)
    end
    return PointCloud(points, normals)
end

PointCloud(;points, normals=nothing)= PointCloud(points, normals)

PointCloud(pcloud::PointCloud) = PointCloud(pcloud.points, pcloud.normals)

@functor PointCloud

# deepcopy generate error when using on PointCloud with parent field of points not Nothing
Base.deepcopy_internal(x::PointCloud, dict::IdDict) =
    PointCloud(copy(x.points), (x.normals===nothing ? nothing : copy(x.normals)))

Base.getindex(p::PointCloud, index::Number) = p.points[:,:,index]

function Base.show(io::IO, p::PointCloud)
    if p.normals isa Nothing
        print(io, "points: $(size(p.points)), normals: nothing")
    else
        print(io, "points: $(size(p.points)), normals: $(size(p.normals))")
    end
end

Base.show(io::IO, ::MIME"text/plain", p::PointCloud) =
    print(io, "PointCloud object:\n   ", p)

"""
    npoints(p::PointCloud)

Returns the size of PointCloud.
"""
npoints(p::PointCloud) = size(p.points,2)
