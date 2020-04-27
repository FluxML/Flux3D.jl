export PointCloud, npoints

"""
    PointCloud

Initialize PointCloud representation.

`points` should be Array of size `(N,D)` where `N` is the number of points and
`D` is dimensionality of each points (i.e. `D`=2 or `D`=3). `normals` is optional
field, if given should be Array of size `(N,D)` where `N` should match with the
`N` of `points` and `D`=2 or `D`=3 (i.e. normals for 2D and 3D PointCloud respectively).

### Feilds:

- `points`      - Points that makes up whole PointCloud. 
- `normals`     - Normals of each points in PointCloud.

### Available Contructor:

- `PointCloud(points::Array{T,2}, normals::Union(Array{R,2}, nothing)=nothing) where {T<:Number,R<:Number}`
- `PointCloud(;points, normals=nothing)`
- `PointCloud(pcloud::PointCloud)`
"""
mutable struct PointCloud <: AbstractObject
    points::Array{Float32,2}
    normals::Union{Array{Float32,2}, Nothing}
end

function PointCloud(points::Array{T,2}, normals::Union{Array{R,2}, Nothing}=nothing) where {T<:Number,R<:Number}
    points = convert(Array{Float32,2},points)

    if normals != nothing
        size(points,1) == size(normals,1) || error("number of points and normals must match in PointCloud.")
        normals = convert(Array{Float32,2},normals);
    end

    PointCloud(points, normals)
end

PointCloud(;points, normals=nothing)= PointCloud(points, normals)

PointCloud(pcloud::PointCloud) = PointCloud(pcloud.points, pcloud.normals)

Base.getindex(p::PointCloud, I...) = p.points[I...]

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
npoints(p::PointCloud) = size(p.points,1)