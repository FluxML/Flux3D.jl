export PointCloud

struct PointCloud <: AbstractObject
    points::Array{Float32,2}
    npoints::Int
end

function PointCloud(points::Array{T,2} where{T<:Number})
    points =convert(Array{Float32,2},points)
    PointCloud(points, size(points,1))
end

PointCloud(points::Array{Float32,2}) = PointCloud(points, size(points,1))

PointCloud(;points) = PointCloud(points)