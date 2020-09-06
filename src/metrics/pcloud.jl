export chamfer_distance

"""
    chamfer_distance(A::PointCloud, B::PointCloud; w1=1.0, w2=1.0)

Computes the chamfer distance between PointCloud `A` and `B`. 

`w1` and `w2` are optional arguments for specifying the weighted mean of between two 
pcloud for computing metrics.
"""
chamfer_distance(A::PointCloud, B::PointCloud; w1::Number = 1.0, w2::Number = 1.0) =
    _chamfer_distance(A.points, B.points, Float32(w1), Float32(w2))

chamfer_distance(
    A::AbstractArray{<:Number},
    B::AbstractArray{<:Number};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(Float32.(A), Float32.(B), Float32(w1), Float32(w2))

chamfer_distance(
    A::AbstractArray{Float32},
    B::AbstractArray{Float32};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(A, B, Float32(w1), Float32(w2))

function _chamfer_distance(
    A::AbstractArray{Float32,2},
    B::AbstractArray{Float32,2},
    w1::Float32 = 1.f0,
    w2::Float32 = 1.f0,
)
    A = reshape(A, size(A)..., 1)
    B = reshape(B, size(B)..., 1)
    return _chamfer_distance(A, B, w1, w2)
end

function _chamfer_distance(
    A::AbstractArray{Float32,3},
    B::AbstractArray{Float32,3},
    w1::Float32 = 1.f0,
    w2::Float32 = 1.f0,
)
    nn_for_A, nn_for_B = @ignore _nearest_neighbors(A, B)

    dist_A_to_B = mean((A .- B[:, nn_for_A]) .^ 2) * 3.f0
    dist_B_to_A = mean((B .- A[:, nn_for_B]) .^ 2) * 3.f0

    distance = (w1 * dist_A_to_B) + (w2 * dist_B_to_A)
    return distance
end

function _nearest_neighbors(x::Array{Float32,3}, y::Array{Float32,3})
    nn_for_x = cat(
        [
            CartesianIndex.(reduce(vcat, knn(KDTree(y[:, :, i]), x[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    nn_for_y = cat(
        [
            CartesianIndex.(reduce(vcat, knn(KDTree(x[:, :, i]), y[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    return nn_for_x, nn_for_y
end

function _nearest_neighbors(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    xx = sum(x .^ 2, dims = 1)
    yy = sum(y .^ 2, dims = 1)
    zz = Flux.batched_mul(permutedims(x, (2, 1, 3)), y)
    rx = reshape(xx, size(xx, 2), 1, :)
    ry = reshape(yy, 1, size(yy, 2), :)
    P = (rx .+ ry) .- (2 .* zz)

    nn_for_x = argmin(P; dims = 2) |> Array
    nn_for_y = argmin(P; dims = 1) |> Array

    nn_for_x = reshape(map(x -> CartesianIndex(x[2], x[3]), nn_for_x), :, size(x, 3))
    nn_for_y = reshape(map(y -> CartesianIndex(y[1], y[3]), nn_for_y), :, size(y, 3))
    return nn_for_x, nn_for_y
end
