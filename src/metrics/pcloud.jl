export chamfer_distance

#TODO: change PointCloud to 3xV format
# chamfer_distance(A::PointCloud, B::PointCloud; w1::Number=1.0, w2::Number=1.0) =
#     _chamfer_distance(A.points, B.points, Float32(w1), Float32(w2))

chamfer_distance(A::AbstractArray{<:Number}, B::AbstractArray{<:Number}; w1::Number=1.0, w2::Number=1.0) =
    _chamfer_distance(Float32.(A), Float32.(B), Float32(w1), Float32(w2))

function _chamfer_distance(A::AbstractArray{Float32, 2}, B::AbstractArray{Float32, 2}, w1::Float32=1.0, w2::Float32=1.0)
    A = reshape(A, size(A)..., 1)
    B = reshape(B, size(B)..., 1)
    return _chamfer_distance(A,B,w1,w2)
end

function _chamfer_distance(A::AbstractArray{Float32, 3}, B::AbstractArray{Float32, 3}, w1::Float32=1.0, w2::Float32=1.0)
    nn_for_A, nn_for_B = _nearest_neighbors(A, B)
    # pcloud Batch reduction
    nn_for_A = mean(nn_for_A; dims=2)
    nn_for_B = mean(nn_for_B; dims=2)
    # pcloud mean reduction
    dist_to_B = mean(nn_for_A)
    dist_to_A = mean(nn_for_B)

    distance = (Float32(w1)*dist_to_B) + (Float32(w2)*dist_to_A)
    return distance
end

function _nearest_neighbors(x::AbstractArray{Float32,3},y::AbstractArray{Float32,3})
    xx = sum(x .^ 2, dims=1)
    yy = sum(y .^ 2, dims=1)
    zz = Flux.batched_mul(permutedims(x, (2,1,3)), y)
    rx = reshape(xx, size(xx, 2), 1, :)
    ry = reshape(yy, 1, size(yy, 2), :)
    P = (rx .+ ry) .- (2 .* zz)
    nn_for_x = minimum(P; dims=2)
    nn_for_y = minimum(P; dims=1)
    return nn_for_x, nn_for_y
end
