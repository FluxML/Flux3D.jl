export chamfer_distance

chamfer_distance(A::PointCloud, B::PointCloud; w1::Number=1.0, w2::Number=1.0) =
    _chamfer_distance(A.points, B.points, Float32(w1), Float32(w2))

chamfer_distance(A::AbstractArray{<:Number}, B::AbstractArray{<:Number}; w1::Number=1.0, w2::Number=1.0) =
    _chamfer_distance(Float32.(A), Float32.(B), Float32(w1), Float32(w2))

function _chamfer_distance(A::AbstractArray{Float32, 2}, B::AbstractArray{Float32, 2}, w1::Float32=1.0, w2::Float32=1.0)
    A = reshape(A, size(A)..., 1)
    B = reshape(B, size(B)..., 1)
    return _chamfer_distance(A,B,w1,w2)
end

function _chamfer_distance(A::AbstractArray{Float32, 3}, B::AbstractArray{Float32, 3}, w1::Float32=1.0, w2::Float32=1.0)
    nn_for_A, nn_for_B = _nearest_neighbors(A, B)
    nn_for_A = mean(nn_for_A; dims=2)
    nn_for_B = mean(nn_for_B; dims=2)
    dist_to_B = mean(nn_for_A)
    dist_to_A = mean(nn_for_B)

    distance = (Float32(w1)*dist_to_B) + (Float32(w2)*dist_to_A)
    return distance
end

function _nearest_neighbors(x::AbstractArray{Float32,3},y::AbstractArray{Float32,3})
    xx = sum(x .^ 2, dims=2)
    yy = sum(y .^ 2, dims=2)
    zz = Flux.batched_mul(x ,permutedims(y, (2,1,3)))
    rx = reshape(xx, :, 1)
    ry = reshape(yy, 1, :)
    P = (rx .+ ry) .- (2 .* zz)
    # nn_for_x, nn_idx_for_x = findmin(P; dims=2)
    # nn_for_y, nn_idx_for_y = findmin(P; dims=1)
    nn_for_x = minimum(P; dims=2)
    nn_for_y = minimum(P; dims=1)
    # nn_idx_for_x = map(i->i[2], dropdims(nn_idx_for_x; dims=2))
    # nn_idx_for_y = map(i->i[1], dropdims(nn_idx_for_y; dims=1))
    return nn_for_x, nn_for_y#, nn_idx_for_x, nn_idx_for_y
end

# function chamfer_distance(A::AbstractArray{<:Number, 2}, B::AbstractArray{<:Number, 2}; w1::Number=1.0, w2::Number=1.0)
#     nn_for_A, nn_for_B = _nearest_neighbors(A, B)
#     dist_to_B = mean(nn_for_A)
#     dist_to_A = mean(nn_for_B)
#
#     distance = (Float32(w1)*dist_to_B) + (Float32(w2)*dist_to_A)
#     return distance
# end
#
#
#
# function _nearest_neighbors(x::AbstractArray{<:Float32,2},y::AbstractArray{<:Float32,2})
#     xx = sum(x .^ 2, dims=2)
#     yy = sum(y .^ 2, dims=2)
#     zz = x * permutedims(y, (2,1))
#     rx = reshape(xx, :, 1)
#     ry = reshape(yy, 1, :)
#     P = (rx .+ ry) .- (2 .* zz)
#
#     # nn_for_x, nn_idx_for_x = findmin(P; dims=2)
#     # nn_for_y, nn_idx_for_y = findmin(P; dims=1)
#     nn_for_x = minimum(P; dims=2)
#     nn_for_y = minimum(P; dims=1)
#
#     # nn_idx_for_x = map(i->i[2], dropdims(nn_idx_for_x; dims=2))
#     # nn_idx_for_y = map(i->i[1], dropdims(nn_idx_for_y; dims=1))
#
#     return nn_for_x, nn_for_y#, nn_idx_for_x, nn_idx_for_y
# end

#
# Zygote.@adjoint function findmin(xs::AbstractArray; dims = :)
#     min, i = findmin(xs, dims = dims)
#     (min, i), function (Δ)
#         Δ′ = zero(xs)
#         Δ′[i] = Δ[1]
#         return (Δ′, nothing)
#     end
# end
