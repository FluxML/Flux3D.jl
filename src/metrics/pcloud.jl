export chamfer_distance

function chamfer_distance(A::AbstractArray{<:Number, 2}, B::AbstractArray{<:Number, 2}; w1::Number=1.0, w2::Number=1.0)
    nn_for_A, nn_for_B = _nearest_neighbors(A, B)
    dist_to_B = sum(nn_for_A)
    dist_to_A = sum(nn_for_B)

    distance = (Float32(w1)*dist_to_B) + (Float32(w2)*dist_to_A)
    return distance
end

chamfer_distance(A::PointCloud, B::PointCloud; w1::Number=1.0, w2::Number=1.0) =
    chamfer_distance(A.points, B.points; w1=w1, w2=w2)

function _nearest_neighbors(x::AbstractArray{<:Float32,2},y::AbstractArray{<:Float32,2})
    xx = sum(x .^ 2, dims=2)
    yy = sum(y .^ 2, dims=2)
    zz = x * permutedims(y, (2,1))
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

_nearest_neighbors(x::AbstractArray{<:Number,2}, y::AbstractArray{<:Number,2}) =
    _nearest_neighbors(Float32.(x), Float32.(y))
#
# Zygote.@adjoint function findmin(xs::AbstractArray; dims = :)
#     min, i = findmin(xs, dims = dims)
#     (min, i), function (Δ)
#         Δ′ = zero(xs)
#         Δ′[i] = Δ[1]
#         return (Δ′, nothing)
#     end
# end
