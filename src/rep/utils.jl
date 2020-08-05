abstract type AbstractObject end
abstract type AbstractCustomObject <: AbstractObject end

#TODO: remove when CUDA.jl v1.2.0+ is out.
stdev(a; mean, dims) = sqrt.(sum((a .- mean) .^ 2, dims = dims) / size(a, 2))

function _lg_cross(A::AbstractArray, B::AbstractArray)
    if !(size(A, 1) == size(B, 1) == 3)
        throw(DimensionMismatch("cross product is only defined for AbstractArray of dimension 3 at dims 1"))
    end
    a1, a2, a3 = A[1, :], A[2, :], A[3, :]
    b1, b2, b3 = B[1, :], B[2, :], B[3, :]
    return vcat(reshape.(
        [(a2 .* b3) - (a3 .* b2), (a3 .* b1) - (a1 .* b3), (a1 .* b2) - (a2 .* b1)],
        1,
        :,
    )...)
end

function _normalize(A::AbstractArray{T,2}; eps::Number = 1e-6, dims::Int = 2) where {T}
    eps = T.(eps)
    norm = max.(_norm(A; dims = dims), eps)
    return (A ./ norm)
end

_norm(A::AbstractArray; dims::Int = 2) = sqrt.(sum(A .^ 2; dims = dims))

function _auxiliary_mesh(list::Vector{<:AbstractArray{T,2}}) where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")

    _N = length(list)
    items_len = Array{Int,1}(undef, _N)
    packed_first_idx = Array{Int,1}(undef, _N)
    packed_to_list_idx = Array{Int,1}(undef, sum(size.(list, 2)))

    cur_idx = 1
    for (i, x) in enumerate(list)
        _len = size(x, 2)
        items_len[i] = _len
        packed_first_idx[i] = cur_idx
        packed_to_list_idx[cur_idx:cur_idx+_len-1] .= i
        cur_idx += _len
    end
    return items_len, packed_first_idx, packed_to_list_idx
end

function _list_to_padded(
    list::Vector{<:AbstractArray{T,2}},
    pad_value::Number,
    pad_size::Union{Nothing,Tuple} = nothing,
) where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")

    if pad_size isa Nothing
        pad_size = (maximum(size.(list, 1)), maximum(size.(list, 2)))
    else
        length(pad_size) == 2 || error("pad_size should be a tuple of length 2")
    end

    padded = @ignore similar(list[1], pad_size..., length(list))
    padded = _list_to_padded!(padded, list, pad_value, pad_size)
    return padded
end

function _list_to_padded!(
    padded::AbstractArray{T,3},
    list::Vector{<:AbstractArray{T,2}},
    pad_value::Number,
    pad_size::Union{Nothing,Tuple} = nothing,
) where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")

    if pad_size isa Nothing
        pad_size = (maximum(size.(list, 1)), maximum(size.(list, 2)))
    else
        length(pad_size) == 2 || error("pad_size should be a tuple of length 2")
    end

    padded = @ignore fill!(padded, T.(pad_value))
    padded = Zygote.bufferfrom(padded)

    for (i, x) in enumerate(list)
        padded[1:size(x, 1), 1:size(x, 2), i] = x
    end

    padded = copy(padded)
end

function _list_to_packed(list::Vector{<:AbstractArray{T,2}}) where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")
    packed = hcat(list...)
    # packed = reduce(vcat, list) #mutating error in zygote
    return packed
end

# function _list_to_packed!(
#     packed::AbstractArray{T,2},
#     list::Vector{<:AbstractArray{T, 2}}
# )   where {T<:Number}
#
#     all(ndims.(list) .== 2) || error("only 2d arrays are supported")
#     packed = Zygote.bufferfrom(packed)
#     cur_idx=1
#     for x in list
#         packed[:, cur_idx:cur_idx+size(x,2)-1] .= x
#         cur_idx += size(x,2)
#     end
#     packed = copy(packed)
#     return packed
# end

function _packed_to_padded(
    packed::AbstractArray{T,2},
    items_len::AbstractArray{<:Number,1},
    pad_value::Number,
) where {T<:Number}

    ndims(packed) == 2 || error("only 2d arrays are supported")

    _N = length(items_len)
    _M = maximum(items_len)
    padded = @ignore fill!(similar(packed, size(packed, 1), _M, _N), T.(pad_value))
    padded = Zygote.bufferfrom(padded)
    cur_idx = 1
    for (i, _len) in enumerate(items_len)
        padded[:, 1:_len, i] = packed[:, cur_idx:cur_idx+_len-1]
        cur_idx += _len
    end

    padded = copy(padded)
    return padded
end

function _packed_to_list(
    packed::AbstractArray{T,2},
    items_len::AbstractArray{<:Number,1},
) where {T<:Number}

    ndims(packed) == 2 || error("only 2d arrays are supported")

    _N = length(items_len)
    _M = maximum(items_len)
    list = @ignore typeof(similar(packed))[]
    list = Zygote.bufferfrom(list)
    cur_idx = 1
    for (i, _len) in enumerate(items_len)
        push!(list, packed[:, cur_idx:cur_idx+_len-1])
        cur_idx += _len
    end
    return copy(list)
end

function _padded_to_packed(
    padded::AbstractArray{T,3},
    items_len::Union{Nothing,AbstractArray{<:Number,1}} = nothing,
    pad_value::Union{Nothing,Number} = nothing,
) where {T<:Number}

    ndims(padded) == 3 || error("padded should be 3 dimension array")
    (pad_value == nothing || items_len == nothing) ||
        error("pad_value and items_len both should not be given")

    packed = reshape(padded, size(padded, 1), :)
    _N = length(items_len)
    _M = size(padded, 2)
    _N == size(padded, 3) ||
        error("items_len length should match the last dimension of padded array")
    _mask = @ignore reduce(
        vcat,
        [collect(1:_len) .+ ((i - 1) * _M) for (i, _len) in enumerate(items_len)],
    )

    return packed[:, _mask]
end

function _padded_to_list(
    padded::AbstractArray{T,3},
    items_len::Union{Nothing,AbstractArray{<:Number,1}},
) where {T<:Number}

    ndims(padded) == 3 || error("padded should be 3 dimension array")

    if items_len === nothing
        return Flux.unstack(padded, 3)
    end

    _N = length(items_len)
    _N == size(padded, 3) ||
        error("items_len length should match the last dimension of padded array")

    list = @ignore typeof(similar(padded, 1, 1))[]
    list = Zygote.bufferfrom(list)

    for (i, _len) in enumerate(items_len)
        push!(list, padded[:, 1:_len, i])
    end

    return copy(list)
end
