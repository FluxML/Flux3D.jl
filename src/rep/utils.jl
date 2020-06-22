abstract type AbstractObject end
abstract type AbstractCustomObject <: AbstractObject end
abstract type AbstractMesh <: AbstractObject end

#TODO: Remove this once new tagged version of Zygote is out
my_ignore(f) = f()
Zygote.@adjoint my_ignore(f) = my_ignore(f), _ -> nothing

#TODO: Remove this when new tagged version of Zygote is out
my_rand(args...) = rand(args...)
Zygote.@nograd my_rand

include("./groupinds.jl")
function uniqueperm(A::AbstractArray; dims::Int)
    ic = groupslices(A; dims=dims)
    return firstinds(ic)
end

function _lg_cross(A::AbstractArray, B::AbstractArray)
    if !(size(A, 2) == size(B, 2) == 3)
        throw(DimensionMismatch("cross product is only defined for AbstractArray of dimension 3 at dims 2"))
    end
    a1, a2, a3 = A[:, 1], A[:, 2], A[:, 3]
    b1, b2, b3 = B[:, 1], B[:, 2], B[:, 3]
    cat((a2 .* b3) - (a3 .* b2), (a3 .* b1) - (a1 .* b3), (a1 .* b2) - (a2 .* b1); dims = 2)
end

_normalize(A::AbstractArray; eps::Number = 1e-6, dims::Int = 2) =
    A ./ max.(sqrt.(sum(A .^ 2; dims = dims)), eps)

_norm(A::AbstractArray; dims::Int = 2) = sqrt.(sum(A .^ 2; dims = dims))

function _auxiliary_mesh(
    list::AbstractArray{<:AbstractArray{T, 2},1}
)   where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")

    _N = length(list)
    items_len = Array{Int,1}(undef, _N)
    packed_first_idx = Array{Int,1}(undef, _N)
    packed_to_list_idx = Array{Int,1}(undef, sum(size.(list,1)))

    cur_idx = 1
    for (i,x) in enumerate(list)
        _len = size(x, 1)
        items_len[i] = _len
        packed_first_idx[i] = cur_idx
        packed_to_list_idx[cur_idx:cur_idx+_len-1] .= i
        cur_idx += _len
    end
    return items_len, packed_first_idx, packed_to_list_idx
end

function _list_to_padded(
    list::AbstractArray{<:AbstractArray{T, 2}, 1},
    pad_value::Number,
    pad_size::Union{Nothing, Tuple}=nothing;
    is_similar::Bool = false
)   where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")
    if is_similar
        return Flux.stack(list, ndims(list[1])+1)
    end

    if pad_size isa Nothing
        pad_size = (maximum(size.(list,1)), maximum(size.(list, 2)))
    else
        length(pad_size) == 2 || error("pad_size should be a tuple of length 2")
    end

    padded = fill(T.(pad_value), pad_size..., length(list))
    padded = Zygote.bufferfrom(padded)

    for (i,x) in enumerate(list)
        padded[1:size(x,1), 1:size(x,2),i] = x
    end

    return copy(padded)
end

function _list_to_packed(
    list::AbstractArray{<:AbstractArray{T, 2},1}
)   where {T<:Number}

    all(ndims.(list) .== 2) || error("only 2d arrays are supported")
    packed = vcat(list...)
    # packed = reduce(vcat, list) #mutating error in zygote
    return packed
end


"""
packed (sum(Mi),3)
padded (N, max(Mi), 3)
"""
function _packed_to_padded(
    packed::AbstractArray{T, 2},
    items_len::AbstractArray{<:Number, 1},
    pad_value::Number,
)   where {T<:Number}

    ndims(packed) == 2 || error("only 2d arrays are supported")

    _N = length(items_len)
    _M = maximum(items_len)
    padded = fill(T.(pad_value), _M, size(packed,2), _N)
    padded = Zygote.bufferfrom(padded)
    cur_idx = 1
    for (i,_len) in enumerate(items_len)
        padded[1:_len,:,i] = packed[cur_idx:cur_idx+_len-1,:]
        cur_idx += _len
    end

    return copy(padded)
end

function _packed_to_list(
    packed::AbstractArray{T, 2},
    items_len::AbstractArray{<:Number, 1}
)   where {T<:Number}

    ndims(packed) == 2 || error("only 2d arrays are supported")

    _N = length(items_len)
    _M = maximum(items_len)
    list = Zygote.bufferfrom(AbstractArray{T,2}[])
    cur_idx=1
    for (i,_len) in enumerate(items_len)
        push!(list, packed[cur_idx:cur_idx+_len-1,:])
        cur_idx += _len
    end
    return copy(list)
end

function _padded_to_packed(
    padded::AbstractArray{T,3},
    items_len::Union{Nothing, AbstractArray{<:Number, 1}}=nothing,
    pad_value::Union{Nothing, Number}=nothing
)   where{T<:Number}

    ndims(padded) == 3 || error("padded should be 3 dimension array")
    (pad_value == nothing || items_len == nothing) || error("pad_value and items_len both should not be given")

    packed = reshape(permutedims(padded,(1,3,2)), :, size(padded,2))
    if pad_value !== nothing
        _mask = my_ignore(()-> reshape(all(packed[:,1] .!= repeat([pad_value],1, size(packed, 2));dims=2), :))  #(_N,3)
        # _mask = [_mask_pad[i,:] .== true for i = 1:size(_mask_pad,1)]   #(_N,)
    else
        _N = length(items_len)
        _M = size(padded, 1)
        _N == size(padded, 3) || error("items_len length should match the last dimension of padded array")

        _mask = my_ignore(()->reduce(vcat, [collect(1:_len) .+ ((i-1)*_M) for (i,_len) in enumerate(items_len)]))
    end
    return packed[_mask, :]
end

function _padded_to_list(
    padded::AbstractArray{T, 3},
    items_len::Union{Nothing, AbstractArray{<:Number, 1}}
)   where {T<:Number}

    ndims(padded) == 3 || error("padded should be 3 dimension array")

    if items_len === nothing
        return Flux.unstack(padded, 3)
    end

    _N = length(items_len)
    _N == size(padded, 3) || error("items_len length should match the last dimension of padded array")

    list = Zygote.bufferfrom(AbstractArray{T,2}[])

    for (i,_len) in enumerate(items_len)
        push!(list, padded[1:_len,:,i])
    end

    return copy(list)
end
