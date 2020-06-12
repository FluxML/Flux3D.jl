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
