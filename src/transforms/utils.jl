abstract type AbstractTransform end
abstract type AbstractCompose <: AbstractTransform end

const EPS = Float32(1e-6)

function Base.show(io::IO, c::AbstractCompose)
    print(io, "$(typeof(c))(")
    join(io, c.transforms, ", ")
    print(io, ")")
end

Base.show(io::IO, t::AbstractTransform) = print(io, "$(typeof(t))(...)")