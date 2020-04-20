abstract type AbstractDataset end
abstract type AbstractDataPoint <: Flux3D.AbstractDataPoint end
abstract type AbstractCustomDataset <: AbstractDataset end

Base.size(v::AbstractCustomDataset) = (v.length,)
Base.length(v::AbstractCustomDataset) = v.length

Base.size(v::AbstractDataset) = error("define Base.size of dataset type: $(typeof(v)).")
Base.length(v::AbstractDataset) = error("define Base.length of dataset type: $(typeof(v)).")

const default_root = normpath(@__DIR__, "..", "..", "datasets")