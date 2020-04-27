module Flux3D

using Flux, NearestNeighbors, LinearAlgebra, Makie, Statistics
using Base: tail
using Flux: @functor
using Zygote: @nograd

# utilities
include("utils.jl")

# representation
include("rep/utils.jl")
include("rep/pcloud.jl")

# transforms
include("transforms/utils.jl")
include("transforms/pcloud.jl")
include("transforms/transforms.jl")

# visualization
include("visualize.jl")

# Dataset module
include("datasets/Dataset.jl")
using .Dataset
export ModelNet10, ModelNet40, AbstractCustomDataset, Dataset

# models
include("models/utils.jl")
include("models/dgcnn.jl")
include("models/pointnet.jl")

end # module