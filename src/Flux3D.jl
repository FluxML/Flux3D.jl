module Flux3D

using Flux, NearestNeighbors, LinearAlgebra, Makie, Statistics
using Base: tail
using Flux: @functor
using Zygote: @nograd

# representation
include("rep/utils.jl")
include("rep/pcloud.jl")

# transforms
include("transforms/utils.jl")
include("transforms/pcloud_func.jl")
include("transforms/transforms.jl")

# Dataset module
include("datasets/Dataset.jl")
using .Dataset
export CustomDataset, ModelNet10, ModelNet40

# visualization
include("visualize.jl")

# models
include("models/utils.jl")
include("models/dgcnn.jl")
include("models/pointnet.jl")

end # module