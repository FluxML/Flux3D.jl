module Flux3D

using Flux, Zygote, NearestNeighbors, LinearAlgebra, Statistics, CuArrays, FileIO, SparseArrays
using Base: tail
import Zygote: @nograd, @ignore
import GeometryBasics
import Flux: @functor, functor, gpu, cpu, Chain

export gpu, cpu, Chain

# borowed from Flux.jl
const use_cuda = Ref(false)
function __init__()
  use_cuda[] = CuArrays.functional() # Can be overridden after load with `Flux.use_cuda[] = false`
  if CuArrays.functional()
    if !CuArrays.has_cudnn()
      @warn "CuArrays.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
  end
end

# representation
include("rep/utils.jl")
include("rep/pcloud.jl")
include("rep/mesh.jl")

# transforms
include("transforms/utils.jl")
include("transforms/pcloud_func.jl")
include("transforms/mesh_func.jl")
include("transforms/transforms.jl")

# metrics
include("metrics/pcloud.jl")
include("metrics/mesh.jl")

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
