module Flux3D

using Flux, NearestNeighbors, LinearAlgebra, Statistics, CuArrays, Requires
using Base: tail
using Zygote: @nograd
import Flux: @functor, functor, gpu, cpu

export gpu, cpu

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

# transforms
include("transforms/utils.jl")
include("transforms/pcloud_func.jl")
include("transforms/transforms.jl")

# Dataset module
include("datasets/Dataset.jl")
using .Dataset
export CustomDataset, ModelNet10, ModelNet40

# visualization
function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("visualize.jl")
end

# models
include("models/utils.jl")
include("models/dgcnn.jl")
include("models/pointnet.jl")

end # module
