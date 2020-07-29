module Flux3D

using Flux, Zygote, NearestNeighbors, LinearAlgebra, Statistics, CUDA, FileIO, SparseArrays, Requires
using Base: tail
import Zygote: @nograd, @ignore
import GeometryBasics
import Flux: @functor, functor, gpu, cpu, Chain

export gpu, cpu, Chain, visualize

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
@init @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" include("visualize.jl")


# models
include("models/utils.jl")
include("models/dgcnn.jl")
include("models/pointnet.jl")

# borowed from Flux.jl
const use_cuda = Ref(false)
@init begin
  use_cuda[] = CUDA.functional() # Can be overridden after load with `Flux.use_cuda[] = false`
  if CUDA.functional()
    if !CUDA.has_cudnn()
      @warn "CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
