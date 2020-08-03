module Dataset

import ..Flux3D
import ..Flux3D: PointCloud
import SHA

export CustomDataset, ModelNet10, ModelNet40

include("utils.jl")
include("custom.jl")
include("modelnet10/mn10.jl")
include("modelnet40/mn40.jl")

# (m::Module)(kwargs...) = m.dataset(kwargs...)

end # module
