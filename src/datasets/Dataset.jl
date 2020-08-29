module Dataset

import ..Flux3D
import SHA

export CustomDataset, ModelNet10, ModelNet40

# Utilities
include("utils.jl")

# Custom Dataset
include("custom.jl")

# ModelNet
include("modelnet/base.jl")
include("modelnet/mn10.jl")
include("modelnet/mn40.jl")

end # module
