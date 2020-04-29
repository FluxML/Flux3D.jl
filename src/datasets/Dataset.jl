module Dataset

import ..Flux3D
import ..Flux3D: PointCloud
import SHA

# export ModelNet10, ModelNet40, AbstractCustomDataset

include("utils.jl")

# include("autodetect.jl")

# include("modelnet10.jl")
include("modelnet10/main.jl")
export ModelNet10

include("modelnet40.jl")

end # module