module Dataset

import ..Flux3D
import ..Flux3D: PointCloud
import SHA

# export ModelNet10, ModelNet40, AbstractCustomDataset

include("utils.jl")

include("modelnet10/main.jl")
export ModelNet10

include("modelnet40/main.jl")
export ModelNet40

end # module