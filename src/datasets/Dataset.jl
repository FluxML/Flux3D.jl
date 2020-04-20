module Dataset

import ..Flux3D
import ..Flux3D: PointCloud 

export ModelNet10, ModelNet40, AbstractCustomDataset

include("utils.jl")
include("autodetect.jl")
include("modelnet10.jl")
include("modelnet40.jl")

end # module