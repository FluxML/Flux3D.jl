module ModelNet10

import ..Dataset, ...Flux3D
import ..Dataset: download_and_verify, AbstractDataset, AbstractDataPoint
import ...Flux3D: PointCloud

include("mn10_pcloud.jl")

"""
    dataset(;mode=:pointcloud, kwargs...)

Returns ModelNet10 dataset.

Supported `mode` are {`:pointcloud`}.

### Optional Arguments:

* For `mode=:pointcloud`

    * `root::String=default_root`   - Root directory of dataset
    * `train::Bool=true`            - Specifies the trainset 
    * `npoints::Int=1024`           - Number of points and normals in each PointCloud.
    * `transform=nothing`           - Transform to be applied to data point.
    * `sampling=nothing`            - 'to be implement'

### Examples:

```jldoctest
julia> dset = ModelNet10.dataset(;mode=:pointcloud, train=false)
julia> typeof(dset[1].data) == PointCloud
```
"""
function dataset(; mode = :pointcloud, kwargs...)
    if mode == :pointcloud
        return ModelNet10PCloud(; kwargs...)
    else
        error("selected mode: $(mode) is not supported (Currently supported mode are {:pointcloud}).")
    end
end

struct MN10DataPoint <: AbstractDataPoint
    idx::Int
    data::Union{PointCloud}
    ground_truth::UInt8
end

const default_root = normpath(@__DIR__, "../../../datasets/modelnet")

const MN10_classes_to_idx = Dict{String,UInt8}([
    ("bathtub", 1),
    ("bed", 2),
    ("chair", 3),
    ("desk", 4),
    ("dresser", 5),
    ("monitor", 6),
    ("night_stand", 7),
    ("sofa", 8),
    ("table", 9),
    ("toilet", 10),
])

const MN10_idx_to_classes = Dict{UInt8,String}([
    (1, "bathtub"),
    (2, "bed"),
    (3, "chair"),
    (4, "desk"),
    (5, "dresser"),
    (6, "monitor"),
    (7, "night_stand"),
    (8, "sofa"),
    (9, "table"),
    (10, "toilet"),
])

const MN10_classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

Base.show(io::IO, p::MN10DataPoint) = print(
    io,
    "idx: $(p.idx), data: $(typeof(p.data)), ground_truth: $(p.ground_truth) ($(MN10_idx_to_classes[p.ground_truth]))",
)

Base.show(io::IO, ::MIME"text/plain", p::MN10DataPoint) =
    print(io, "ModelNet10 DataPoint:\n   ", p)

end # module
