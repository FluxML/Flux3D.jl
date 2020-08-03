module ModelNet40

import ..Dataset, ...Flux3D
import ..Dataset: download_and_verify, AbstractDataset, AbstractDataPoint
import ...Flux3D: PointCloud

include("mn40_pcloud.jl")

"""
    dataset(;mode=:pointcloud, kwargs...)

Returns ModelNet40 dataset.

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
julia> dset = ModelNet40.dataset(;mode=:pointcloud, train=false)
julia> typeof(dset[1].data) == PointCloud
```
"""
function dataset(; mode = :pointcloud, kwargs...)
    if mode == :pointcloud
        return ModelNet40PCloud(; kwargs...)
    else
        error("selected mode: $(mode) is not supported (Currently supported mode are {:pointcloud}).")
    end
end

struct MN40DataPoint <: AbstractDataPoint
    idx::Int
    data::Union{PointCloud}
    ground_truth::UInt8
end

const default_root = normpath(@__DIR__, "../../../datasets/modelnet")

const MN40_classes_to_idx = Dict{String,UInt8}([
    ("airplane", 1),
    ("bathtub", 2),
    ("bed", 3),
    ("bench", 4),
    ("bookshelf", 5),
    ("bottle", 6),
    ("bowl", 7),
    ("car", 8),
    ("chair", 9),
    ("cone", 10),
    ("cup", 11),
    ("curtain", 12),
    ("desk", 13),
    ("door", 14),
    ("dresser", 15),
    ("flower_pot", 16),
    ("glass_box", 17),
    ("guitar", 18),
    ("keyboard", 19),
    ("lamp", 20),
    ("laptop", 21),
    ("mantel", 22),
    ("monitor", 23),
    ("night_stand", 24),
    ("person", 25),
    ("piano", 26),
    ("plant", 27),
    ("radio", 28),
    ("range_hood", 29),
    ("sink", 30),
    ("sofa", 31),
    ("stairs", 32),
    ("stool", 33),
    ("table", 34),
    ("tent", 35),
    ("toilet", 36),
    ("tv_stand", 37),
    ("vase", 38),
    ("wardrobe", 39),
    ("xbox", 40),
])

const MN40_idx_to_classes = Dict{UInt8,String}([
    (1, "airplane"),
    (2, "bathtub"),
    (3, "bed"),
    (4, "bench"),
    (5, "bookshelf"),
    (6, "bottle"),
    (7, "bowl"),
    (8, "car"),
    (9, "chair"),
    (10, "cone"),
    (11, "cup"),
    (12, "curtain"),
    (13, "desk"),
    (14, "door"),
    (15, "dresser"),
    (16, "flower_pot"),
    (17, "glass_box"),
    (18, "guitar"),
    (19, "keyboard"),
    (20, "lamp"),
    (21, "laptop"),
    (22, "mantel"),
    (23, "monitor"),
    (24, "night_stand"),
    (25, "person"),
    (26, "piano"),
    (27, "plant"),
    (28, "radio"),
    (29, "range_hood"),
    (30, "sink"),
    (31, "sofa"),
    (32, "stairs"),
    (33, "stool"),
    (34, "table"),
    (35, "tent"),
    (36, "toilet"),
    (37, "tv_stand"),
    (38, "vase"),
    (39, "wardrobe"),
    (40, "xbox"),
])

const MN40_classes = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

Base.show(io::IO, p::MN40DataPoint) = print(
    io,
    "idx: $(p.idx), data: $(typeof(p.data)), ground_truth: $(p.ground_truth) ($(MN40_idx_to_classes[p.ground_truth]))",
)

Base.show(io::IO, ::MIME"text/plain", p::MN40DataPoint) =
    print(io, "ModelNet40 DataPoint:\n   ", p)

end # module
