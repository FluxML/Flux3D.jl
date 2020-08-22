module ModelNet40

import ..Dataset, ...Flux3D
import ..Dataset: default_root, download_and_verify, DataPoint, AbstractDataset


"""
    dataset(;mode=:pointcloud, kwargs...)

Returns ModelNet40 dataset.

Supported `mode` are {`:pointcloud`}.

### Optional Arguments:
    * `root::String=default_root`   - Root directory of dataset
    * `train::Bool=true`            - Specifies the trainset
    * `donwload::Bool=true`         - Specifies to auto-download the dataset incase specified dataset is found in root directory
    * `transform=nothing`           - Transform to be applied to data point.
    * `categories::Vector{String}`  - Specifies the categories to be used in dataset.

### Examples:

```jldoctest
julia> dset = ModelNet40.dataset(;mode=:pointcloud, train=false)
julia> typeof(dset[1].data) == PointCloud
```
"""
function dataset(;
    root::String = default_root,
    train::Bool = true,
    download=true,
    transform = nothing,
    categories = MN40_classes
)
    _path = load_dataset(root, download)
    train ? _split = "train" : _split = "test"
    datapaths = []
    for category in categories
        category in MN10_classes || error("given category: $(category) is not a valid ModelNet10 category.")
        datapath = [ (category, joinpath(_path, category, _split, filename)) for filename in readdir(joinpath(_path, category,_split)) ]
        append!(datapaths, datapath)
    end

    classes_to_idx = Dict{String,UInt8}([(categories[i],i) for i in 1:length(categories)])
    idx_to_classes = Dict{UInt8,String}([(i,categories[i]) for i in 1:length(categories)])

    _length = length(datapaths)
    return MN10(
        root,
        _path,
        train,
        _length,
        datapaths,
        transform,
        categories,
        classes_to_idx,
        idx_to_classes
    )
end

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
