module ModelNet10

import ..Dataset, ...Flux3D
import ..Dataset: default_root, download_and_verify, DataPoint, AbstractDataset

"""
    dataset(;kwargs...)

Returns ModelNet10 dataset.

### Optional Arguments:
    * `root::String=default_root`   - Root directory of dataset
    * `train::Bool=true`            - Specifies the trainset
    * `donwload::Bool=true`         - Specifies to auto-download the dataset incase specified dataset is found in root directory
    * `transform=nothing`           - Transform to be applied to data point.
    * `categories::Vector{String}`  - Specifies the categories to be used in dataset.

### Examples:

```jldoctest
julia> dset = ModelNet10.dataset(, train=false)
julia> typeof(dset[1].data) == TriMesh
```
"""
function dataset(;
    root::String = default_root,
    train::Bool=true,
    download=true,
    transform=nothing,
    categories=MN10_classes
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

"""
    MN10

ModelNet10 dataset.

### Fields:

* `root::String`                                        - Root directory of dataset
* `path::String`                                        - Directory of dataset
* `train::Bool`                                         - Specifies the trainset
* `length::Int`                                         - Length of dataset
* `datapaths::Array`                                    - Array containing the shape and path for each datapoint
* `transform::Union{Flux3D.AbstractTransform, Nothing}` - Transform to be applied to data point
* `categories::Vector{String}`                          - Categories to be used in dataset
* `classes_to_idx::Dict{String, UInt8}`                 - Dict mapping from shape name to class_idx
* `idx_to_classes::Dict{UInt8, String}`                 - Dict mapping from class_idx to shape name
"""
struct ModelNet <: AbstractDataset
    root::String
    path::String
    train::Bool
    length::Int
    datapaths::Array
    transform::Union{Flux3D.AbstractTransform,Nothing}
    categories::Vector{String}
    classes_to_idx::Dict{String,UInt8}
    idx_to_classes::Dict{UInt8,String}
end

function load_dataset(root::String, donwload::Bool)
    ispath(root) || mkpath(root)
    local_dir = joinpath(root, "ModelNet10")
    local_path = joinpath(root, "ModelNet10.zip")
    hash = "d64e9c5cfc479bac3260b164ae3c75ba83e94a1d216fbcd3f59ce2a9686d3762"

    if (!isdir(local_dir))
        if (!isfile(local_path))
            if (dowload)
                download_and_verify(
                    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
                    local_path,
                    hash,
                )
            else
                error("dataset not found and auto-download option is set false.")
            end
        end
        run(`unzip -q $local_path -d $root`)
    end

    return local_dir
end

function Base.getindex(v::MN10, idx::Int)
    data = Flux3D.load_trimesh(v.datapaths[idx][2])
    cls = v.classes_to_idx[v.datapaths[idx][1]]
    category = v.datapaths[idx][1]
    if v.transform != nothing
        data = v.transform(data)
    end
    return DataPoint(idx, data, cls, category)
end

Base.size(v::MN10) = (v.length,)
Base.length(v::MN10) = v.length

function Base.show(io::IO, dset::MN10)
    print(
        io,
        "ModelNet10 Dataset:",
        "\n    root: $(dset.root)",
        "\n    train: $(dset.train)",
        "\n    length: $(dset.length)",
        "\n    transform: $(dset.transform)",
        "\n    categories: $(length(dset.categories))",
    )
end

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

end # module
