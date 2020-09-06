"""
    ModelNet

Base ModelNet for MN10/40.

### Fields:

* `root::String`                           - Root directory of dataset
* `path::String`                           - Directory of dataset
* `train::Bool`                            - Specifies the trainset
* `length::Int`                            - Length of dataset
* `datapaths::Array`                       - Array containing the shape and path for each datapoint
* `transform`                              - Transform to be applied to data point
* `categories::Vector{String}`             - Categories to be used in dataset
* `classes_to_idx::Dict{String, UInt8}`    - Dict mapping from shape name to class_idx
* `idx_to_classes::Dict{UInt8, String}`    - Dict mapping from class_idx to shape name
"""
struct ModelNet <: AbstractDataset
    root::String
    path::String
    train::Bool
    length::Int
    datapaths::Array
    transform::Any
    categories::Vector{String}
    classes_to_idx::Dict{String,UInt8}
    idx_to_classes::Dict{UInt8,String}
end

function _mn_dataset(
    root::String,
    train::Bool,
    download::Bool,
    transform,
    categories::Vector{String},
    variant::Int,
)
    root = normpath(root)
    _path = load_dataset(root, download, variant)
    train ? _split = "train" : _split = "test"
    datapaths = []
    for category in categories
        category in MN10_classes ||
            error("given category: $(category) is not a valid ModelNet10 category.")
        datapath = [
            (category, joinpath(_path, category, _split, filename))
            for
            filename in readdir(joinpath(_path, category, _split)) if
            split(filename, ".")[end] == "off"
        ]
        append!(datapaths, datapath)
    end

    classes_to_idx = Dict{String,UInt8}([(categories[i], i) for i = 1:length(categories)])
    idx_to_classes = Dict{UInt8,String}([(i, categories[i]) for i = 1:length(categories)])

    _length = length(datapaths)
    return ModelNet(
        root,
        _path,
        train,
        _length,
        datapaths,
        transform,
        categories,
        classes_to_idx,
        idx_to_classes,
    )
end

function load_dataset(root::String, download::Bool, variant::Int)
    ispath(root) || mkpath(root)
    local_dir = joinpath(root, "ModelNet$(variant)")
    local_path = joinpath(root, "ModelNet$(variant).zip")
    hash_mn10 = "9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51"
    hash_mn40 = "42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac"
    if (variant == 10)
        hash = hash_mn10
    elseif (variant == 40)
        hash = hash_mn40
    end

    if (!isdir(local_dir))
        if (!isfile(local_path))
            if (download)
                download_and_verify(
                    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet$(variant).zip",
                    local_path,
                    hash,
                )
            else
                error("dataset not found and auto-download option is set false.")
            end
        end
        run(`unzip -qo $local_path -d $root`)
    end

    return local_dir
end

function Base.getindex(v::ModelNet, idx::Int)
    data = Flux3D.load_trimesh(v.datapaths[idx][2])
    cls = v.classes_to_idx[v.datapaths[idx][1]]
    category = v.datapaths[idx][1]
    if v.transform != nothing
        data = v.transform(data)
    end
    return DataPoint(idx, data, cls, category)
end

Base.size(v::ModelNet) = (v.length,)
Base.length(v::ModelNet) = v.length

function Base.show(io::IO, dset::ModelNet)
    print(
        io,
        "ModelNet Dataset:",
        "\n    root: $(dset.root)",
        "\n    train: $(dset.train)",
        "\n    length: $(dset.length)",
        "\n    transform: $(dset.transform)",
        "\n    categories: $(length(dset.categories))",
    )
end
