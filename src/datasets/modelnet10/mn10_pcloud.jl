function pcloud_load(root::String = default_root)
    mkpath(root)
    local_dir = joinpath(root, "modelnet40_normal_resampled")
    local_path = joinpath(root, "modelnet40_normal_resampled.zip")
    hash = "d64e9c5cfc479bac3260b164ae3c75ba83e94a1d216fbcd3f59ce2a9686d3762"

    if (!isdir(local_dir))
        if (!isfile(local_path))
            # dataset prepared by authors of pointnet2
            download_and_verify(
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip",
                local_path,
                hash,
            )
        end
        run(`unzip -q $local_path -d $root`)
    end
    return local_dir
end

"""
    ModelNet10PCloud

PointCloud version of ModelNet10 dataset.

### Fields:

* `root::String`    - Root directory of dataset
* `path::String`    - Directory of dataset                                          
* `train::Bool`     - Specifies the trainset
* `length::Int`     - Length of dataset.
* `datapaths::Array`    - Array containing the shape and path for each datapoint.
* `npoints::Int`        - Number of points and normals in each PointCloud.
* `transform::Union{Flux3D.AbstractTransform, Nothing}` - - Transform to be applied to data point.
* `sampling::Nothing`   - 'to be implement'
* `classes_to_idx::Dict{String, UInt8}` - Dict mapping from shape name to class_idx
* `classes::Array{String,1}`    - Array of shape names.

"""
struct ModelNet10PCloud <: AbstractDataset
    root::String
    path::String
    train::Bool
    length::Int
    datapaths::Array
    npoints::Int
    transform::Union{Flux3D.AbstractTransform,Nothing}
    sampling::Any #TODO Add type-assertion accordingly to include two possible option {"top", "uniform"}
    classes_to_idx::Dict{String,UInt8}
    classes::Array{String,1}
end

function pcloud_extract(datapath, npoints)
    cls = MN10_classes_to_idx[datapath[1]]
    pset = Array{Float32}(undef, npoints, 3)
    nset = Array{Float32}(undef, npoints, 3)
    stream = open(datapath[2], "r")
    for i = 1:npoints
        tmp = map((x -> parse(Float32, x)), split(readline(stream, keep = false), ","))
        pset[i, :] = tmp[1:3]
        nset[i, :] = tmp[4:6]
    end
    return (pset, nset, cls)
end

function ModelNet10PCloud(;
    root::String = default_root,
    train::Bool = true,
    npoints::Int = 1024,
    transform = nothing,
    sampling = nothing,
)
    _path = pcloud_load(root)
    train ? _split = "train" : _split = "test"
    shapeids = [line for line in readlines(joinpath(_path, "modelnet10_$(_split).txt"))]
    shape_names = [join(split(shapeids[i], "_")[1:end-1], "_") for i = 1:length(shapeids)]
    datapaths = [
        (shape_names[i], joinpath(_path, shape_names[i], (shapeids[i]) * ".txt"))
        for i = 1:length(shapeids)
    ]
    _length = length(datapaths)
    ModelNet10PCloud(
        root,
        _path,
        train,
        _length,
        datapaths,
        npoints,
        transform,
        sampling,
        MN10_classes_to_idx,
        MN10_classes,
    )
end

function Base.getindex(v::ModelNet10PCloud, idx::Int)
    pset, nset, cls = pcloud_extract(v.datapaths[idx], v.npoints)
    if v.transform != nothing
        data = v.transform(PointCloud(pset, nset))
    else
        data = PointCloud(pset, nset)
    end
    return MN10DataPoint(idx, data, cls)
end

Base.size(v::ModelNet10PCloud) = (v.length,)
Base.length(v::ModelNet10PCloud) = v.length

function Base.show(io::IO, dset::ModelNet10PCloud)
    print(io, "ModelNet10(")
    print("mode = point_cloud, ")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length), ")
    print("npoints = $(dset.npoints), ")
    print("transform = $(typeof(dset.transform))(...))")
end
