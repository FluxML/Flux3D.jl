function pcloud_load(root::String = default_root)
    #TODO: donwload link of ModelNet is down
    error("Autodownload is currently not supported for ModelNet. \n
           Download dataset from following link
           https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip")
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
    ModelNet40PCloud

PointCloud version of ModelNet40 dataset.

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
struct ModelNet40PCloud <: AbstractDataset
    root::String
    path::String
    train::Bool
    length::Int
    datapaths::Array #TODO add detailed type assertion
    npoints::Int
    transform::Union{Flux3D.AbstractTransform,Nothing}
    sampling::Any #TODO Add type-assertion accordingly to include two possible option {"top", "uniform"}
    classes_to_idx::Dict{String,UInt8}
    classes::Array{String,1}
end

function pcloud_extract(path, npoints)
    pset = Array{Float32}(undef, npoints, 3)
    nset = Array{Float32}(undef, npoints, 3)
    stream = open(path, "r")
    for i = 1:npoints
        tmp = map((x -> parse(Float32, x)), split(readline(stream, keep = false), ","))
        pset[i, :] = tmp[1:3]
        nset[i, :] = tmp[4:6]
    end
    return (pset, nset)
end

function ModelNet40PCloud(;
    root::String = default_root,
    train::Bool = true,
    npoints::Int = 1024,
    transform = nothing,
    sampling = nothing,
)
    _path = pcloud_load(root)
    train ? _split = "train" : _split = "test"
    shapeids = [line for line in readlines(joinpath(_path, "modelnet40_$(_split).txt"))]
    shape_names = [join(split(shapeids[i], "_")[1:end-1], "_") for i = 1:length(shapeids)]
    datapaths = [
        (shape_names[i], joinpath(_path, shape_names[i], (shapeids[i]) * ".txt"))
        for i = 1:length(shapeids)
    ]
    _length = length(datapaths)
    ModelNet40PCloud(
        root,
        _path,
        train,
        _length,
        datapaths,
        npoints,
        transform,
        sampling,
        MN40_classes_to_idx,
        MN40_classes,
    )
end

function Base.getindex(v::ModelNet40PCloud, idx::Int)
    cls = v.classes_to_idx[v.datapaths[idx][1]]
    (pset, nset) = pcloud_extract(v.datapaths[idx][2], v.npoints)
    if v.transform != nothing
        data = v.transform(PointCloud(pset, nset))
    else
        data = PointCloud(pset, nset)
    end
    return MN40DataPoint(idx, data, cls)
end

Base.size(v::ModelNet40PCloud) = (v.length,)
Base.length(v::ModelNet40PCloud) = v.length

function Base.show(io::IO, dset::ModelNet40PCloud)
    print(io, "ModelNet40(")
    print("mode = point_cloud, ")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length))")
end
