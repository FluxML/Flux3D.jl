function ModelNet40(;mode="point_cloud")

    if mode == "point_cloud"
        return ModelNet40PCloud
    else
        error("Selected mode: $(mode) is not supported (Currently supported mode are {\"point_cloud\"}).")
    end
end

struct MN40DataPoint <: AbstractDataPoint
    idx::Int
    data::Union{PointCloud}
    ground_truth::UInt8
end

struct ModelNet40PCloud <: AbstractDataset
    root::String 
    path::String #contains the path to dataset
    train::Bool
    transform::Union{Flux3D.AbstractTransform, Nothing}
    npoints::Int
    sampling #TODO Add type-assertion accordingly to include two possible option {"top", "uniform"}
    datapaths::Array
    length::Int
    classes_to_idx::Dict{String, UInt8}
    classes::Array{String,1}
end

function MN40_extract(path, npoints)
    pset = Array{Float32}(undef, npoints, 3)
    stream = open(path, "r")
    for i in 1:npoints
        pset[i, :] = map((x->parse(Float32, x)), split(readline(stream, keep=false), ",")[1:3])
    end
    return pset
end

function ModelNet40PCloud(;root::String=default_root, train::Bool=true, npoints::Int=1024, transform=nothing, sampling=nothing)
    _path = normpath(dataset("ModelNet40PCloud", root))
    train ? _split="train" : _split="test"
    cat_file = joinpath(_path, "modelnet40_shape_names.txt")
    
    classes = []
    classes_to_idx = []
    for (i, line) in enumerate(readlines(cat_file))
        push!(classes, line)
        push!(classes_to_idx, (line, convert(UInt8,i)))
    end
    classes_to_idx = Dict{String, UInt8}(classes_to_idx)

    shapeids = [line for line in readlines(joinpath(_path, "modelnet40_$(_split).txt"))]
    shape_names = [join(split(shapeids[i], "_")[1:end-1], "_") for i in 1:length(shapeids)]
    datapaths = [(shape_names[i], joinpath(_path, shape_names[i], (shapeids[i])*".txt")) for i in 1:length(shapeids)]
    _length = length(datapaths)
    ModelNet40PCloud(root, _path, train, transform, npoints, sampling, datapaths, _length, classes_to_idx, classes)
end

function Base.getindex(v::ModelNet40PCloud, idx::Int)
    cls = v.classes_to_idx[v.datapaths[idx][1]]
    pset = MN40_extract(v.datapaths[idx][2], v.npoints)
    if v.transform != nothing
        data = v.transform(PointCloud(pset))
    else
        data = PointCloud(pset)
    end
    return MN40DataPoint(idx, data, cls)
end

Base.size(v::ModelNet40PCloud) = (v.length,)
Base.length(v::ModelNet40PCloud) = v.length

Base.show(io::IO, p::MN40DataPoint) = 
    print(io, "idx: $(p.idx), data: $(typeof(p.data)), ground_truth: $(p.ground_truth)")

Base.show(io::IO, ::MIME"text/plain", p::MN40DataPoint) = 
    print(io, "ModelNet40 DataPoint:\n   ", p)

function Base.show(io::IO, dset::ModelNet40PCloud) 
    print(io, "ModelNet40(")
    print("mode = point_cloud, ")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length))")
end