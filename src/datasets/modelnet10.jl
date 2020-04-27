function ModelNet10(;mode="point_cloud")

    if mode == "point_cloud"
        return ModelNet10PCloud
    else
        error("Selected mode: $(mode) is not supported (Currently supported mode are {\"point_cloud\"}).")
    end
end

const MN10_classes_to_idx = Dict{String, UInt8}([("bathtub",1), ("bed",2), ("chair",3), ("desk",4), ("dresser",5),
    ("monitor",6), ("night_stand",7), ("sofa",8), ("table",9), ("toilet",10)])

const MN10_idx_to_classes = Dict{UInt8, String}([(1,"bathtub"), (2,"bed"), (3,"chair"), (4,"desk"), (5,"dresser"),
    (6,"monitor"), (7,"night_stand"), (8,"sofa"), (9,"table"), (10,"toilet")])

const MN10_classes = ["bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet"]

struct MN10DataPoint <: AbstractDataPoint
    idx::Int
    data::Union{PointCloud}
    ground_truth::UInt8
end

struct ModelNet10PCloud <: AbstractDataset
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

function MN10_extract(datapath, npoints)
    cls = MN10_classes_to_idx[datapath[1]]
    pset = Array{Float32}(undef, npoints, 3)
    stream = open(datapath[2], "r")
    for i in 1:npoints
        pset[i, :] = map((x->parse(Float32, x)), split(readline(stream, keep=false), ",")[1:3])
    end
    return (pset,cls)
end

function ModelNet10PCloud(;root::String=default_root, train::Bool=true, npoints::Int=1024, transform=nothing, sampling=nothing)
    _path = normpath(dataset("ModelNet10PCloud", root))
    train ? _split="train" : _split="test"
    shapeids = [line for line in readlines(joinpath(_path, "modelnet10_$(_split).txt"))]
    shape_names = [join(split(shapeids[i], "_")[1:end-1], "_") for i in 1:length(shapeids)]
    datapaths = [(shape_names[i], joinpath(_path, shape_names[i], (shapeids[i])*".txt")) for i in 1:length(shapeids)]
    _length = length(datapaths)
    ModelNet10PCloud(root, _path, train, transform, npoints, sampling, datapaths, _length, MN10_classes_to_idx, MN10_classes)
end

function Base.getindex(v::ModelNet10PCloud, idx::Int)
    pset, cls = MN10_extract(v.datapaths[idx], v.npoints)
    if v.transform != nothing
        data = v.transform(PointCloud(pset))
    else
        data = PointCloud(pset)
    end
    return MN10DataPoint(idx, data, cls)
end

Base.size(v::ModelNet10PCloud) = (v.length,)
Base.length(v::ModelNet10PCloud) = v.length

Base.show(io::IO, p::MN10DataPoint) = 
    print(io, "idx: $(p.idx), data: $(typeof(p.data)), ground_truth: $(p.ground_truth) ($(MN10_idx_to_classes[p.ground_truth]))")

Base.show(io::IO, ::MIME"text/plain", p::MN10DataPoint) = 
    print(io, "ModelNet10 DataPoint:\n   ", p)

function Base.show(io::IO, dset::ModelNet10PCloud) 
    print(io, "ModelNet10(")
    print("mode = point_cloud, ")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length))")
end