abstract type AbstractDataset end
abstract type AbstractDataPoint end

const default_root = normpath(@__DIR__, "../../datasets/")

function download_and_verify(url, path, hash)
    tmppath = tempname()
    download(url, tmppath)
    hash_download = open(tmppath) do f
        bytes2hex(SHA.sha256(f))
    end
    if hash_download !== hash
        msg = "Hash Mismatch!\n"
        msg *= "  Expected sha256:   $hash\n"
        msg *= "  Calculated sha256: $hash_download"
        error(msg)
    end
    mv(tmppath, path; force = true)
end

struct DataPoint <: AbstractDataPoint
    idx::Int
    data::Flux3D.AbstractObject
    ground_truth::UInt8
    category_name::String
end

function Base.show(io::IO, d::DataPoint)
    print(
        io,
        "DataPoint:",
        "\n    idx: ",
        d.idx,
        "\n    data: ",
        typeof(d.data),
        "\n    ground_truth: ",
        d.ground_truth,
        "\n    category_name: ",
        d.category_name,
    )
end

Base.size(d::AbstractDataset) = error("define Base.size of dataset type: $(typeof(d)).")
Base.length(d::AbstractDataset) = error("define Base.length of dataset type: $(typeof(d)).")

Base.show(io::IO, p::AbstractDataPoint) =
    print(io, "idx: $(p.idx), data: $(typeof(p.data))")

function Base.show(io::IO, dset::AbstractDataset)
    print(io, "Dataset(")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length))")
end
