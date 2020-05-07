abstract type AbstractDataset end
abstract type AbstractDataPoint end

function download_and_verify(url, path, hash)
    tmppath = tempname()
    download(url, tmppath)
    hash_download = open(tmppath) do f
        bytes2hex(SHA.sha256(f))
    end
    if hash_download !== hash
        msg  = "Hash Mismatch!\n"
        msg *= "  Expected sha256:   $hash\n"
        msg *= "  Calculated sha256: $hash_download"
        error(msg)
    end
    mv(tmppath, path; force=true)
end

Base.size(d::AbstractDataset) = error("define Base.size of dataset type: $(typeof(d)).")
Base.length(d::AbstractDataset) = error("define Base.length of dataset type: $(typeof(d)).")

Base.show(io::IO, p::AbstractDataPoint) = 
    print(io, "idx: $(p.idx), data: $(typeof(p.data))")

Base.show(io::IO, ::MIME"text/plain", p::AbstractDataPoint) = 
    print(io, "DataPoint:\n   ", p)

function Base.show(io::IO, dset::AbstractDataset) 
    print(io, "Dataset(")
    print("root = $(dset.root), ")
    print("train = $(dset.train), ")
    print("length = $(dset.length))")
end