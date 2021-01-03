export ScalePointCloud,
    RotatePointCloud,
    ReAlignPointCloud,
    NormalizePointCloud,
    ScaleTriMesh,
    RotateTriMesh,
    ReAlignTriMesh,
    NormalizeTriMesh,
    TranslateTriMesh,
    OffsetTriMesh,
    TriMeshToVoxelGrid,
    PointCloudToVoxelGrid,
    VoxelGridToTriMesh,
    PointCloudToTriMesh,
    TriMeshToPointCloud,
    VoxelGridToPointCloud


"""
    ScalePointCloud(factor::Number; inplace::Bool=true)

Scale PointCloud with a given scaling factor `factor`.

`factor` should be strictly greater than `0.0` for obvious reason.
`inplace` is optional keyword argument, to make transformation in-place.
If `inplace` is set to `false`, it will create deepcopy of PointCloud.
Given `factor`, this transform scale each point in PointCloud, ie. `point = point * factor`

See also: [`scale`](@ref), [`scale!`](@ref)
"""
struct ScalePointCloud <: AbstractTransform
    factor::Float32
    inplace::Bool
end

function ScalePointCloud(factor::Number; inplace::Bool = true)
    factor > 0.0 || error("factor must be greater than 0.0")
    ScalePointCloud(Float32(factor), inplace)
end

@functor ScalePointCloud

function (t::ScalePointCloud)(pcloud::PointCloud)
    t.inplace || (pcloud = deepcopy(pcloud))
    scale!(pcloud, t.factor)
    return pcloud
end

Base.show(io::IO, t::ScalePointCloud) =
    print(io, "$(typeof(t))(factor=$(t.factor); inplace=$(t.inplace))")

"""
    RotatePointCloud(rotmat::AbstractArray{<:Number,2}; inplace::Bool=true)

Rotate PointCloud with a given rotation matrix `rotmat`.

`rotmat` must be `AbstractArray{<:Number,2}` of size `(3,3)`.
`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of PointCloud.
Given `rotmat`, this transform will rotate each point coordinates (ie. x,y,z) in PointCloud.

See also: [`rotate`](@ref), [`rotate!`](@ref)
"""
struct RotatePointCloud <: AbstractTransform
    rotmat::AbstractArray{Float32,2}
    inplace::Bool
end

function RotatePointCloud(rotmat::AbstractArray{<:Number,2}; inplace::Bool = true)
    size(rotmat) == (3, 3) ||
        error("rotmat must be (3,3) array, but instead got $(size(rotmat)) array")
    return RotatePointCloud(Float32.(rotmat), inplace)
end

@functor RotatePointCloud

function (t::RotatePointCloud)(pcloud::PointCloud)
    t.inplace || (pcloud = deepcopy(pcloud))
    rotate!(pcloud, t.rotmat)
    return pcloud
end

Base.show(io::IO, t::RotatePointCloud) =
    print(io, "$(typeof(t))(rotmat; inplace=$(t.inplace))")

"""
    ReAlignPointCloud(target::PointCloud; inplace::Bool=true)
    ReAlignPointCloud(target::AbstractArray{<:Number, 2}; inplace::Bool=true)

Re-Align PointCloud to axis aligned bounding box of `target` PointCloud.

`input` PointCloud and `target` PointCloud should be of same size.
`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of PointCloud.

See also: [`realign`](@ref), [`realign!`](@ref)
"""
struct ReAlignPointCloud <: AbstractTransform
    t_min::AbstractArray{Float32,2}
    t_max::AbstractArray{Float32,2}
    inplace::Bool
end

function ReAlignPointCloud(target::PointCloud, index::Number = 1; inplace::Bool = true)
    points = target[index]
    t_min = minimum(points, dims = 2)
    t_max = maximum(points, dims = 2)
    ReAlignPointCloud(t_min, t_max, inplace)
end

ReAlignPointCloud(target::AbstractArray{<:Number}; inplace::Bool = true) =
    ReAlignPointCloud(PointCloud(target), inplace = inplace)

@functor ReAlignPointCloud

function (t::ReAlignPointCloud)(pcloud::PointCloud)
    t.inplace || (pcloud = deepcopy(pcloud))
    realign!(pcloud, t.t_min, t.t_max)
    return pcloud
end

Base.show(io::IO, t::ReAlignPointCloud) =
    print(io, "$(typeof(t))(target=PointCloud(...); inplace=$(t.inplace))")

"""
    NormalizePointCloud(; inplace::Bool=true)

Normalize PointCloud with mean at origin and unit standard deviation.

`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of PointCloud.

See also: [`normalize`](@ref), [`normalize!`](@ref)
"""
struct NormalizePointCloud <: AbstractTransform
    inplace::Bool
end

NormalizePointCloud(; inplace::Bool = true) = NormalizePointCloud(inplace)

@functor NormalizePointCloud

function (t::NormalizePointCloud)(pcloud::PointCloud)
    t.inplace || (pcloud = deepcopy(pcloud))
    normalize!(pcloud)
    return pcloud
end

Base.show(io::IO, t::NormalizePointCloud) = print(io, "$(typeof(t))(;inplace=$(t.inplace))")

#TODO: Add support for multidimension factor
"""
    ScaleTriMesh(factor::Number; inplace::Bool=true)

Scale TriMesh with a given scaling factor `factor`.

`factor` should be strictly greater than `0.0` for obvious reason.
`inplace` is optional keyword argument, to make transformation in-place.
If `inplace` is set to `false`, it will create deepcopy of TriMesh.
Given `factor`, this transform scale each vertices in TriMesh, ie. `point = point * factor`

See also: [`scale`](@ref), [`scale!`](@ref)
"""
struct ScaleTriMesh <: AbstractTransform
    factor::Float32
    inplace::Bool
end

function ScaleTriMesh(factor::Number; inplace::Bool = true)
    factor > 0.0 || error("factor must be greater than 0.0")
    ScaleTriMesh(Float32(factor), inplace)
end

@functor ScaleTriMesh

function (t::ScaleTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    scale!(m, t.factor)
    return m
end

Base.show(io::IO, t::ScaleTriMesh) =
    print(io, "$(typeof(t))(factor=$(t.factor); inplace=$(t.inplace))")

"""
    RotateTriMesh(rotmat::AbstractArray{<:Number,2}; inplace::Bool=true)

Rotate vertices of TriMesh with a given rotation matrix `rotmat`.

`rotmat` must be `AbstractArray{<:Number,2}` of size `(3,3)`.
`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of TriMesh.
Given `rotmat`, this transform will rotate each vertices coordinates (ie. x,y,z) in TriMesh.

See also: [`rotate`](@ref), [`rotate!`](@ref)
"""
struct RotateTriMesh <: AbstractTransform
    rotmat::AbstractArray{Float32,2}
    inplace::Bool
end

function RotateTriMesh(rotmat::AbstractArray{<:Number,2}; inplace::Bool = true)
    size(rotmat) == (3, 3) ||
        error("rotmat must be (3,3) array, but instead got $(size(rotmat)) array")
    return RotateTriMesh(Float32.(rotmat), inplace)
end

@functor RotateTriMesh

function (t::RotateTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    rotate!(m, t.rotmat)
    return m
end

Base.show(io::IO, t::RotateTriMesh) =
    print(io, "$(typeof(t))(rotmat; inplace=$(t.inplace))")

"""
    ReAlignTriMesh(target::TriMesh; inplace::Bool=true)

Re-Align TriMesh to axis aligned bounding box of mesh at `index` in TriMesh `target`.

`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of TriMesh.

See also: [`realign`](@ref), [`realign!`](@ref)
"""
struct ReAlignTriMesh <: AbstractTransform
    t_min::AbstractArray{Float32,2}
    t_max::AbstractArray{Float32,2}
    inplace::Bool
end

function ReAlignTriMesh(target::TriMesh, index::Integer = 1; inplace::Bool = true)
    verts = get_verts_list(target)[index]
    t_min = minimum(verts, dims = 2)
    t_max = maximum(verts, dims = 2)
    ReAlignTriMesh(t_min, t_max, inplace)
end

@functor ReAlignTriMesh

function (t::ReAlignTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    realign!(m, t.t_min, t.t_max)
    return m
end

Base.show(io::IO, t::ReAlignTriMesh) =
    print(io, "$(typeof(t))(target=TriMesh(...); inplace=$(t.inplace))")

"""
    NormalizeTriMesh(; inplace::Bool=true)

Normalize TriMesh with mean at origin and unit standard deviation.

`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of TriMesh.

See also: [`normalize`](@ref), [`normalize!`](@ref)
"""
struct NormalizeTriMesh <: AbstractTransform
    inplace::Bool
end

NormalizeTriMesh(; inplace::Bool = true) = NormalizeTriMesh(inplace)

@functor NormalizeTriMesh

function (t::NormalizeTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    normalize!(m)
    return m
end

Base.show(io::IO, t::NormalizeTriMesh) = print(io, "$(typeof(t))(;inplace=$(t.inplace))")

"""
    TranslateTriMesh(vector::AbstractArray{<:Number}; inplace::Bool=true)

Translate TriMesh with given translation `vector`.

`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of TriMesh.

See also: [`translate`](@ref), [`translate!`](@ref)
"""
struct TranslateTriMesh <: AbstractTransform
    vector::AbstractArray{Float32,1}
    inplace::Bool
end

function TranslateTriMesh(vector::AbstractArray{<:Number,1}; inplace::Bool = true)
    (size(vector) == (3,)) ||
        error("vector must be (3, ), but instead got $(size(vector)) array")
    TranslateTriMesh(Float32.(vector), inplace)
end

TranslateTriMesh(vector::Number; inplace::Bool = true) =
    TranslateTriMesh(fill(vector, (3,)); inplace = inplace)

@functor TranslateTriMesh

function (t::TranslateTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    translate!(m, t.vector)
    return m
end

Base.show(io::IO, t::TranslateTriMesh) =
    print(io, "$(typeof(t))(vector=$(t.vector);inplace=$(t.inplace))")

"""
    OffsetTriMesh(offset_verts::AbstractArray{<:Number,2}; inplace::Bool=true)

Add offset to the TriMesh by given offset vertices `offset_verts`

`inplace` is optional keyword argument, to make transformation in-place
If `inplace` is set to `false`, it will create deepcopy of TriMesh.

See also: [`offset`](@ref), [`offset!`](@ref)
"""
struct OffsetTriMesh <: AbstractTransform
    offset_verts::AbstractArray{Float32,2}
    inplace::Bool
end

OffsetTriMesh(offset_verts::AbstractArray{<:Number,2}; inplace::Bool = true) =
    OffsetTriMesh(Float32.(offset_verts), inplace)

@functor OffsetTriMesh

function (t::OffsetTriMesh)(m::TriMesh)
    t.inplace || (m = deepcopy(m))
    offset!(m, t.offset_verts)
    return m
end

Base.show(io::IO, t::OffsetTriMesh) =
    print(io, "$(typeof(t))(offset_verts; inplace=$(t.inplace))")

"""
    TriMeshToVoxelGrid(resolution::Int=32)

Converts a TriMesh to VoxelGrid having specified `resolution`.

See also: [`TriMesh`](@ref), [`VoxelGrid`](@ref)
"""
struct TriMeshToVoxelGrid <: AbstractTransform
    resolution::Int
    TriMeshToVoxelGrid(res::Int = 32) = new(res)
end

(t::TriMeshToVoxelGrid)(m::TriMesh) = VoxelGrid(m, t.resolution)

Base.show(io::IO, t::TriMeshToVoxelGrid) =
    print(io, "$(typeof(t))(resolution=$(t.resolution))")

"""
    PointCloudToVoxelGrid(resolution::Int=32)

Converts a PointCloud to VoxelGrid having specified `resolution`.

See also: [`PointCloud`](@ref), [`VoxelGrid`](@ref)
"""
struct PointCloudToVoxelGrid <: AbstractTransform
    resolution::Int
    PointCloudToVoxelGrid(res::Int = 32) = new(res)
end

(t::PointCloudToVoxelGrid)(p::PointCloud) = VoxelGrid(p, t.resolution)

Base.show(io::IO, t::PointCloudToVoxelGrid) =
    print(io, "$(typeof(t))(resolution=$(t.resolution))")

"""
    VoxelGridToTriMesh(; threshold=0.5, algo=:Exact)

Converts a VoxelGrid to TriMesh.

`threshold` is the threshold from which to make binary voxels, and `algo`
is the mode to be used to convert binary voxels. Available `algo` are
[:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets].

See also: [`TriMesh`](@ref), [`VoxelGrid`](@ref)
"""
struct VoxelGridToTriMesh <: AbstractTransform
    threshold::Float32
    algo::Symbol
end

function VoxelGridToTriMesh(; thresh::Number = 0.5f0, algo = :MarchingCubes)
    algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets] || error(
        "given algo=$(algo) is not supported. Accepted algos are {:Exact,:MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets}.",
    )
    (0 <= thresh <= 1) || error("given threshold=$(thresh) is not between [0,1]")
    return VoxelGridToTriMesh(Float32(thresh), algo)
end

(t::VoxelGridToTriMesh)(v::VoxelGrid) = TriMesh(v; thresh = t.threshold, algo = t.algo)

Base.show(io::IO, t::VoxelGridToTriMesh) =
    print(io, "$(typeof(t))((threshold=$(t.threshold), algo=$(t.algo))")

"""
    PointCloudToTriMesh(resolution::Int=32)

Converts a PointCloud to TriMesh having specified `resolution`.

See also: [`PointCloud`](@ref), [`TriMesh`](@ref)
"""
struct PointCloudToTriMesh <: AbstractTransform
    resolution::Int
    algo::Symbol
    function PointCloudToTriMesh(res::Int = 32; algo = :MarchingCubes)
        algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets] || error(
            "given algo=$(algo) is not supported. Accepted algos are {:Exact,:MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets}.",
        )
        return new(res, algo)
    end
end

(t::PointCloudToTriMesh)(p::PointCloud) = TriMesh(p, t.resolution)

Base.show(io::IO, t::PointCloudToTriMesh) =
    print(io, "$(typeof(t))(resolution=$(t.resolution))")

"""
    TriMeshToPointCloud(npoints::Int=1000)

Converts a TriMesh to PointCloud having `npoints`.

See also: [`TriMesh`](@ref), [`PointCloud`](@ref)
"""
struct TriMeshToPointCloud <: AbstractTransform
    npoints::Int
    function TriMeshToPointCloud(npoints::Int = 1024)
        npoints >= 0 || error("npoints cannot be less than 0")
        return new(npoints)
    end
end

(t::TriMeshToPointCloud)(m::TriMesh) = PointCloud(m, t.npoints)

Base.show(io::IO, t::TriMeshToPointCloud) = print(io, "$(typeof(t))(npoints=$(t.npoints))")

"""
    VoxelGridToPointCloud(npoints::Int=1000; threshold=0.5, algo=:Exact)

Converts a VoxelGrid to PointCloud having `npoints`.

`threshold` is the threshold from which to make binary voxels, and `algo`
is the mode to be used to convert binary voxels. Available `algo` are
[:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets].

See also: [`PointCloud`](@ref), [`VoxelGrid`](@ref)
"""
struct VoxelGridToPointCloud <: AbstractTransform
    npoints::Int
    threshold::Float32
    algo::Symbol
end

function VoxelGridToPointCloud(
    npoints::Int = 1024;
    thresh::Number = 0.5f0,
    algo::Symbol = :MarchingCubes,
)
    npoints >= 0 || error("npoints cannot be less than 0")
    algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets] || error(
        "given algo=$(algo) is not supported. Accepted algos are {:Exact,:MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets}.",
    )
    (0 <= thresh <= 1) || error("given threshold=$(thresh) is not between [0,1]")
    return VoxelGridToPointCloud(npoints, Float32(thresh), algo)
end

(t::VoxelGridToPointCloud)(v::VoxelGrid) =
    PointCloud(v, t.npoints; thresh = t.threshold, algo = t.algo)

Base.show(io::IO, t::VoxelGridToPointCloud) = print(
    io,
    "$(typeof(t))(npoints=$(t.npoints), threshold=$(t.threshold), algo=$(t.algo))",
)
