export VoxelGrid

"""
    VoxelGrid

Initialize VoxelGrid representation.

`voxels` should be Array of size `(N, N, N, B)` where `N` is the number of
voxels features and `B` is the batch size of VoxelGrid.

### Fields:

- `voxels`      - voxels features of VoxelGrid.

### Available Contructor:

- `VoxelGrid(voxels::AbstractArray)`
- `VoxelGrid(;voxelsAbstractArray)`
- `VoxelGrid(v::VoxelGrid)`
"""
mutable struct VoxelGrid{T<:Float32} <: AbstractObject
    voxels::AbstractArray{T,4}
end

# VoxelGrid(voxels::AbstractArray{Float32,4}) = VoxelGrid(voxels)

VoxelGrid(voxels::AbstractArray{T,4}) where {T} = VoxelGrid(Float32.(voxels))

function VoxelGrid(voxels::AbstractArray{T,3}) where {T}
    voxels = reshape(voxels, size(voxels)...,1)
    return VoxelGrid(voxels)
end

VoxelGrid(;voxels) = VoxelGrid(voxels)
VoxelGrid(v::VoxelGrid) = VoxelGrid(v.voxels)

@functor VoxelGrid

Base.getindex(v::VoxelGrid, index::Number) = v.voxels[:, :, :, index]

function Base.show(io::IO, m::VoxelGrid{T}) where {T}
    print(
        io,
        "VoxelGrid{$(T)} Structure:\n    Batch size: ",
        size(m.voxels, 4),
        "\n    Voxels features: ",
        size(m.voxels, 1),
        "\n    Storage type: ",
        typeof(m.voxels),
    )
end

_assert_voxel(v::VoxelGrid) = all(0.0 .>= v .<= 1.0)
