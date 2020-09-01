```@meta
CurrentModule = Flux3D
```

# Transforms

## Usage

Transforms can be applied to corresponding 3D structure.
`Chain` can be used to compose multiple functions/transforms. Same as `Flux.Chain`.

`Chain` also supports indexing and slicing. See Example below.

#### Example

```julia
julia> t = Chain(ScalePointCloud(2.0), NormalizePointCloud())
Chain(ScalePointCloud(factor=2.0; inplace=true), NormalizePointCloud(;inplace=true))

julia> p = PointCloud(rand(1024,3))
PointCloud{Float32} Structure:
    Batch size: 1
    Points: 3
    Normals 0
    Storage type: Array{Float32,3}

julia> t(p) == t[2](t[1](p))
true

julia> m = load_trimesh("teapot.obj")
TriMesh{Float32, UInt32, Array} Structure:
    Batch size: 1
    Max verts: 1202
    Max faces: 2256
    offset: -1
    Storage type: Array

julia> t = NormalizeTriMesh()
NormalizeTriMesh(;inplace=true)

julia> t(m)
TriMesh{Float32, UInt32, Array} Structure:
    Batch size: 1
    Max verts: 1202
    Max faces: 2256
    offset: -1
    Storage type: Array
```

---

## List of all available Transforms

```@index
Pages = ["transforms.md"]
```

---

## TriMesh Transforms

```@docs
NormalizeTriMesh
ScaleTriMesh
RotateTriMesh
ReAlignTriMesh
TranslateTriMesh
OffsetTriMesh
```

---

## PointCloud Transforms

```@docs
NormalizePointCloud
ScalePointCloud
RotatePointCloud
ReAlignPointCloud
```

---

## Conversions Transforms

```@docs
TriMeshToVoxelGrid
PointCloudToVoxelGrid
VoxelGridToTriMesh
PointCloudToTriMesh
TriMeshToPointCloud
VoxelGridToPointCloud
```
