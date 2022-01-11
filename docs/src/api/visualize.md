```@meta
CurrentModule = Flux3D
```

# Vusualize functions

!!! note
    Rendering of 3D structure is done using AbstractPlotting. Therefore, for visualization purpose we will be required to install Makie and compatible backend (GLMakie or WGLMakie). To install it simply run `] add Makie` in the julia prompt.

## Example

```julia
julia> using Flux3D, Makie

julia> Makie.set_theme!(show_axis = false, scale=false)

julia> m = load_trimesh("teapot.obj")
TriMesh{Float32, UInt32, Array} Structure:
    Batch size: 1
    Max verts: 1202
    Max faces: 2256
    offset: -1
    Storage type: Array

julia> p = PointCloud(m)
PointCloud{Float32} Structure:
    Batch size: 1
    Points: 1000
    Normals 0
    Storage type: Array{Float32,3}

julia> v = VoxelGrid(m)
VoxelGrid{Float32} Structure:
    Batch size: 1
    Voxels features: 32
    Storage type: Array{Float32,4}

julia> vbox(visulize(m), visualize(p), visualize(v))
```

```@raw html
<p align="center">
    <img width=600 height=300 src="../../assets/visualize.png">
</p>
```

## Visualize

```@docs
visualize
```
