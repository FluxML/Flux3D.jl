```@meta
CurrentModule = Flux3D
```

# Flux3D: 3D Deep Learning Library in Julia

Flux3D.jl is a 3D vision library, written completely in Julia. This package utilizes [Flux.jl](github.com/FluxML/Flux.jl) and [Zygote.jl](github.com/FluxML/Zygote.jl) as its building blocks for training 3D vision models and for supporting differentiation. This package also have support of CUDA GPU acceleration with [CUDA.jl](github.com/JuliaGPU/CUDA.jl).The primary motivation for this library is to provide:

* Batched Data structure for 3D data like PointCloud and TriMesh for storing and computation.
* Transforms and general utilities for processing 3D structures.
* Metrics for defining loss objectives and predefined 3D models.
* Easy access to loading an pre-processing standard 3D datasets.
* Visualization utilities for PointCloud and TriMesh.

Any suggestions, issues and pull requests are most welcome.

---

## Installation

Download Julia 1.3 or later.

Currently the library is under development and is not registered. But to install the master branch, type the following in the julia prompt.

```julia
] add https://github.com/FluxML/Flux3D.jl
```

!!! note
    Rendering of 3D structure is done using AbstractPlotting. Therefore, for visualization purpose we will be required to install Makie and compatible backend (GLMakie or WGLMakie). To install it simply run `] add Makie` in the julia prompt.

---

## Contents

```@contents
Pages = ["index.md"]
Depth = 2
```

### 3D Structures

```@contents
Pages = [
    "rep/pointcloud.md"
    "rep/trimesh.md"
]
Depth = 2
```

### Datasets

```@contents
Pages = [
    "datasets/modelnet.md"
    "datasets/utils.md"
]
Depth = 2
```

### API Documentation

```@contents
Pages = [
    "api/transforms.md"
    "api/metrics.md"
    "api/visualize.md"
]
Depth = 2
```
