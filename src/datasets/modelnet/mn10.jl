"""
    ModelNet10(;kwargs...)

Returns ModelNet10 dataset.

### Optional Key Arguments:
    * `root::String=default_root`   - Root directory of dataset
    * `train::Bool=true`            - Specifies the trainset
    * `transform=nothing`           - Transform to be applied to data point.
    * `categories::Vector{String}`  - Specifies the categories to be used in dataset.

### Examples:

```jldoctest
julia> dset = ModelNet10(train=false)
julia> typeof(dset[1].data) == TriMesh
```
"""
ModelNet10(;
    root::String = default_root,
    train::Bool = true,
    download::Bool = false,
    transform = nothing,
    categories::Vector{String} = MN10_classes,
) = _mn_dataset(root, train, download, transform, categories, 10)

const MN10_classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]
