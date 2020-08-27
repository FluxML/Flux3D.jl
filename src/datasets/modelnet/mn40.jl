"""
    ModelNet40(;kwargs...)

Returns ModelNet40 dataset.

### Optional Key Arguments:
    * `root::String=default_root`   - Root directory of dataset
    * `train::Bool=true`            - Specifies the trainset
    * `transform=nothing`           - Transform to be applied to data point.
    * `categories::Vector{String}`  - Specifies the categories to be used in dataset.

### Examples:

```jldoctest
julia> dset = ModelNet40(train=false)
julia> typeof(dset[1].data) == TriMesh
```
"""
ModelNet40(;
    root::String = default_root,
    train::Bool = true,
    download::Bool = false,
    transform = nothing,
    categories::Vector{String} = MN10_classes,
) = _mn_dataset(root, train, download, transform, categories, 40)

const MN40_classes = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]
