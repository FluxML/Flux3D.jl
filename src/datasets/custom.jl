"""
    CustomDataset

Minimal Custom Dataset.
`CustomDataset` also support indexing and slicing. 

### Fields:

* `length::Int`         - Length of dataset. 
* `getdata::Function`   - Function which takes `idx` as input and returns corresponding data   

### Available Contructor:

* `CustomDataset(length::Int, getdata::Function)`
* `CustomDataset(;length::Int, getdata::Function)`

### Examples:

```julia
julia> x = rand(10,32)
julia> getdata(idx) = x[idx,:]
julia> dset = CustomDataset(size(x,1), getdata)
julia> [x[1,:], x[2,:]] == dset[1:2]
```
"""
struct CustomDataset <: AbstractDataset
    length::Int
    getdata::Function
end

CustomDataset(; length::Int, getdata::Function) = CustomDataset(length, getdata)

Base.size(d::CustomDataset) = (d.length,)
Base.length(d::CustomDataset) = d.length

Base.firstindex(d::CustomDataset) = 1
Base.lastindex(d::CustomDataset) = length(d)

Base.getindex(d::CustomDataset, idx::Int) = d.getdata(idx)
Base.getindex(d::CustomDataset, r::AbstractArray{<:Any,1}) = [d[ri] for ri in r] #TODO: revise this
Base.getindex(d::CustomDataset, c::Colon) = d[1:length(d)]

Base.show(io::IO, d::CustomDataset) =
    print(io, "CustomDataset(length=$(d.length), getdata=$(d.getdata))")
