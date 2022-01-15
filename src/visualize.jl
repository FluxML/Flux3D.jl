import Makie
import GeometryBasics
export visualize, visualize!

"""
    visualize(pcloud::PointCloud; kwargs...)

Visualize PointCloud `pcloud` at `index`.

Dimension of points in PointCloud `pcloud` must be 3.

### Optional Arguments:
- color (Symbol)       - Color of the marker, default `:blue`
- markersize (Number)  - Size of the marker, default `npoints(pcloud)/5000`

"""
function visualize(p::PointCloud, index::Number = 1; kwargs...)
    points = cpu(p[index])
    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :lightgreen)
    get!(kwargs, :markersize, npoints(p) / 5000)

    Makie.meshscatter(points[1, :], points[2, :], points[3, :]; kwargs...)
end

function visualize!(axis3::Makie.Axis3, p::PointCloud, index::Number = 1; kwargs...)
    points = cpu(p[index])
    size(points, 1) == 3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :lightgreen)
    get!(kwargs, :markersize, npoints(p) / 5000)

    Makie.meshscatter!(axis3, points[1, :], points[2, :], points[3, :]; kwargs...)
end

"""
    visualize(m::TriMesh, index::Int=1; kwargs...)

Visualize mesh at `index` in TriMesh `m`.

### Optional Arguments:
- color (Symbol)       - Color of the marker, default `:red`

"""
function visualize(m::GeometryBasics.Mesh; kwargs...) where {T,R}
    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :orange)

    Makie.mesh(GeometryBasics.normal_mesh(m); kwargs...)
end

function visualize!(axis3::Makie.Axis3, m::GeometryBasics.Mesh; kwargs...) where {T,R}
    Makie.mesh!(
        axis3,
        visualize(m; kwargs...).plot.input_args[1].val
    )
end

visualize(m::TriMesh, index::Int = 1; kwargs...) = visualize(GBMesh(m, index); kwargs...)

"""
     visualize!(axis3::Makie.Axis3, object, args...; kwargs...)
Similar to `Flux3D.visualize` except it accepts `axis3::Makie.Axis3` and update it by rendering `object`.
See also [`visualize`](@ref).
"""
function visualize!(axis3::Makie.Axis3, m::TriMesh, args...; kwargs...)
    Makie.mesh!(
        axis3,
        visualize(GBMesh(m, args...); kwargs...).plot.input_args[1].val
    )
end

"""
    visualize(v::VoxelGrid, index::Int=1; kwargs...)

Visualize voxel at `index` in VoxelGrid `v`.

### Optional Arguments:
- color (Symbol)       - Color of the marker, default `:red`

"""
function visualize(
    v::VoxelGrid,
    index::Int = 1,
    thresh::Number = 0.49f0;
    algo = :Exact,
    kwargs...,
)
    algo in [:Exact, :MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets] ||
        error("given algo: $(algo) is not supported. Accepted algo are
              {:Exact,:MarchingCubes, :MarchingTetrahedra, :NaiveSurfaceNets}.")
    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :violet)
    method = algo == :Exact ? _voxel_exact : _voxel_algo
    v, f = method(cpu(v[index]), Float32(thresh), algo)

    m = GBMesh(v, f)
    Makie.mesh(GeometryBasics.normal_mesh(m); kwargs...)
end

function visualize!(
    axis3::Makie.Axis3,
    v::VoxelGrid,
    args...;
    kwargs...,
)
    Makie.mesh!(
        axis3,
        visualize(v, args...,; kwargs...).plot.input_args[1].val,
    )
end

visualize(v::Dataset.AbstractDataPoint; kwargs...) = visualize(v.data; kwargs...)

visualize(v::AbstractCustomObject; kwargs...) =
    error("Define visualize function for custom type: $(typeof(v)).
            Use `import Flux3D.visualize` and define function
            `visualize(v::$(typeof(v)); kwargs...)`")
