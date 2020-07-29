import AbstractPlotting
import GeometryBasics
export visualize

"""
    visualize(pcloud::PointCloud; kwargs...)

Visualize PointCloud `pcloud` at `index`.

Dimension of points in PointCloud `pcloud` must be 3.

### Optional Arguments:
- color (Symbol)       - Color of the marker, default `:blue`
- markersize (Number)  - Size of the marker, default `0.02*npoints(pcloud)/1024`
"""
function visualize(v::PointCloud, index::Number=1; kwargs...)
    points = v[index]
    size(points,1)==3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :blue)
    get!(kwargs, :markersize, 40 / npoints(v))

    AbstractPlotting.meshscatter(v.points[:, 1], v.points[:, 2],v.points[:, 3];
                                 kwargs...)
end

"""
    visualize(m::TriMesh, index::Int=1; kwargs...)
    visualize(verts, faces; kwargs...)

Visualize mesh at `index` in TriMesh `m`.

### Optional Arguments:
"""

function visualize(m::GeometryBasics.Mesh; kwargs...) where{T,R}
    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :red)

    AbstractPlotting.mesh(GeometryBasics.normal_mesh(m); kwargs...)
end

visualize(m::TriMesh, index::Int=1; kwargs...) = visualize(GBMesh(m, index); kwargs...)

visualize(v::Dataset.AbstractDataPoint; kwargs...) =
    visualize(v.data; kwargs...)

visualize(v::AbstractCustomObject; kwargs...) =
    error("Define visualize function for custom type: $(typeof(v)).
            Use `import Flux3D.visualize` and define function
            `visualize(v::$(typeof(v)); kwargs...)`")
