import Makie: AbstractPlotting
import GeometryBasics: normal_mesh
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

function visualize(m::TriMesh; kwargs...)
    gb_m = GBMesh(m)

    kwargs = convert(Dict{Symbol,Any}, kwargs)
    get!(kwargs, :color, :red)

    AbstractPlotting.mesh(normal_mesh(gb_m); kwargs...)
end

visualize(v::Dataset.AbstractDataPoint; kwargs...) =
    visualize(v.data; kwargs...)

visualize(v::AbstractCustomObject; kwargs...) =
    error("Define visualize function for custom type: $(typeof(v)).
            Use `import Flux3D.visualize` and define function
            `visualize(v::$(typeof(v)); kwargs...)`")
