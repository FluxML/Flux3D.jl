import Makie.AbstractPlotting.meshscatter

export visualize

"""
    visualize(pcloud::PointCloud; kwargs...)

Visualize PointCloud `pcloud` at `index`.

Dimension of points in PointCloud `pcloud` must be 3.

### Optional Arguments:
- color (Symbol)        - Color of the marker, default `:blue`
- markersize (Number)   - Size of the marker, default `0.02*npoints(pcloud)/1024`  
"""
function visualize(v::PointCloud, index::Number=1; kwargs...)
    points = v[index]    
    size(points,1)==3 || error("dimension of points in PointCloud must be 3.")

    kwargs = convert(Dict{Symbol, Any}, kwargs)
    get!(kwargs, :color, :blue)
    get!(kwargs, :markersize, 0.02*npoints(v)/1024)

    meshscatter(points[:,1],points[:,2],points[:,3]; kwargs...)
end

visualize(v::Dataset.AbstractDataPoint; kwargs...) = visualize(v.data; kwargs...)

visualize(v::AbstractCustomObject; kwargs...) = error("Define visualize function for custom type: $(typeof(v)). 
                                                        Use `import Flux3D.visualize` and define function 
                                                        `visualize(v::$(typeof(v)); kwargs...)`")