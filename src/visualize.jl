export visualize

function visualize(v::PointCloud; kwargs...)
    # get!(kwargs, :markersize) do
    #     0.02*v.npoints/1024
    # end
    # get!(kwargs, :color) do
    #     :blue
    # end
    color = get(kwargs, :color, :blue)
    markersize = get(kwargs, :markersize, 0.02*v.npoints/1024)
    Makie.AbstractPlotting.meshscatter(v.points[:,1],v.points[:,2],v.points[:,3], color=color, markersize=markersize)
end

visualize(v::AbstractDataPoint; kwargs...) = visualize(v.data; kwargs...)

visualize(v::AbstractCustomObject; kwargs...) = error("Define visualize function for custom type: $(typeof(v)). 
                                                        Use `import Flux3D.visualize` and define function 
                                                        `visualize(v::$(typeof(v)); kwargs...)`")