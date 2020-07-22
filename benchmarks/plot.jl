using Gadfly, DataFrames, Cairo

function read_benchmarks(fname, framework)
    isfile(fname) || error("given file location $(fname) is invalid")
    data = DataFrame(
        device = String[],
        transforms = String[],
        npoints = Int[],
        time_ms = Float64[],
    )
    for line in eachline(fname)
        raw = split(line)
        push!(data,(raw[1], raw[2], parse(Int, raw[3]), parse(Float64, raw[4])))
    end
    data[!, :framework] .= framework
    return data
end

function plot_benchmarks(bm)
    # using Theme(background_color = colorant"white")
    # as arg in plot to force white background
    p = plot(
        bm,
        xgroup = "transforms",
        ygroup = "device",
        color = "framework",
        x = "npoints",
        y = "time_ms",
        Theme(background_color = colorant"white"),
        Geom.subplot_grid(Geom.point, Geom.line, Scale.x_log2, Scale.y_log10),
    )
    return p
end

bm_flux3d = read_benchmarks(joinpath(@__DIR__, "bm_flux3d.txt"), "Flux3D.jl")
bm_kaolin = read_benchmarks(joinpath(@__DIR__, "bm_kaolin.txt"), "Kaolin")
bm = vcat(bm_flux3d, bm_kaolin)

p = plot_benchmarks(bm)

draw(PNG(joinpath(@__DIR__, "bm_plot.png"), 40cm, 20cm), p)
