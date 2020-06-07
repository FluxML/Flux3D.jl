using Flux3D, Statistics, FileIO, Zygote

mkpath(joinpath(@__DIR__,"assets"))         
mkpath(joinpath(@__DIR__,"img"))

download("https://github.com/nirmalsuthar/public_files/raw/master/dolphin.obj", 
         joinpath(@__DIR__,"assets/dolphin.obj"))
download("https://github.com/nirmalsuthar/public_files/raw/master/sphere.obj", 
         joinpath(@__DIR__,"assets/sphere.obj"))
         
tgt = load_trimesh(joinpath(@__DIR__,"assets/dolphin.obj"))
src = load_trimesh(joinpath(@__DIR__,"assets/sphere.obj"))

center = mean(tgt.vertices, dims=1)
tgt.vertices = tgt.vertices .- center
scale = maximum(abs.(tgt.vertices))
tgt.vertices = tgt.vertices ./ scale

save(joinpath(@__DIR__, "img", "target.png"), visualize(tgt))
save(joinpath(@__DIR__, "img", "source.png"), visualize(src))

function distChamfer(a,b)
    x = Float32.(a)
    y = Float32.(b)
    xx = sum(x.^2, dims=2)
    yy = sum(y.^2, dims=2)
    zz = x * transpose(y)
    rx = reshape(xx, 1,:)
    ry = reshape(yy, 1,:)
    P = (transpose(reshape(xx,1,:)) .+ reshape(yy,1,:)) .- (2 .* zz)
    return minimum(P, dims=2), minimum(P, dims=1)
end

function chamfer_loss(src, tgt)
    d1,d2 = distChamfer(src, tgt)
    return sum(d1)
end

offset_mesh(m::TriMesh, a::Array) = TriMesh(m.vertices + a, m.faces)

function loss_dolphin(x::Array, src::TriMesh, tgt::TriMesh)
    src = offset_mesh(src, x)
    sample_src = sample_points(src, 5000)
    sample_tgt = sample_points(tgt, 5000)
    loss1 = chamfer_loss(sample_src, sample_tgt)
    loss2 = laplacian_loss(src)
    loss3 = edge_loss(src)
    return loss1 + (0.1*loss2) + (0.05*loss3)
end

lr = 0.04

function customtrain(off)
    for itr in 1:50
        gs = gradient(x->loss_dolphin(x, src, tgt), off)[1]
        off = off - (lr .* gs)
        if (itr%3 == 0)
            save(joinpath(@__DIR__, "img","new_s_$(itr).png"), visualize(offset_mesh(src, off)))
        end
    end
    return off
end

@info("Training...")
off = zeros(Float32, size(src.vertices)...)
off = customtrain(off)

src.vertices = ((src.vertices+off) .* scale) .+ center
save(joinpath(@__DIR__, "img","final.png"), visualize(src))

# TODO: Save final obj, when saving function is implemented