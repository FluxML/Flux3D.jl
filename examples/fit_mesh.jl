using Flux3D, Statistics, FileIO, Zygote, Flux

mkpath(joinpath(@__DIR__, "assets"))
mkpath(joinpath(@__DIR__, "results"))

download(
    "https://github.com/nirmalsuthar/public_files/raw/master/dolphin.obj",
    joinpath(@__DIR__, "assets/dolphin.obj"),
)
download(
    "https://github.com/nirmalsuthar/public_files/raw/master/sphere.obj",
    joinpath(@__DIR__, "assets/sphere.obj"),
)

dolphin = load_trimesh(joinpath(@__DIR__, "assets/dolphin.obj"))
src = load_trimesh(joinpath(@__DIR__, "assets/sphere.obj"))

tgt = deepcopy(dolphin)
tgt = Flux3D.normalize!(tgt)

save(joinpath(@__DIR__, "results", "target.png"), visualize(tgt))
save(joinpath(@__DIR__, "results", "source.png"), visualize(src))

function loss_dolphin(x::Array, src::TriMesh, tgt::TriMesh)
    src = Flux3D.offset(src, x)
    loss1 = chamfer_distance(src, tgt, 5000)#; w1=1., w2=0.2)
    loss2 = laplacian_loss(src)
    loss3 = edge_loss(src)
    return (loss1) + (0.1 * loss2) + (loss3)
end

lr = 1.0
opt = Flux.Optimise.Momentum(lr, 0.9)

function customtrain(_offset)
    θ = Zygote.Params([_offset])
    for itr = 1:2000
        gs = gradient(θ) do
            loss_dolphin(_offset, src, tgt)
        end
        Flux.update!(opt, _offset, gs[_offset])
        if (itr % 100 == 0)
            save(
                joinpath(@__DIR__, "results", "src_$(itr).png"),
                visualize(Flux3D.offset(src, off)),
            )
        end
    end
    return _offset
end

@info("Training...")
_offset = zeros(Float32, size(get_verts_packed(src))...)
_offset = customtrain(_offset)

final_mesh = Flux3D.offset(src, _offset)
final_mesh = Flux3D.realign!(final_mesh, dolphin)

save(joinpath(@__DIR__, "results", "final.png"), visualize(final_mesh))
save_trimesh(joinpath(@__DIR__, "results", "final_mesh.obj"), final_mesh)
