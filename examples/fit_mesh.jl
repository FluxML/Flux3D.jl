# # Supervised 3D Mesh Reconstruction
#
# ## Problem Description:
# We are given an initial source shape (sphere in this case) and we want to deform
# this source shape to fit target shape (dolphin in this case). For this
# demonstration, we will be using Triangle Mesh for the representation of source
# and target shape.
#
# Triangle Mesh has two main components, vertices and faces. Deformation of source
# shape to fit target shape can be achieved by offsetting source's vertices to fit
# target surface. Also, the number of vertices and faces is not equal in source
# and target shape.
#
# !!!note
#     For visualization purpose we will require to install AbstractPlotting and
#     compatible backend (GLMakie or WGLMakie). To install it simply run
#     `] add AbstractPlotting GLMakie` in the julia prompt.

using Flux3D, Zygote, Flux, FileIO, Statistics, Plots
using AbstractPlotting, GLMakie

Flux3D.AbstractPlotting.inline!(true)
Flux3D.AbstractPlotting.set_theme!(show_axis = false)

# ## Downloading obj file of sphere and dolphin

download("https://github.com/nirmalsuthar/public_files/raw/master/dolphin.obj",
         "dolphin.obj")
download("https://github.com/nirmalsuthar/public_files/raw/master/sphere.obj",
         "sphere.obj")

# ## Loading Triangle Mesh
# Triangle Mesh is handled by TriMesh in Flux3D. TriMesh also supports batched
# format, namely padded, packed and list, which allow us to use fast batched
# operations. We can load TriMesh with load_trimesh function
# (supports `obj`, `stl`, `ply`, `off` and `2DM`)

dolphin = load_trimesh("dolphin.obj")
src = load_trimesh("sphere.obj")

# ## Preprocessing data
# Preprocessing tgt (dolphin), such that its mean is zero and also scale it
# according to the bounding box of src (sphere), So that src can converge at
# greater speed.

tgt = deepcopy(dolphin)
verts = get_verts_packed(tgt)
center = mean(verts, dims=2)
verts = verts .- center
scale = maximum(abs.(verts))
verts = verts ./ scale
tgt._verts_packed = verts

# ## Visualizing TriMesh
# We will use `visualize` function for visualizing TriMesh. This function uses
# **Makie** for plotting. In fact, we can also visualize PointCloud using this
# function, which makes this function handy dealing with different 3D format.

Flux3D.AbstractPlotting.vbox(visualize(src), visualize(tgt))

# ```@raw html
# <p align="center">
#     <img width=480 height=270 src="../../src/assets/fitmesh_initial.png">
# </p>
# ```

# ## Defining loss objective
# Starting from the src mesh, we will deform src mesh by offsetting its vertices
# (by offset array), such that new deformed mesh is close to target mesh.
# Therefore, our loss function will optimize the offset array. We will be using
# the following metrics to define loss objective:
#
# * `chamfer_distance` - the distance between the deformed mesh and target mesh, which is calculated by taking randomly 5000 points from the surface of each mesh and calculating chamfer_distance between these two pointcloud.
# * `laplacian_loss` - also known as Laplacian smoothing will act as a regularizer.
# * `edge_loss` - this will minimize edges length in deformed mesh, also act as a regularizer.

function loss_dolphin(x::AbstractArray, src::TriMesh, tgt::TriMesh)
    src = Flux3D.offset(src, x)
    loss1 = chamfer_distance(src, tgt, 5000)
    loss2 = laplacian_loss(src)
    loss3 = edge_loss(src)
    return loss1 + 0.1*loss2 + loss3
end
# ## Defining learning rate and optimizer

lr = 1.0
opt = Flux.Optimise.Momentum(lr, 0.9)

# ## Using GPU for fast training [**Optional**]
# We can convert the TriMesh structure to GPU or CPU using`gpu` and `cpu`
# function which is exactly the same syntax as Flux.

tgt = tgt |> gpu
src = src |> gpu
_offset = zeros(Float32, size(get_verts_packed(src))...) |> gpu

# ## Optimizing the offset array
# We first initialize offset array as zeros, hence deformed mesh is equivalent to
# src mesh (sphere). Next, we calculate loss using this offset array and we
# compute derivatives wrt. offset array and finally optimize the array.

@info("Training...")
θ = Zygote.Params([_offset])
for itr in 1:2001
    gs = gradient(θ) do
        loss_dolphin(_offset, src, tgt)
    end
    Flux.update!(opt, _offset, gs[_offset])
    if (itr%10 == 1)
        loss = loss_dolphin(_offset, src, tgt)
        @show itr, loss
        save("src_$(itr).png", visualize(Flux3D.offset(src, _offset)))
    end
end

anim = @animate for i ∈ 1:8
    Plots.plot(load("src_$(1+250*(i-1)).png"), showaxis=false)
end
gif(anim, "src_deform.gif", fps = 2)

# ```@raw html
# <p align="center">

#     <img width=256 height=256 src="../../src/assets/fitmesh_anim.gif">
# </p>
# ```
# ## Postprocessing the predicted mesh
# We create a new TriMesh by offsetting src by final offset array and scale up
# the final_mesh by the same scaling factor we scale down tgt, such that
# final_mesh has similar bounding box as dolphin mesh.

final_mesh = Flux3D.offset(src, _offset)
final_mesh = Flux3D.scale!(final_mesh, scale)

# ## Saving the final_mesh
# Flux3D provide IO function `save_trimesh`  to save TriMesh
# (supports `obj`, `stl`, `ply`, `off` and `2DM`)

save_trimesh("results/final_mesh.off", final_mesh)
save("results/final_mesh.png", visualize(final_mesh))
Flux3D.AbstractPlotting.vbox(visualize(final_mesh), visualize(dolphin))

# ```@raw html
# <p align="center">
#     <img width=480 height=270 src="../../src/assets/fitmesh_final.png">
# </p>
# ```

# ## Finally..
# * Look into the other examples in `examples/`
# * Read more the TriMesh in TriMesh section in documentation and
# Metrics/Transforms section for manipulating TriMesh and computing standard
# metrics.
