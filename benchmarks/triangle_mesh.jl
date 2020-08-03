using Flux3D, BenchmarkTools

# mesh = load_trimesh(joinpath(@__DIR__, "../test/meshes/teapot.obj"))
mesh = load_trimesh("./test/meshes/teapot.obj")

trail = @benchmark $get_edges_packed($mesh; refresh = true)
minimum(trail.times) * 1.0e-6

trail = @benchmark $get_faces_to_edges_packed($mesh; refresh = true)
minimum(trail.times) * 1.0e-6

trail = @benchmark $get_laplacian_packed($mesh; refresh = true)
minimum(trail.times) * 1.0e-6

trail = @benchmark $compute_verts_normals_packed($mesh)
minimum(trail.times) * 1.0e-6

trail = @benchmark $compute_faces_areas_packed($mesh)
minimum(trail.times) * 1.0e-6

trail = @benchmark $compute_faces_normals_packed($mesh)
minimum(trail.times) * 1.0e-6

trail = @benchmark $laplacian_loss($mesh)
minimum(trail.times) * 1.0e-6

trail = @benchmark $edge_loss($mesh)
minimum(trail.times) * 1.0e-6

trail = @benchmark $chamfer_distance($mesh, $mesh, 10000)
minimum(trail.times) * 1.0e-6

trail = @benchmark $sample_points($mesh, 10000)
minimum(trail.times) * 1.0e-6
