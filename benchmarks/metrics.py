import torch
import numpy as np
import kaolin as kal
import kaolin.transforms as T
import kaolin.rep.TriangleMesh as M
import kaolin.metrics.mesh as ops
import kaolin.rep.Mesh as mesh
import time, os

def setup_benchmark_record(names):
    benchmarks = {}
    for name in names:
        benchmarks[name] = []
    return benchmarks

def generate_trimesh(npoints, device='cpu'):
#     verts = torch.ones([npoints,3], dtype=torch.float32)
#     verts = verts.cumsum(dim=0) / npoints
#     faces = torch.randint(npoints, size=(npoints, 3), dtype=torch.int64)
# #     faces = torch.arange(npoints, dtype=torch.int64).repeat(3).view(npoints,3)
#     m = M.from_tensors(verts, faces)
    m = M.from_obj(npoints)
    m.to(device)
    return m

def cpu_time(t, args):
    start_time = time.time()
    t(*args)
    end_time = time.time()
    return (end_time - start_time)

def gpu_time(t, args):
    # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    t(*args) # Run some things here
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms/1000

def run_benchmarks_(benchmarks, x, benchmark_func, device, niters=51):
    for (transform, args, name) in x:
        btime = []
        for i in range(niters):
            trial = benchmark_func(transform, args)
            btime.append(trial)
        time = min(btime)*1000
        print("{}: {} ms".format(name, time))
        benchmarks[name].append(time)

ROT_MATRIX = torch.tensor([[1.0, 2.0, 3.0],
                           [0.2, 0.5, 0.9],
                           [3.0, 2.0, 1.0]])

# npoint_arr = 2 ** np.array([12])#, 14, 16, 18, 20])
mesh_arr = ["../teapot.obj", "../truck.obj"]


names = ["sample_points", "chamfer_distance", "edge_loss", "laplacian_loss"]

cpu_bm = setup_benchmark_record(names)
gpu_bm = setup_benchmark_record(names)

print("DEVICE: CPU")
device = "cpu"
for _npoints in mesh_arr:
    arr = [
#         (sample_points, [generate_trimesh(_npoints), _npoints], "sample_points"),
        (ops.chamfer_distance, [generate_trimesh(_npoints,device),
                                generate_trimesh(_npoints,device),_npoints],
         "chamfer_distance"),
        (ops.edge_length, [generate_trimesh(_npoints,device)], "edge_loss"),
        (mesh.compute_laplacian, [generate_trimesh(_npoints,device)], "laplacian_loss")
    ]
    print("Running benchmarks for npoints = {}".format(_npoints))
    run_benchmarks_(
        cpu_bm,
        arr,
        cpu_time,
        device
    )
    print()

if torch.cuda.is_available():
    print("CUDA is on. Running GPU Benchmarks")
    print("DEVICE: GPU")
    device = "cuda"
    for _npoints in npoint_arr:
        pcloud_arr = [(ops.chamfer_distance,
                       [generate_trimesh(_npoints, device), generate_trimesh(_npoints, device),
                       1.0,1.0,_npoints], "chamfer_distance"),
            (ops.edge_length, [generate_trimesh(_npoints)],
                       "edge_loss"),
            (ops.laplacian_loss, [generate_trimesh(_npoints),
                                  generate_trimesh(_npoints)], "laplacian_loss"),
            (T.SampleTriangleMesh(_npoints), [generate_trimesh(_npoints)], "sample_points")

        print("Running benchmarks for npoints = {}".format(_npoints))
        run_benchmarks_(
            gpu_bm,
            arr,
            _npoints,
            cpu_time,
            device
        )
        print()

def save_bm(fname, rep, cpu_benchmarks, gpu_benchmarks):
    with open(fname, 'a') as io:
        device = "cpu"
        for key, values in cpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("{} {} {} {} {} ms\n".format(rep, device, key, p, v))

        device = "gpu"
        for key, values in gpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("{} {} {} {} {} ms\n".format(rep, device, key, p, v))

fname = os.path.join(os.path.dirname(__file__), "bm_kaolin.txt")
os.remove(fname)
save_bm(fname, "PointCloud", cpu_bm_pcloud, gpu_bm_pcloud)
save_bm(fname, "TriMesh", cpu_bm_trimesh, gpu_bm_trimesh)
print("Benchmarks have been saved at {}".format(fname))
