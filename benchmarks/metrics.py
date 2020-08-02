import torch
import numpy as np
import kaolin as kal
import kaolin.transforms as T
import kaolin.rep.TriangleMesh as M
import kaolin.metrics.mesh as ops
from kaolin.metrics.point import chamfer_distance
import kaolin.rep.Mesh as mesh
import time, os

def setup_benchmark_record(names):
    benchmarks = {}
    for name in names:
        benchmarks[name] = []
    return benchmarks

def generate_pcloud(npoints, device='cpu'):
    verts = torch.ones([npoints,3], dtype=torch.float32, requires_grad=True)
    verts = verts.cumsum(dim=0) / npoints
    return verts.to(device)

def generate_trimesh(npoints, device='cpu'):
    verts = torch.ones([npoints,3], dtype=torch.float32, requires_grad=True)
    verts = verts.cumsum(dim=0) / npoints
    faces = torch.arange(npoints, dtype=torch.int64).repeat(3).view(npoints,3)
    m = M.from_tensors(verts, faces)
    m.to(device)
    return m

def laplacian_loss(m):
    adj_sparse = m.compute_adjacency_matrix_sparse()
    neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1)
    loss = torch.sparse.mm(adj_sparse, m.vertices)
    loss = torch.norm(loss, dim=1)
    return torch.mean(loss)

def cpu_time(t, args, grad_tensor):
    f_start_time = time.time()
    result = t(*args)
    f_end_time = time.time()

    b_start_time = time.time()
    result.backward(retain_graph=True)
    b_end_time = time.time()

    f_time = f_end_time - f_start_time
    b_time = b_end_time - b_start_time
    return (f_time, b_time)

def gpu_time(t, args, grad_tensor):
    # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = t(*args) # Run some things here
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    f_time = start_event.elapsed_time(end_event)/1000

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result.backward(retain_graph=True) # Run some things here
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    b_time = start_event.elapsed_time(end_event)/1000

    return (f_time, b_time)

def run_benchmarks_(benchmarks, x, benchmark_func, device, niters=51):
    for (func, args, name) in x:
        if name=="chamfer_distance" and device=='cpu':
            continue
        btime_f = []
        btime_b = []
        for i in range(niters):
            grad_tensor = torch.ones(1).to(device)
            trial_f,trial_b = benchmark_func(func, args, grad_tensor)
            btime_f.append(trial_f)
            btime_b.append(trial_b)
        time_f = min(btime_f)*1000
        time_b = min(btime_b)*1000
        time_t = time_f + time_b
        print("{} (forward): {} ms".format(name, time_f))
        print("{} (back): {} ms".format(name, time_b))
        print("{} (total): {} ms".format(name, time_t))
        benchmarks[name].append([time_f,time_b,time_t])

ROT_MATRIX = torch.tensor([[1.0, 2.0, 3.0],
                           [0.2, 0.5, 0.9],
                           [3.0, 2.0, 1.0]])

npoint_arr = 2 ** np.array([6, 8, 10, 12, 14])

names = [
    "chamfer_distance",
    "edge_loss",
    "laplacian_loss"
]

cpu_bm = setup_benchmark_record(names)
gpu_bm = setup_benchmark_record(names)

print("DEVICE: CPU")
device = "cpu"
for _npoints in npoint_arr:
    arr = [
        #(sample_points, [generate_trimesh(_npoints), _npoints], "sample_points"),
        # (chamfer_distance,
        #  [generate_pcloud(_npoints,device),generate_pcloud(_npoints,device)],
        #  "chamfer_distance"),
        (ops.edge_length, [generate_trimesh(_npoints,device)], "edge_loss"),
        (laplacian_loss, [generate_trimesh(_npoints,device)], "laplacian_loss")
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
        arr = [
            #(sample_points, [generate_trimesh(_npoints), _npoints], "sample_points"),
            (chamfer_distance,
             [generate_pcloud(_npoints,device),generate_pcloud(_npoints,device)],
             "chamfer_distance"),
            (ops.edge_length, [generate_trimesh(_npoints,device)], "edge_loss"),
            (laplacian_loss, [generate_trimesh(_npoints,device)], "laplacian_loss")
        ]
        print("Running benchmarks for npoints = {}".format(_npoints))
        run_benchmarks_(
            gpu_bm,
            arr,
            gpu_time,
            device
        )

def save_bm(fname, rep, cpu_benchmarks, gpu_benchmarks):
    with open(fname, 'w') as io:
        device = "cpu"
        for key, values in cpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("Kaolin(Forward) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[0]))
                io.write("Kaolin(Backward) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[1]))
                io.write("Kaolin(Total) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[2]))

        device = "gpu"
        for key, values in gpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("Kaolin(Forward) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[0]))
                io.write("Kaolin(Backward) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[1]))
                io.write("Kaolin(Total) {} {} {} {} {} ms\n"
                         .format(rep, device, key, p, v[2]))

fname = os.path.join(os.path.dirname(__file__), "bm_kaolin_metrics.txt")
save_bm(fname, "Metrics", cpu_bm, gpu_bm)
print("Benchmarks have been saved at {}".format(fname))
