import torch
import numpy as np
import kaolin as kal
import kaolin.transforms as T
import kaolin.rep.TriangleMesh as M
import time, os

def setup_benchmark_record(names):
    benchmarks = {}
    for name in names:
        benchmarks[name] = []
    return benchmarks

def realign_trimesh(mesh, device='cpu'):
    pc = M.from_obj(mesh)
    pc.to(device)
    rot = T.RotateMesh(-ROT_MATRIX.to(device))
    return rot(pc)

def cpu_time(t, p, n_iters=101):
    benchmark_time = []
    for i in range(n_iters):
        start_time = time.time()
        t(p)
        end_time = time.time()

        if i is 1: # Ignore first iteration
            continue
        benchmark_time.append(end_time - start_time)
    return benchmark_time

def gpu_time(t, p, n_iters=101):
    benchmark_time = []
    for i in range(n_iters):
        # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Run some things here
        a = t(p)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)

        if i is 1: # Ignore first iteration
            continue
        benchmark_time.append(elapsed_time_ms/1000)
    return benchmark_time

def run_benchmarks_(benchmarks, x, mesh, benchmark_func, device):
    for (transform, name) in x:
        pc = M.from_obj(mesh)
        pc.to(device)
        trial = benchmark_func(transform, pc)
        time = min(trial) * 1.0e3
        print("{}: {} ms".format(name, time))
        benchmarks[name].append(time)

ROT_MATRIX = torch.tensor([[1.0, 2.0, 3.0],
                           [0.2, 0.5, 0.9],
                           [3.0, 2.0, 1.0]])

mesh_arr = ["../../teapot.obj", "../../truck.obj"]

names = ["ScaleTriMesh", "RotateTriMesh",
         "RealignTriMesh", "NormalizeTriMesh", "Chain"]

print("DEVICE: CPU")
device = "cpu"
cpu_benchmarks = setup_benchmark_record(names)

print("DEVICE: CPU")
inplace = True
device = "cpu"
cpu_benchmarks = setup_benchmark_record(names)
for mesh in mesh_arr:
    arr = [(T.ScaleMesh(.5, inplace=inplace), "ScaleTriMesh"),
       (T.RotateMesh(ROT_MATRIX.to(device), inplace=inplace), "RotateTriMesh"),
       (T.RealignMesh(realign_trimesh(mesh, device).vertices), "RealignTriMesh"),
       (T.NormalizeMesh(inplace=inplace), "NormalizeTriMesh"),
       (T.Compose([T.ScaleMesh(.5, inplace=inplace),
              T.RotateMesh(ROT_MATRIX.to(device), inplace=inplace),
              T.RealignMesh(realign_trimesh(mesh, device).vertices),
              T.NormalizeMesh(inplace=inplace)]), "Chain")]
#     print("Running benchmarks for npoints = {}".format(_npoints))
    run_benchmarks_(cpu_benchmarks, arr, mesh, cpu_time, device)
    print()

gpu_benchmarks = setup_benchmark_record(names)
if torch.cuda.is_available():
    print("CUDA is on. Running GPU Benchmarks")
    print("DEVICE: GPU")
    device = "cuda"
    inplace = True
    for mesh in mesh_arr:
        arr = [(T.ScaleMesh(.5, inplace=inplace), "ScaleTriMesh"),
           (T.RotateMesh(ROT_MATRIX.to(device), inplace=inplace), "RotateTriMesh"),
           (T.RealignMesh(realign_trimesh(mesh, device).vertices), "RealignTriMesh"),
           (T.NormalizeMesh(inplace=inplace), "NormalizeTriMesh"),
           (T.Compose([T.ScaleMesh(.5, inplace=inplace),
                  T.RotateMesh(ROT_MATRIX.to(device), inplace=inplace),
                  T.RealignMesh(realign_trimesh(mesh, device).vertices),
                  T.NormalizeMesh(inplace=inplace)]), "Chain")]
    #     print("Running benchmarks for npoints = {}".format(_npoints))
        run_benchmarks_(gpu_benchmarks, arr, mesh, cpu_time, device)
        print()

def save_bm(fname, cpu_benchmarks, gpu_benchmarks):
    with open(fname, 'w') as io:
        device = "cpu"
        for key, values in cpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("{} {} {} {} ms\n".format(device, key, p, v))

        device = "gpu"
        for key, values in gpu_benchmarks.items():
            for p,v in zip(npoint_arr, values):
                io.write("{} {} {} {} ms\n".format(device, key, p, v))

npoints_arr = map(x->load_trimesh(x).V, mesh_arr)
fname = os.path.join(os.path.dirname(__file__), "bm_kaolin.txt")
save_bm(fname, cpu_benchmarks, gpu_benchmarks)
print("Benchmarks have been saved at {}".format(fname))
