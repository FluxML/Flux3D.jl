import torch
import numpy as np
import kaolin as kal
import kaolin.transforms as T
import time, os

def setup_benchmark_record(names):
    benchmarks = {}
    for name in names:
        benchmarks[name] = []
    return benchmarks

def generate_point_cloud(npoints, device='cpu'):
    points = torch.ones([npoints,3], dtype=torch.float32)
    points = points.cumsum(dim=0)
    return kal.rep.PointCloud(points / npoints, device=device)

def realign_point_cloud(npoints, device='cpu'):
    pc = generate_point_cloud(npoints, device)
    rot = T.RotatePointCloud(-ROT_MATRIX.to(device))
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
        t(p)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)

        if i is 1: # Ignore first iteration
            continue
        benchmark_time.append(elapsed_time_ms/1000)
    return benchmark_time

def run_benchmarks_(benchmarks, x, npoints, benchmark_func, device):
    for (transform, name) in x:
        pc = generate_point_cloud(npoints, device)
        # bug in kaolin (normalize doesn't accept PointCloud)
        if name == "NormalizePointCloud":
            pc = pc.points
        trial = benchmark_func(transform, pc)
        time = min(trial) * 1.0e3 # converting second to millisecond
        print("{}: {} ms".format(name, time))
        benchmarks[name].append(time)

ROT_MATRIX = torch.tensor([[1.0, 2.0, 3.0],
                           [0.2, 0.5, 0.9],
                           [3.0, 2.0, 1.0]])

npoint_arr = 2 ** np.array([12, 14, 16, 18, 20])

names = ["ScalePointCloud", "RotatePointCloud",
         "RealignPointCloud", "NormalizePointCloud", "Chain"]

print("DEVICE: CPU")
device = "cpu"
cpu_benchmarks = setup_benchmark_record(names)

for _npoints in npoint_arr:
    arr = [(T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False), "ScalePointCloud"),
       (T.RotatePointCloud(ROT_MATRIX.to(device), inplace=False), "RotatePointCloud"),
       (T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False), "RealignPointCloud"),
       (T.NormalizePointCloud(inplace=False), "NormalizePointCloud"),
       (T.Compose([T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False),
                  T.RotatePointCloud(torch.randn(3,3).to(device), inplace=False),
                  T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False),
                  T.NormalizePointCloud(inplace=False)]), "Chain")]
    print("Running benchmarks for npoints = {}".format(_npoints))
    run_benchmarks_(cpu_benchmarks, arr, _npoints, cpu_time, device)
    print()

gpu_benchmarks = setup_benchmark_record(names)
if torch.cuda.is_available():
    print("CUDA is on. Running GPU Benchmarks")
    print("DEVICE: GPU")
    device = "cuda"
    for _npoints in npoint_arr:
        arr = [(T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False), "ScalePointCloud"),
           (T.RotatePointCloud(ROT_MATRIX.to(device), inplace=False), "RotatePointCloud"),
           (T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False), "RealignPointCloud"),
           (T.NormalizePointCloud(inplace=False), "NormalizePointCloud"),
           (T.Compose([T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False),
                      T.RotatePointCloud(torch.randn(3,3).to(device), inplace=False),
                      T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False),
                      T.NormalizePointCloud(inplace=False)]), "Chain")]
        print("Running benchmarks for npoints = {}".format(_npoints))
        run_benchmarks_(gpu_benchmarks, arr, _npoints, gpu_time, device)
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

fname = os.path.join(os.path.dirname(__file__), "bm_kaolin.txt")
save_bm(fname, cpu_benchmarks, gpu_benchmarks)
print("Benchmarks have been saved at {}".format(fname))
