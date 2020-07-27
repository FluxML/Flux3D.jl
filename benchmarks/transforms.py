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

def generate_pcloud(npoints, device='cpu'):
    points = torch.ones([npoints,3], dtype=torch.float32)
    points = points.cumsum(dim=0)
    return kal.rep.PointCloud(points / npoints, device=device)

def generate_trimesh(npoints, device='cpu'):
    verts = torch.ones([npoints,3], dtype=torch.float32)
    verts = verts.cumsum(dim=0) / npoints
    faces = torch.arange(npoints*3, dtype=torch.int32).view(npoints,3)
    m = M.from_tensors(verts, faces)
    m.to(device)
    return m

def realign_point_cloud(npoints, device='cpu'):
    pc = generate_pcloud(npoints, device)
    rot = T.RotatePointCloud(-ROT_MATRIX.to(device))
    return rot(pc)

def realign_trimesh(npoints, device='cpu'):
    pc = generate_trimesh(npoints, device)
    rot = T.RotateMesh(-ROT_MATRIX.to(device))
    return rot(pc)

def cpu_time(t, p):
    start_time = time.time()
    t(p)
    end_time = time.time()
    return (end_time - start_time)

def gpu_time(t, p):
    # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Run some things here
    a = t(p)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms/1000

def run_benchmarks_(benchmarks, x, npoints, generate_func, benchmark_func, device, niters=51):
    for (transform, name) in x:
        btime = []
        for i in range(niters):
            pc = generate_func(npoints, device)
            # bug in kaolin (normalize doesn't accept PointCloud)
            if name == "NormalizePointCloud":
                pc = pc.points
            trial = benchmark_func(transform, pc)
            btime.append(trial)

        time = min(btime)*1000
        print("{}: {} ms".format(name, time))
        benchmarks[name].append(time)

ROT_MATRIX = torch.tensor([[1.0, 2.0, 3.0],
                           [0.2, 0.5, 0.9],
                           [3.0, 2.0, 1.0]])

npoint_arr = 2 ** np.array([12, 14, 16, 18, 20])

names = ["ScalePointCloud", "RotatePointCloud",
         "ReAlignPointCloud", "NormalizePointCloud", "Chain"]

cpu_bm_pcloud = setup_benchmark_record(names)
gpu_bm_pcloud = setup_benchmark_record(names)

names = ["ScaleTriMesh", "RotateTriMesh",
         "ReAlignTriMesh", "NormalizeTriMesh", "Chain"]

cpu_bm_trimesh = setup_benchmark_record(names)
gpu_bm_trimesh = setup_benchmark_record(names)

print("DEVICE: CPU")
device = "cpu"
for _npoints in npoint_arr:
    pcloud_arr = [(T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False), "ScalePointCloud"),
       (T.RotatePointCloud(ROT_MATRIX.to(device), inplace=False), "RotatePointCloud"),
       (T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False), "ReAlignPointCloud"),
       (T.NormalizePointCloud(inplace=False), "NormalizePointCloud"),
       (T.Compose([T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False),
                  T.RotatePointCloud(torch.randn(3,3).to(device), inplace=False),
                  T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False),
                  T.NormalizePointCloud(inplace=False)]), "Chain")
    ]

    trimesh_arr = [(T.ScaleMesh(.5, inplace=True), "ScaleTriMesh"),
       (T.RotateMesh(ROT_MATRIX.to(device), inplace=True), "RotateTriMesh"),
       (T.RealignMesh(realign_trimesh(_npoints, device).vertices), "ReAlignTriMesh"),
       (T.NormalizeMesh(inplace=True), "NormalizeTriMesh"),
       (T.Compose([T.ScaleMesh(.5, inplace=True),
              T.RotateMesh(ROT_MATRIX.to(device), inplace=True),
              T.RealignMesh(realign_trimesh(_npoints, device).vertices),
              T.NormalizeMesh(inplace=True)]), "Chain")
    ]
    print("Running benchmarks for npoints = {}".format(_npoints))
    run_benchmarks_(
        cpu_bm_pcloud,
        pcloud_arr,
        _npoints,
        generate_pcloud,
        cpu_time,
        device
    )
    run_benchmarks_(
        cpu_bm_trimesh,
        trimesh_arr,
        _npoints,
        generate_trimesh,
        cpu_time,
        device
    )
    print()

if torch.cuda.is_available():
    print("CUDA is on. Running GPU Benchmarks")
    print("DEVICE: GPU")
    device = "cuda"
    for _npoints in npoint_arr:
        pcloud_arr = [(T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False), "ScalePointCloud"),
           (T.RotatePointCloud(ROT_MATRIX.to(device), inplace=False), "RotatePointCloud"),
           (T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False), "ReAlignPointCloud"),
           (T.NormalizePointCloud(inplace=False), "NormalizePointCloud"),
           (T.Compose([T.ScalePointCloud(torch.Tensor([.5]).to(device), inplace=False),
                      T.RotatePointCloud(torch.randn(3,3).to(device), inplace=False),
                      T.RealignPointCloud(realign_point_cloud(_npoints, device), inplace=False),
                      T.NormalizePointCloud(inplace=False)]), "Chain")
        ]

        trimesh_arr = [(T.ScaleMesh(.5, inplace=True), "ScaleTriMesh"),
           (T.RotateMesh(ROT_MATRIX.to(device), inplace=True), "RotateTriMesh"),
           (T.RealignMesh(realign_trimesh(_npoints, device).vertices), "ReAlignTriMesh"),
           (T.NormalizeMesh(inplace=True), "NormalizeTriMesh"),
           (T.Compose([T.ScaleMesh(.5, inplace=True),
                  T.RotateMesh(ROT_MATRIX.to(device), inplace=True),
                  T.RealignMesh(realign_trimesh(_npoints, device).vertices),
                  T.NormalizeMesh(inplace=True)]), "Chain")
        ]
        print("Running benchmarks for npoints = {}".format(_npoints))
        run_benchmarks_(
            gpu_bm_pcloud,
            pcloud_arr,
            _npoints,
            generate_pcloud,
            gpu_time,
            device
        )
        run_benchmarks_(
            gpu_bm_trimesh,
            trimesh_arr,
            _npoints,
            generate_trimesh,
            gpu_time,
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
if os.path.exists(fname):
    os.remove(fname)
save_bm(fname, "PointCloud", cpu_bm_pcloud, gpu_bm_pcloud)
save_bm(fname, "TriMesh", cpu_bm_trimesh, gpu_bm_trimesh)
print("Benchmarks have been saved at {}".format(fname))
