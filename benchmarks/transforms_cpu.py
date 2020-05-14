import torch, time
import kaolin as kal
import kaolin.transforms as T 

def run_benchmarks_cpu(arr, npoints, n_iters):

    for (t, name) in arr:
        print("*"*10, "Transforms {}".format(name), "*"*10)
        benchmark_time = []
        
        points = torch.randn(npoints, 3)
        p = kal.rep.PointCloud(points, device='cpu')

        # bug in kaolin (normalize doesn't accept PointCloud)
        if name == "NormalizePointCloud":
            p = points

        for i in range(n_iters):

            start_time = time.time()
            t(p)
            end_time = time.time()

            if i is 1: # Ignore first iteration
                continue
            benchmark_time.append(end_time - start_time)

        print("Benchmark Time : {}".format(min(benchmark_time)))

n_iters = 101
device = torch.device('cpu')

for npoints in [2**14, 2**16, 2**18, 2**20]:
    arr = [(T.ScalePointCloud(torch.Tensor([.5]).to(device=device)), "ScalePointCloud"),
       (T.RotatePointCloud(torch.randn(3,3).to(device)), "RotatePointCloud"),
       (T.RealignPointCloud(kal.rep.PointCloud(torch.randn(npoints,3), device=device)), "ReAlignPointCloud"),
       (T.NormalizePointCloud(), "NormalizePointCloud"),
       (T.Compose([T.ScalePointCloud(torch.Tensor([.5]).to(device)),
                  T.RotatePointCloud(torch.randn(3,3).to(device)),
                  T.RealignPointCloud(kal.rep.PointCloud(torch.randn(npoints,3), device=device)),
                  T.NormalizePointCloud()]), "Compose")]
    print("*"*10, "npoints = {}, device = {}".format(npoints, device), "*"*10)
    run_benchmarks_cpu(arr, npoints, n_iters, device=device)    
    print()

# Google Colab output [runtime: GPU, npoints: 16384] 

# ********** npoints = 16384, device = cpu **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 2.2411346435546875e-05
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 3.790855407714844e-05
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.0016744136810302734
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.0008723735809326172
# ********** Transforms Compose **********
# Benchmark Time : 0.001056671142578125

# ********** npoints = 65536, device = cpu **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 0.0003638267517089844
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 0.0001957416534423828
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.007279872894287109
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.0038008689880371094
# ********** Transforms Compose **********
# Benchmark Time : 0.0050737857818603516

# ********** npoints = 262144, device = cpu **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 0.0006532669067382812
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 0.0007166862487792969
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.027949810028076172
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.014463663101196289
# ********** Transforms Compose **********
# Benchmark Time : 0.0179598331451416

# ********** npoints = 1048576, device = cpu **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 0.001964569091796875
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 0.002869844436645508
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.11122250556945801
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.05746889114379883
# ********** Transforms Compose **********
# Benchmark Time : 0.07275104522705078