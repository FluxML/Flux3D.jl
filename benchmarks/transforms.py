import torch, time
import kaolin as kal
import kaolin.transforms as T 

def run_benchmarks(arr, npoints, n_iters):

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

npoints = 1024
n_iters = 101
arr = [(T.ScalePointCloud(torch.Tensor([.5])), "ScalePointCloud"),
       (T.RotatePointCloud(torch.randn(3,3)), "RotatePointCloud"),
       (T.RealignPointCloud(kal.rep.PointCloud(torch.randn(npoints,3))), "ReAlignPointCloud"),
       (T.NormalizePointCloud(), "NormalizePointCloud"),
       (T.Compose([T.ScalePointCloud(torch.Tensor([.5])),
                  T.RotatePointCloud(torch.randn(3,3)),
                  T.RealignPointCloud(kal.rep.PointCloud(torch.randn(npoints,3))),
                  T.NormalizePointCloud()]), "Compose")]

run_benchmarks(arr, npoints, n_iters)

# Google Colab output [runtime: GPU, npoints: 16384] 

# ********** Transforms ScalePointCloud **********
# Benchmark Time : 2.2172927856445312e-05
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 3.123283386230469e-05
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.0016832351684570312
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.0008790493011474609
# ********** Transforms Compose **********
# Benchmark Time : 0.0010504722595214844