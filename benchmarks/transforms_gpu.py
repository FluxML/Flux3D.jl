import torch, time
import kaolin as kal
import kaolin.transforms as T 

def run_benchmarks_gpu(arr, npoints, n_iters, device=torch.device('cuda')):

    for (t, name) in arr:
        print("*"*10, "Transforms {}".format(name), "*"*10)
        benchmark_time = []
        
        points = torch.randn(npoints, 3)
        p = kal.rep.PointCloud(points, device=device)

        # bug in kaolin (normalize doesn't accept PointCloud)
        if name == "NormalizePointCloud":
            p = points

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

        print("Benchmark Time : {}".format(min(benchmark_time)))

n_iters = 101
device = torch.device('cuda')

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
    run_benchmarks_gpu(arr, npoints, n_iters, device=device)    
    print()

# Google Colab output [runtime: GPU] 

# ********** npoints = 16384, device = cuda **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 9.180799871683121e-05
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 4.095999896526337e-05
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.0032031359672546385
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.0009214720129966736
# ********** Transforms Compose **********
# Benchmark Time : 0.0033438079357147217

# ********** npoints = 65536, device = cuda **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 7.311999797821045e-05
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 3.9583999663591384e-05
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.012607616424560547
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.003664128065109253
# ********** Transforms Compose **********
# Benchmark Time : 0.013127679824829102

# ********** npoints = 262144, device = cuda **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 8.65280032157898e-05
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 0.00010063999891281128
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.05852726364135742
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.014281056404113769
# ********** Transforms Compose **********
# Benchmark Time : 0.06796083068847657

# ********** npoints = 1048576, device = cuda **********
# ********** Transforms ScalePointCloud **********
# Benchmark Time : 0.00016339200735092163
# ********** Transforms RotatePointCloud **********
# Benchmark Time : 0.0003421120047569275
# ********** Transforms ReAlignPointCloud **********
# Benchmark Time : 0.3308052368164062
# ********** Transforms NormalizePointCloud **********
# Benchmark Time : 0.05749808120727539
# ********** Transforms Compose **********
# Benchmark Time : 0.3319744873046875