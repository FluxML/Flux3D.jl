# Benchmarks (Transforms and Metrics)

## Usage

### TriMesh and PointCloud transforms

For benchmarking Flux3D.jl transforms run `julia transforms.jl` and for benchmarking Kaolin transforms run `python transforms.py`. After running this each benchmarking a text file namely `bm_flux3d.txt` & `bm_kaolin.txt`, will be saved for use in plotting purpose.

### Metrics

For benchmarking Flux3D.jl metrics run `julia metrics.jl` and for benchmarking Kaolin metrics run `python metrics.py`. After running this each benchmarking a text file namely `bm_flux3d_metrics.txt` & `bm_kaolin_metrics.txt`, will be saved for use in plotting purpose.

### Plotting

After running all four benchmarking script, run the following.

```julia
julia plot.jl
```

Plots will saved in `/benchmarks/pics` folder.

## Note

* CUSPARSE is currently broken for Float32 and cusparse-matrix is taking lot greater time, so using Sparse for laplacian_loss even on gpu.
* Laplacian_loss in kaolin is slightly different from Flux3D, so for benchmarking in kaolin I am using a simple custom laplacian_loss which matched that of Flux3D.