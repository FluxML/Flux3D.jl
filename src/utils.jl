import Flux: gpu, cpu
import CuArrays: cu

export gpu

gpu(x::PointCloud) = use_cuda[] ? PointCloud(cu(x.points), cu(x.normals)) : x
gpu(t::ScalePointCloud) = t
gpu(t::RotatePointCloud) = use_cuda[] ? RotatePointCloud(cu(t.rotmat), t.inplace) : t
gpu(t::NormalizePointCloud) = t
gpu(t::ReAlignPointCloud) = use_cuda[] ? ReAlignPointCloud(gpu(t.target_min), gpu(t.target_max), t.inplace) : t

# using CuArrays
# function bench_t(t, a)
#     CuArrays.@sync begin
#         t(a)
#     end
# end

# @btime bench_t($t, $a)