# Flux3D

[![Build Status](https://github.com/nirmal-suthar/Flux3D.jl/workflows/CI/badge.svg)](https://github.com/nirmal-suthar/Flux3D.jl/actions)
[![Coverage](https://codecov.io/gh/nirmal-suthar/Flux3D.jl/branch/master/graph/badge.svg?token=8kpPqDfChf)](https://codecov.io/gh/nirmal-suthar/Flux3D.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nirmal-suthar.github.io/Flux3D.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nirmal-suthar.github.io/Flux3D.jl/dev)

| transforms | Framework | benchmark_time | 
|:--:|:--:|:--:|
|ScalePointCloud|Flux3D|__0.0000039 s__|
||Kaolin| 0.0000222 s|
|RotatePointCloud|Flux3D|0.0000409 s|
||Kaolin|__0.0000312 s__|
|ReAlignPointCloud|Flux3D|__0.0002318 s__|
||Kaolin|0.0016832 s|
|NormalizePointCloud|Flux3D|__0.0000715 s__|
||Kaolin|0.0008790 s|