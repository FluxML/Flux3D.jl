# Flux3D

[![Build Status](https://github.com/nirmal-suthar/Flux3D.jl/workflows/CI/badge.svg)](https://github.com/nirmal-suthar/Flux3D.jl/actions)
[![Coverage](https://codecov.io/gh/nirmal-suthar/Flux3D.jl/branch/master/graph/badge.svg?token=8kpPqDfChf)](https://codecov.io/gh/nirmal-suthar/Flux3D.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nirmal-suthar.github.io/Flux3D.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nirmal-suthar.github.io/Flux3D.jl/dev)

## CPU Benchmarks [Google Colab, runtime:GPU]

| transforms | Framework | npoints=2<sup>14</sup> | npoints=2<sup>16</sup> | npoints=2<sup>20</sup> |
|:--:|:--:|:--:|:--:|:--:|
|ScalePointCloud|Flux3D|__0.0032 ms__|__0.0128 ms__|__0.4952 ms__|
||Kaolin|0.0224 ms|0.3638 ms|1.9645 ms|
|RotatePointCloud|Flux3D|__0.0348 ms__|__0.1920 ms__|3.5250 ms|
||Kaolin|0.0379 ms|0.1957 ms|__2.8698 ms__|
|ReAlignPointCloud|Flux3D|__0.1710 ms__|__0.7320 ms__|__11.824 ms__|
||Kaolin|1.6744 ms|7.2798 ms|111.22 ms|
|NormalizePointCloud|Flux3D|__0.0813 ms__|__0.3930 ms__|__7.8250 ms__|
||Kaolin|0.8723 ms|3.8008 ms|57.468 ms|

## GPU Benchmarks [Google Colab, runtime:GPU]

| transforms | Framework | npoints=2<sup>14</sup> | npoints=2<sup>16</sup> | npoints=2<sup>20</sup> |
|:--:|:--:|:--:|:--:|:--:|
|ScalePointCloud|Flux3D|__0.0350 ms__|__0.0423 ms__|__0.1448 ms__|
||Kaolin|0.0918 ms|0.07312 ms|0.1634 ms|
|RotatePointCloud|Flux3D|__0.0236 ms__|__0.0313 ms__|__0.2227 ms__|
||Kaolin|0.0409 ms|0.0396 ms|0.3421 ms|
|ReAlignPointCloud|Flux3D|__0.7195 ms__|__0.7083 ms__|__1.0020 ms__|
||Kaolin|3.2031 ms|12.607 ms|330.80 ms|
|NormalizePointCloud|Flux3D|1.3030 ms|__1.4050 ms__|__1.6380 ms__|
||Kaolin|__0.9214 ms__|3.6641 ms|57.498 ms|
