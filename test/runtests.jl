using Flux3D
using Test, Statistics

include("utils.jl")

@testset "Flux3D" begin

    @info "Testing Representation..."
    @testset "Representation" begin
        include("rep.jl")
    end

    @info "Testing Transforms..."
    @testset "Transforms" begin
        include("transforms/pcloud_func.jl")
        include("transforms/transforms.jl")
    end

    @info "Testing Models..."
    @testset "Models" begin
        include("models.jl")
    end

    @info "Testing Dataset..."
    @testset "Dataset" begin
        include("dataset.jl")
    end

    @info "Testing GPU support..."
    @testset "CUDA" begin
        if Flux3D.use_cuda[]
        include("cuda/rep.jl")
        include("cuda/transforms.jl")
        else
        @warn "CUDA unavailable, not testing GPU support"
        end
    end

end # testset Flux3D