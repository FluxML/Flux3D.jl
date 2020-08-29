using Flux3D, Zygote, Test, Statistics

include("utils.jl")

@testset "Flux3D" begin

    @info "Testing Representation..."
    @testset "Representation" begin
        include("rep.jl")
    end

    @info "Testing Conversions..."
    @testset "Conversions" begin
        include("conversions.jl")
    end

    @info "Testing Transforms..."
    @testset "Transforms" begin
        include("transforms/pcloud_func.jl")
        include("transforms/mesh_func.jl")
        include("transforms/transforms.jl")
    end

    @info "Testing Metrics..."
    @testset "Metrics" begin
        include("metrics.jl")
    end

    @info "Testing Models..."
    @testset "Models" begin
        include("models.jl")
    end

    @info "Testing Dataset..."
    @testset "Dataset" begin
        include("datasets/custom.jl")

        # FIXME unzip throws error on GPU CI due to weird reason
        if !Flux3D.use_cuda[]
            include("datasets/modelnet.jl")
        end
    end

    @info "Testing GPU support..."
    @testset "CUDA" begin
        if Flux3D.use_cuda[]
            using CUDA
            include("cuda/rep.jl")
            include("cuda/conversions.jl")
            include("cuda/transforms.jl")
            include("cuda/metrics.jl")
        else
            @warn "CUDA unavailable, not testing GPU support"
        end
    end

end # testset Flux3D
