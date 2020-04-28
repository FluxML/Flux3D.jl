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
        include("transforms/pcloud.jl")
        include("transforms/transforms.jl")
    end

    @info "Testing Models..."
    @testset "Models" begin
        include("models.jl")
    end

    # @info "Testing Dataset"   
    # @testset "Dataset" begin      #TODO: uncomment this part, when done with ci setup.
    #     include("dataset.jl") 
    # end

end # testset Flux3D