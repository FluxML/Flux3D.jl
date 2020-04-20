using Flux3D
using Test

include("utils.jl")

@testset "Flux3D" begin

    @testset "Representation" begin
        include("rep.jl")
    end

    @testset "Model" begin
        include("models.jl")
    end

    @testset "Dataset" begin
        include("dataset.jl")
    end

end # testset Flux3D