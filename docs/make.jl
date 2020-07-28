using Flux3D
using Documenter

makedocs(;
    modules=[Flux3D],
    doctest = false,
    authors="Nirmal P. Suthar <nirmalps@iitk.ac.in>",
    repo="https://github.com/nirmal-suthar/Flux3D.jl/blob/{commit}{path}#L{line}",
    sitename="Flux3D.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nirmal-suthar.github.io/Flux3D.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials"         => [
            "PointNet classication"        => "tutorials/pointnet.md",
            "DGCNN classication"           => "tutorials/dgcnn.md",
            "Supervised 3D reconstruction" => "tutorials/fit_mesh.md",
        ],
        "3D Structure"      => [
            "PointCloud"                   => "rep/pointcloud.md",
            "TriMesh"                      => "rep/trimesh.md",
        ],
        "Datasets"          => [
            "ModelNet"                     => "datasets/modelnet.md",
            "Utilities"                    => "datasets/utils.md"
        ],
        "API Documentation" => [
            "Transforms"                   => "api/transforms.md",
            "Metrics"                      => "api/metrics.md",
            "Visualization"                => "api/visualize.md",
            # "3D Models"     => "api/models.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/nirmal-suthar/Flux3D.jl.git",
)
