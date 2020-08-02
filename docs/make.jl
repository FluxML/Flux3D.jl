using Flux3D
using Documenter

makedocs(;
    modules = [Flux3D],
    authors = "Nirmal P. Suthar <nirmalps@iitk.ac.in>",
    repo = "https://github.com/nirmal-suthar/Flux3D.jl/blob/{commit}{path}#L{line}",
    sitename = "Flux3D.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://nirmal-suthar.github.io/Flux3D.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/nirmal-suthar/Flux3D.jl.git")
