using Documenter, WaterLily

makedocs(
    modules = [WaterLily],
    repo="https://github.com/gabrielweymouth/WaterLily.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https:/weymouth.github.io/WaterLily.jl/",
        assets=String[],
    ),
    authors = "Gabriel Weymouth",
    sitename = "WaterLily.jl",
    pages = ["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/weymouth/WaterLily.jl.git",
)
