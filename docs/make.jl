using Documenter, WaterLily

makedocs(
    modules = [WaterLily],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://weymouth.github.io/WaterLily.jl/",
        assets=String[],
        mathengine = MathJax3()
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
    target = "build",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#" ],
)
