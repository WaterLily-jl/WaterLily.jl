using Documenter, WaterLily

makedocs(
    modules = [WaterLily],
    repo="https://github.com/gabrielweymouth/WaterLily.jl",
    sitename="WaterLily",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https:/Zitzeronion.github.io/WaterLily.jl",
        assets=String[],
    authors = "Gabriel Weymouth",
    sitename = "WaterLily.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/Zitzeronion/WaterLily.jl.git",
    push_preview = true
)
