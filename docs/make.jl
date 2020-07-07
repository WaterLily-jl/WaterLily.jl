using Documenter, WaterLily

makedocs(
    modules = [WaterLily],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Gabriel Weymouth",
    sitename = "WaterLily.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/gabrielweymouth/WaterLily.jl.git",
    push_preview = true
)
