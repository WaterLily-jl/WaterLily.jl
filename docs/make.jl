using Documenter, WaterLily

## find all image files in assets/ dir and copy in docs/src/assets/

recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

image_files = []

for pattern in [r"\.gif", r"\.jpg", r"\.png"]
    global image_files = vcat(image_files, recursive_find(joinpath(@__DIR__, "../assets/"), pattern))
end
@show image_files
@show joinpath(@__DIR__, "src/assets/"*basename.(image_files[1]))
for file in image_files
    cp(file, joinpath(@__DIR__, "src/assets/"*basename.(file)), force=true)
end


## now ready to make the docs

makedocs(
    modules = [WaterLily],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://WaterLily-jl.github.io/WaterLily.jl/",
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
    repo = "github.com/WaterLily-jl/WaterLily.jl.git",
    target = "build",
    branch = "gh-pages",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#" ],
)
