using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

# Headless GR: write plots to file without opening a window
ENV["GKSwstype"] = "100"

using Documenter, DocumenterVitepress, WaterLily, Plots, StaticArrays
using JLD2, ReadVTK, WriteVTK

# Parts of the API live in package extensions, and only exist once the corresponding package
# is loaded. Load them here so that their docstrings are documented and checked too.
const EXTENSIONS = map(
    ext -> Base.get_extension(WaterLily, ext),
    [:WaterLilyPlotsExt, :WaterLilyWriteVTKExt, :WaterLilyReadVTKExt, :WaterLilyJLD2Ext],
)

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# DocumenterVitepress 0.3.9 renders a `LocalLink` -- a link to a file in the docs that is not
# itself a page, such as the `.jl` script and `.ipynb` notebook Literate links at the top of
# every example -- using the path Documenter stored, which is relative to the build root rather
# than to the page. On a page one directory down, `TwoD_Circle.jl` therefore comes out as
# `examples/TwoD_Circle.jl`, which resolves to `examples/examples/TwoD_Circle.jl`. The writer's
# `LocalImage` method gets this right, so mirror it here through the documented `render`
# extension point. Remove this once the upstream method does the same.
function DocumenterVitepress.render(
        io::IO, mime::MIME"text/plain", node::Documenter.MarkdownAST.Node,
        link::Documenter.LocalLink, page, doc; kwargs...,
    )
    path = if isempty(link.fragment)
        relpath(joinpath(doc.user.build, link.path), dirname(page.build))
    else
        replace(relpath(page.build, dirname(page.build)), ".md" => "") * "#" * link.fragment
    end
    path = replace(path, "\\" => "/")
    print(io, "[")
    DocumenterVitepress.render(io, mime, node, node.children, page, doc; kwargs...)
    print(io, "]($(replace(path, " " => "%20")))")
end

# DocumenterVitepress copies its own `package.json` into `docs/` for the Vitepress step and
# deletes it again afterwards. That file comes out of the read-only package directory, so it is
# read-only too, and a build interrupted during the Vitepress step leaves it behind: the next
# build then fails with "Permission denied" when it tries to update it. Clear out an unwritable
# leftover before starting.
let package_json = joinpath(@__DIR__, "package.json")
    if isfile(package_json)
        writable = try
            open(io -> true, package_json, "a")
        catch
            false
        end
        writable || (@info "Removing read-only docs/package.json left by an interrupted build";
                     rm(package_json; force = true))
    end
end

# Copy the images referenced from the README (which is spliced into the WaterLily
# docstring, and therefore into the front page) next to the generated pages
include("copy_assets.jl")

# Generate the examples
include("generate.jl")

# Build documentation.
@timeit dto "makedocs" makedocs(
    modules = [WaterLily, EXTENSIONS...],
    # The site is rendered by Vitepress rather than by Documenter's own HTML writer. Documenter
    # still does all of the work up to and including running the `@example` blocks; this writer
    # emits Markdown into `build/.documenter`, which Vitepress then turns into the static site in
    # `build/1`. `logo.png` and `favicon.ico` are picked up from `src/assets` automatically, and
    # the stylesheet lives in `src/.vitepress/theme/overrides.css` -- this writer has no `assets`.
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/WaterLily-jl/WaterLily.jl",
        devbranch = "master",
        devurl = "dev",
        description = "A differentiable and backend-agnostic Julia solver for incompressible " *
            "viscous flow around dynamic bodies",
        # In liveserver mode only the Markdown is written, and Vitepress' own dev server
        # (started by `liveserver.jl`) renders and hot-reloads it.
        build_vitepress = !liveserver,
    ),
    sitename = "WaterLily.jl",
    draft = liveserver,
    # In draft mode the examples are not executed and the scripts/notebooks are not written,
    # so the links to them (and to the figures they produce) cannot resolve locally.
    warnonly = is_ci ? false : [:cross_references],
    pages = Any[
        "Home" => "index.md",
        "Examples" => [
            "Examples overview" => "examples/index.md",
            "examples/TwoD_Circle.md",
            "examples/TwoD_Square.md",
            "examples/TwoD_MultipleBodies.md",
            "examples/TwoD_SlowStartCircle.md",
            "examples/TwoD_OscillatingFlow.md",
            "examples/TwoD_Hover.md",
            "examples/ThreeD_TaylorGreenVortex.md",
            "examples/ThreeD_Sphere.md",
        ],
        "API reference" => [
            "Reference overview" => "reference/index.md",
            "reference/simulation.md",
            "reference/bodies.md",
            "reference/flow.md",
            "reference/poisson.md",
            "reference/metrics.md",
            "reference/grid.md",
            "reference/io.md",
            "reference/plotting.md",
            "reference/utilities.md",
        ],
        "Code gallery" => [
            "Code gallery overview" => "gallery/index.md",
            "gallery/ParametricBodies_FlexiblePlate.md",
        ],
        "The wider ecosystem" => [
            "Ecosystem overview" => "ecosystem.md",
        ],
    ],
)

# Vitepress serves only Markdown routes and the contents of `public/`, so the `.jl` scripts and
# `.ipynb` notebooks that Literate writes next to each generated page are not part of the
# rendered site. Copy them in afterwards, so that the download links at the top of every example
# and gallery page resolve. `bases.txt` lists one base URL per site Vitepress built, each in its
# own numbered folder.
const MD_DIR = joinpath(@__DIR__, "build", ".documenter")
const BASES_FILE = joinpath(@__DIR__, "build", "bases.txt")

if !liveserver && isfile(BASES_FILE)
    @timeit dto "copy scripts and notebooks" for (i, _) in enumerate(eachline(BASES_FILE)),
            (root, _, files) in walkdir(MD_DIR),
            file in files
        endswith(file, ".jl") || endswith(file, ".ipynb") || continue
        # the Literate sources themselves are not part of the site
        contains(root, "literate-") && continue
        src = joinpath(root, file)
        dst = joinpath(@__DIR__, "build", string(i), relpath(src, MD_DIR))
        mkpath(dirname(dst))
        cp(src, dst; force = true)
    end
end

# Deploy built documentation. Vitepress bakes the base URL into the site, so every version
# alias (`stable`, `v1`, `dev`, ...) is a separate build; `DocumenterVitepress.deploydocs`
# pushes each of them in turn, which is why it takes no `versions` argument.
if !liveserver
    @timeit dto "deploydocs" DocumenterVitepress.deploydocs(
        repo = "github.com/WaterLily-jl/WaterLily.jl.git",
        devbranch = "master",
        target = "build",
        branch = "gh-pages",
        push_preview = true,
    )
end

print_timer(dto)
