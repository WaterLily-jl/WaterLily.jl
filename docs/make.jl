using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

# Headless GR: write plots to file without opening a window
ENV["GKSwstype"] = "100"

using Documenter, DocumenterVitepress, WaterLily, Plots, StaticArrays
using JLD2, ReadVTK, WriteVTK
import CairoMakie # `import`, so that its exports do not clash with those of Plots

# Parts of the API live in package extensions, and only exist once the corresponding package
# is loaded. Load them here so that their docstrings are documented and checked too.
const EXTENSIONS = map(
    ext -> Base.get_extension(WaterLily, ext),
    [:WaterLilyPlotsExt, :WaterLilyMakieExt, :WaterLilyWriteVTKExt, :WaterLilyReadVTKExt,
     :WaterLilyJLD2Ext],
)

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

# Build documentation.
@timeit dto "makedocs" makedocs(
    # Without this, Documenter takes the current directory as the root when make.jl is not
    # run as a file (e.g. evaluated piece by piece in an editor). From the repository root
    # it then reads WaterLily's own `src/` as the docs source, and writes the site, and the
    # npm install that DocumenterVitepress runs next to it, into the repository root.
    root = @__DIR__,
    modules = [WaterLily, EXTENSIONS...],
    # The site is rendered by Vitepress rather than by Documenter's own HTML writer. Documenter
    # still does all of the work up to and including running the `@example` blocks; this writer
    # emits Markdown into `build/.documenter`, which Vitepress then turns into the static site in
    # `build/1`. `logo.png` and `favicon.ico` are picked up from `src/assets` automatically, and
    # the stylesheet lives in `src/.vitepress/theme/overrides.css` -- this writer has no `assets`.
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/WaterLily-jl/WaterLily.jl",
        devbranch = "main",
        devurl = "dev",
        description = "A differentiable and backend-agnostic Julia solver for incompressible " *
            "viscous flow around dynamic bodies",
        # In liveserver mode only the Markdown is written, and Vitepress' own dev server
        # (started by `liveserver.jl`) renders and hot-reloads it.
        build_vitepress = !liveserver,
    ),
    sitename = "WaterLily.jl",
    draft = liveserver,
    # In draft mode the `@example` blocks are not executed. A full build, local or on CI, is
    # strict.
    warnonly = liveserver ? [:cross_references] : false,
    pages = Any[
        "Home" => "index.md",
        "About" => "about.md",
        "Examples" => [
            "Examples overview" => "examples/index.md",
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
        ],
        "The wider ecosystem" => [
            "Ecosystem overview" => "ecosystem.md",
        ],
    ],
)

# Deploy built documentation. Vitepress bakes the base URL into the site, so every version
# alias (`stable`, `v1`, `dev`, ...) is a separate build; `DocumenterVitepress.deploydocs`
# pushes each of them in turn, which is why it takes no `versions` argument.
if !liveserver
    @timeit dto "deploydocs" DocumenterVitepress.deploydocs(
        repo = "github.com/WaterLily-jl/WaterLily.jl.git",
        devbranch = "main",
        target = "build",
        branch = "gh-pages",
        push_preview = true,
    )
end

print_timer(dto)
