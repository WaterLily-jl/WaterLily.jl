#!/usr/bin/env julia

# Make sure docs environment is active and instantiated. `resolve` makes an existing Manifest
# pick up the `[sources]` entry in Project.toml, which points WaterLily at this checkout.
import Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

# Communicate with docs/make.jl that we are running in live mode. This puts Documenter in
# draft mode (the examples are not executed) and tells the Vitepress writer to emit only the
# Markdown, because the Vitepress dev server started below does the rendering itself.
push!(ARGS, "liveserver")

# Write the Markdown into docs/build/.documenter ...
include(joinpath(@__DIR__, "make.jl"))

# ... and hand it to Vitepress' own dev server, which serves the site and hot-reloads it
# whenever the Markdown changes. Rerun this script to pick up changes to docstrings or to
# the Literate sources in src/literate-examples and src/literate-gallery.
import DocumenterVitepress
DocumenterVitepress.dev_docs(joinpath(@__DIR__, "build"))
