SRCDIR:=$(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: liveserver

# Draft build (examples are not executed) served by the Vitepress dev server, with hot reload
liveserver:
	julia --project=${SRCDIR} ${SRCDIR}/liveserver.jl

# Install the docs environment. `resolve` makes an existing Manifest pick up the
# `[sources]` entry in Project.toml, which points WaterLily at this checkout
instantiate:
	julia --project=${SRCDIR} -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Full build: runs the examples and renders the static site into build/1
build: instantiate
	julia --project=${SRCDIR} --threads=auto ${SRCDIR}/make.jl

# Serve the last full build. Uses `vitepress preview` rather than a plain file server
# because the site is built with `cleanUrls`, so pages are linked without the `.html`
# suffix and a static server would 404 on a direct link or a reload.
preview:
	julia --project=${SRCDIR} -e 'import DocumenterVitepress; \
	  DocumenterVitepress.run_vitepress_command("${SRCDIR}/build", "preview")'

# Full build, then serve it
serve: build preview

clean:
	julia ${SRCDIR}/clean.jl

.PHONY: default liveserver instantiate build preview serve clean
