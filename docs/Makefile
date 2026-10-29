SRCDIR:=$(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: liveserver

# Draft build (examples are not executed) served by the Vitepress dev server, with hot reload
liveserver:
	julia --project=${SRCDIR} ${SRCDIR}/liveserver.jl

# Full build: runs the examples and renders the static site into build/1
build:
	julia --project=${SRCDIR} --threads=auto ${SRCDIR}/make.jl

# Serve a finished full build
serve: build
	julia --project=${SRCDIR} -e 'import LiveServer; LiveServer.serve(dir = "${SRCDIR}/build/1")'

clean:
	julia ${SRCDIR}/clean.jl

.PHONY: default liveserver build serve clean
