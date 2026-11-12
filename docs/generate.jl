# Generate the examples with Literate.jl
import Literate

EXAMPLES_IN = joinpath(@__DIR__, "src", "literate-examples")
EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLES_OUT)

# Code gallery
GALLERY_IN = joinpath(@__DIR__, "src", "literate-gallery")
GALLERY_OUT = joinpath(@__DIR__, "src", "gallery")
mkpath(GALLERY_OUT)

# Run Literate on all examples and gallery programs
@timeit dto "Literate." for (IN, OUT) in [(EXAMPLES_IN, EXAMPLES_OUT), (GALLERY_IN, GALLERY_OUT)],
        program in readdir(IN; join = true)
    name = basename(program)
    if endswith(program, ".jl")
        if !liveserver
            script = @timeit dto "script()" @timeit dto name Literate.script(program, OUT)
            code = strip(read(script, String))
        else
            code = "<< no script output when building as draft >>"
        end

        # remove "hidden" lines which are not shown in the markdown
        code_clean = join(filter(x -> !endswith(x, "#hide"), split(code, r"\r?\n")), "\n")
        code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
        code_clean = strip(code_clean)

        mdpost(str) = replace(str, "@__CODE__" => code_clean)

        ## Gallery programs are shown but not run: they need packages outside this environment,
        ## and some are far too big for a documentation build. Plain ```julia fences instead of
        ## Documenter's ```@example means the code is rendered verbatim and never executed.
        codefence = OUT == GALLERY_OUT ? ("```julia" => "```") : ("```@example $(first(splitext(name)))" => "```")
        @timeit dto "markdown()" @timeit dto name begin
            Literate.markdown(program, OUT; postprocess = mdpost, codefence)
        end
        if !liveserver
            # Notebooks are shipped unexecuted: the simulations are too expensive to
            # run a second time, and the markdown pages already carry the output.
            @timeit dto "notebook()" @timeit dto name begin
                Literate.notebook(program, OUT, execute = false)
            end
        end
    elseif any(endswith.(program, (".png", ".jpg", ".gif")))
        cp(program, joinpath(OUT, name); force = true)
    else
        @warn "ignoring $program"
    end
end
