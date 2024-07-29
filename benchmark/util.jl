using Plots, StatsPlots, LaTeXStrings, CategoricalArrays, Printf, ColorSchemes

iarg(arg, args) = occursin.(arg, args) |> findfirst
arg_value(arg, args) = split(args[iarg(arg, args)], "=")[end]
metaparse(x) = eval(Meta.parse(x))
parsepatterns(x) = replace(x,","=>("\",\""),"["=>("[\""),"]"=>("\"]"))

function parse_cla(args; cases=["tgv"], log2p=[(6,7)], max_steps=[100], ftype=[Float32], backend=Array)
    cases = !isnothing(iarg("cases", args)) ? arg_value("cases", args) |> metaparse : cases
    log2p = !isnothing(iarg("log2p", args)) ? arg_value("log2p", args) |> metaparse : log2p
    max_steps = !isnothing(iarg("max_steps", args)) ? arg_value("max_steps", args) |> metaparse : max_steps
    ftype = !isnothing(iarg("ftype", args)) ? arg_value("ftype", args) |> metaparse : ftype
    backend = !isnothing(iarg("backend", args)) ? arg_value("backend", args) |> x -> eval(Symbol(x)) : backend
    return cases, log2p, max_steps, ftype, backend
end

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            synchronize($b)
        end
    end |> esc
end

function add_to_suite!(suite, sim_function; p=(3,4,5), s=100, ft=Float32, backend=Array, bstr="CPU", remeasure=false)
    suite[bstr] = BenchmarkGroup([bstr])
    for n in p
        sim = sim_function(n, backend; T=ft)
        sim_step!(sim, typemax(ft); max_steps=5, verbose=false, remeasure=remeasure) # warm up
        suite[bstr][repr(n)] = BenchmarkGroup([repr(n)])
        KA_backend = get_backend(sim.flow.p)
        @add_benchmark sim_step!($sim, $typemax($ft); max_steps=$s, verbose=false, remeasure=$remeasure) $KA_backend suite[bstr][repr(n)] "sim_step!"
    end
end

waterlily_dir = get(ENV, "WATERLILY_ROOT", "")
git_hash = read(`git -C $waterlily_dir rev-parse --short HEAD`, String) |> x -> strip(x, '\n')
getf(str) = eval(Symbol(str))

backend_str = Dict(Array => "CPUx"*@sprintf("%.2d", Threads.nthreads()))
check_compiler(compiler,parse_str) = try occursin(parse_str, read(`$compiler --version`, String)) catch _ false end
_cuda = check_compiler("nvcc","release")
_rocm = check_compiler("hipcc","version")
_cuda && (using CUDA: CuArray; backend_str[CuArray] = "GPU-NVIDIA")
_rocm && (using AMDGPU: ROCArray; backend_str[ROCArray] = "GPU-AMD")
(_cuda || _rocm) && (using GPUArrays: allowscalar; allowscalar(false))

# Plotting utils
using Plots

fontsize = 14
speedup_fontsize = 14
Plots.default(
    fontfamily = "Computer Modern",
    linewidth = 1,
    framestyle = :box,
    grid = false,
    left_margin = Plots.Measures.Length(:mm, 24),
    right_margin = Plots.Measures.Length(:mm, 0),
    bottom_margin = Plots.Measures.Length(:mm, 5),
    top_margin = Plots.Measures.Length(:mm, 5),
    legendfontsize = fontsize,
    tickfontsize = fontsize,
    labelfontsize = fontsize,
)

# Fancy logarithmic scale ticks for plotting
# https://github.com/JuliaPlots/Plots.jl/issues/3318
"""
    get_tickslogscale(lims; skiplog=false)
Return a tuple (ticks, ticklabels) for the axis limit `lims`
where multiples of 10 are major ticks with label and minor ticks have no label
skiplog argument should be set to true if `lims` is already in log scale.
"""
function get_tickslogscale(lims::Tuple{T, T}; skiplog::Bool=false) where {T<:AbstractFloat}
    mags = if skiplog
        # if the limits are already in log scale
        floor.(lims)
    else
        floor.(log10.(lims))
    end
    rlims = if skiplog; 10 .^(lims) else lims end

    total_tickvalues = []
    total_ticknames = []

    rgs = range(mags..., step=1)
    for (i, m) in enumerate(rgs)
        if m >= 0
            tickvalues = range(Int(10^m), Int(10^(m+1)); step=Int(10^m))
            ticknames  = vcat([string(round(Int, 10^(m)))],
                              ["" for i in 2:9],
                              [string(round(Int, 10^(m+1)))])
        else
            tickvalues = range(10^m, 10^(m+1); step=10^m)
            ticknames  = vcat([string(10^(m))], ["" for i in 2:9], [string(10^(m+1))])
        end

        if i==1
            # lower bound
            indexlb = findlast(x->x<rlims[1], tickvalues)
            if isnothing(indexlb); indexlb=1 end
        else
            indexlb = 1
        end
        if i==length(rgs)
            # higher bound
            indexhb = findfirst(x->x>rlims[2], tickvalues)
            if isnothing(indexhb); indexhb=10 end
        else
            # do not take the last index if not the last magnitude
            indexhb = 9
        end

        total_tickvalues = vcat(total_tickvalues, tickvalues[indexlb:indexhb])
        total_ticknames = vcat(total_ticknames, ticknames[indexlb:indexhb])
    end
    return (total_tickvalues[1:end-1], total_ticknames[1:end-1])
end

"""
    fancylogscale!(p; forcex=false, forcey=false)
Transform the ticks to log scale for the axis with scale=:log10.
forcex and forcey can be set to true to force the transformation
if the variable is already expressed in log10 units.
"""
function fancylogscale!(p::Plots.Subplot; forcex::Bool=false, forcey::Bool=false)
    kwargs = Dict()
    for (ax, force, lims) in zip((:x, :y), (forcex, forcey), (xlims, ylims))
        axis = Symbol("$(ax)axis")
        ticks = Symbol("$(ax)ticks")

        if force || p.attr[axis][:scale] == :log10
            # Get limits of the plot and convert to Float
            ls = float.(lims(p))
            ts = if force
                (vals, labs) = get_tickslogscale(ls; skiplog=true)
                (log10.(vals), labs)
            else
                get_tickslogscale(ls)
            end
            kwargs[ticks] = ts
        end
    end

    if length(kwargs) > 0
        plot!(p; kwargs...)
    end
    p
end
fancylogscale!(p::Plots.Plot; kwargs...) = (fancylogscale!(p.subplots[1]; kwargs...); return p)
fancylogscale!(; kwargs...) = fancylogscale!(plot!(); kwargs...)

function Base.unique(ctg::CategoricalArray)
    l = levels(ctg)
    newctg = CategoricalArray(l)
    levels!(newctg, l)
end

function annotated_groupedbar(xx, yy, group; series_annotations="", bar_width=1.0, plot_kwargs...)
    gp = groupedbar(xx, yy, group=group, series_annotations="", bar_width=bar_width; plot_kwargs...)
    m = length(unique(group))       # number of items per group
    n = length(unique(xx))          # number of groups
    xt = (1:n) .- 0.5               # plot x-coordinate of groups' centers
    dx = bar_width/m                # each group occupies bar_width units along x
    # dy = diff([extrema(yy)...])[1]
    x2 = [xt[i] + (j - m/2 - 0.3)*dx for j in 1:m, i in 1:n][:]
    k = 1
    for i in 1:n, j in 1:m
        y0 = gp[1][2j][:y][i]*1.3# + 0.04*dy
        if isfinite(y0)
            annotate!(x2[(i-1)*m + j]*1.01, y0, text(series_annotations[k], :center, :black, speedup_fontsize))
            k += 1
        end
    end
    gp
end

# Find files utils
using Glob
function rdir(dir, patterns)
    results = String[]
    patterns = [Glob.FilenameMatch("*" * p * "*") for p in patterns]
    for (root, _, files) in walkdir(dir)
        fpaths = joinpath.(root, files)
        length(fpaths) == 0 && continue
        matches = [filter(x -> occursin(p, x), fpaths) for p in patterns]
        push!(results, vcat(matches...)...)
    end
    results
end

# Benchmark sizes
tests_dets = Dict(
    "tgv" => Dict("size" => (1, 1, 1), "title" => "TGV"),
    "sphere" => Dict("size" => (16, 6, 6), "title" => "Sphere"),
    "cylinder" => Dict("size" => (12, 6, 2), "title" => "Moving cylinder"),
    "donut" => Dict("size" => (2, 1, 1), "title" => "Donut"),
    "jelly" => Dict("size" => (1, 1, 4), "title" => "Jelly"),
)
