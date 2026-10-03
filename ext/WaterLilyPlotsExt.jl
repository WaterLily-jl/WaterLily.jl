module WaterLilyPlotsExt

using Plots, WaterLily
using ForwardDiff: value
import Plots: mm
import WaterLily: flood,addbody,body_plot!,sim_gif!,plot_logger
gr()

"""
    flood(f;shift=(-.5,-.5),cfill=:RdBu_11,clims=(),levels=10,xlim=(0,size(f,1)),ylim=(0,size(f,2)),kv...)

Plot a filled contour plot of the 2D array `f`, which must be a CPU array (host memory).

Keyword arguments:
    - `shift::Tuple`: Offset of the plotted coordinates relative to the array indices, in units of
        cells. Defaults to `(-0.5,-0.5)` since `f` is assumed to live on cell edges (e.g. vorticity);
        pass `(0.,0.)` for cell-centered data (e.g. pressure).
    - `cfill`: Colormap passed to `Plots.contourf` as `color`.
    - `clims::Tuple`: `(min,max)` values to clamp `f` to before plotting. Defaults to (-5,5).
    - `levels::Int`: Number of contour levels.
    - `xlim::Tuple`, `ylim::Tuple`: Axis limits, in the same shifted coordinates as `shift`. Default
        to the full extent of `f`.
    - `kv...`: Additional keyword arguments passed to `Plots.contourf`.
"""
function flood(f::AbstractArray;shift=(-.5,-.5),cfill=:seismic,clims=(-5,5),levels=10, xlim=(0,size(f)[1]), ylim=(0,size(f)[2]),kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        minf,maxf = minimum(f),maximum(f)
        clims = ifelse(maxf-minf<0.001, (-1,1), (minf,maxf))
    end
    Plots.contourf(axes(f,1).-0.5.+shift[1],axes(f,2).-0.5.+shift[2],f'|>Array,
                   linewidth=0, levels=levels, color=cfill, clims = clims,
                   aspect_ratio=:equal, xlim=xlim, ylim=ylim; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
"""
    body_plot!(sim,dat,dat_plot;levels=[0],lines=:black,CIs=inside(sim.flow.p))

Non-allocating: overlay the body's zero level-set contour, reusing the `dat`/`dat_plot` buffers
preallocated by the caller (e.g. `sim_gif!`).
"""
function body_plot!(sim,dat,dat_plot;levels=[0],lines=:black,CIs=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    copyto!(dat,sim.flow.σ)
    restrict_plot!(dat_plot,dat,CIs)
    contour!(axes(dat_plot,1).-0.5,axes(dat_plot,2).-0.5,dat_plot';levels,lines)
end
"""
    body_plot!(sim;levels=[0],lines=:black,CIs=inside(sim.flow.p))

Allocating convenience method: overlay the body's zero level-set contour on the current plot.
Intended for one-off/manual plotting (see `flood`); allocates fresh buffers on every call,
unlike the non-allocating `body_plot!(sim,dat,dat_plot;kv...)` used internally by `sim_gif!`.
"""
function body_plot!(sim;levels=[0],lines=:black,CIs=inside(sim.flow.p))
    dat = Array(sim.flow.σ)
    ndims(dat)==3 && @assert any(==(1), size(CIs)) "3D CIs must include a singleton dimension (e.g. a cut plane) to reduce the data to a 2D slice for plotting, got size $(size(CIs))."
    dat_plot = dropdims(value.(dat[CIs]), dims=Tuple(findall(==(1), size(CIs))))
    body_plot!(sim,dat,dat_plot;levels,lines,CIs)
end

restrict_plot!(dat_plot,dat,CIs) = (dat_plot .= value.(@view dat[CIs]); dat_plot)

function vorticity!(dat, sim)
    a = sim.flow.σ
    @WaterLily.inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    copyto!(dat, a)
end

"""
    sim_gif!(sim;duration=1,step=0.1,verbose=true,CIs=inside(sim.flow.p),
                    remeasure=false,plotbody=false,f=vorticity!,video=nothing,framerate=20,
                    udf=nothing,udf_kwargs=nothing,hidedecorations=false,kv...)

Make a gif of 2D field of the simulation `sim`, stepping the flow forward and plotting `f(sim)` with `flood` at each frame.
Users can pass a function `f` used to post-process the flow field data and copy the scalar field into a CPU buffer array.
The default visualization function returns the z-vorticity scaled by `L/U`:
```julia
function vorticity!(dat, sim)
    a = sim.flow.σ
    @WaterLily.inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    copyto!(dat, a)
end
```

Keyword arguments:

    - `duration::Number`: Simulation duration (in convective time units) to animate.
    - `step::Number`: Time step between animation frames.
    - `verbose::Bool`: Print simulation information at each frame.
    - `CIs::CartesianIndices`: Region to plot, and to outline the body within if `plotbody=true`. Defaults to `inside(sim.flow.p)`.
    - `remeasure::Bool`: Update the body position at each step.
    - `plotbody::Bool`: Overlay the body's zero level-set contour.
    - `f::Function`: Visualization function with interface `f(dat, sim)`, transferring the plotted data
        (device-to-host) into the preallocated buffer `dat` (allocated once, full domain
        size, and reused every frame). Defaults to the z-vorticity scaled by `L/U`.
    - `video::String`: Path to save the animation. Saved as an mp4 if the path ends in `.mp4`, otherwise as a gif.
        Defaults to a temporary gif file.
    - `framerate::Int`: Gif framerate.
    - `udf::Function`: User-defined function passed into `sim_step!`.
    - `udf_kwargs::Dict{Symbol}`: User-defined function keyword arguments passed into `sim_step!`. Needs to be a `Dict{Symbol}` or any
        `Pair{Symbol,Any}` iterator.
    - `hidedecorations::Bool`: Hide the axis ticks, labels, grid and colorbar, and shrink the plot margins to zero.
    - `kv...`: Additional keyword arguments passed to `flood`.
"""
function sim_gif!(sim;duration=1,step=0.1,verbose=true,CIs=inside(sim.flow.p),
                    remeasure=false,plotbody=false,f=vorticity!,video=nothing,framerate=20,
                    udf=nothing,udf_kwargs=nothing,hidedecorations=false,kv...)
    !isnothing(udf) && !isnothing(udf_kwargs) && (@assert all(isa(kw, Pair{Symbol}) for kw in udf_kwargs) "udf_kwargs needs to contain Pair{Symbol,Any} elements, eg. Dict{Symbol,Any}.")
    isnothing(udf) && (udf_kwargs=[])
    dat = Array(sim.flow.σ)
    ndims(dat)==3 && @assert any(==(1), size(CIs)) "3D CIs must include a singleton dimension (e.g. a cut plane) to reduce the data to a 2D slice for plotting, got size $(size(CIs))."
    dat_plot = dropdims(value.(dat[CIs]), dims=Tuple(findall(==(1), size(CIs))))
    t₀ = round(WaterLily.sim_time(sim))
    anim = @time @animate for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure,udf,udf_kwargs...)
        f(dat,sim); restrict_plot!(dat_plot,dat,CIs)
        flood(dat_plot; kv...)
        plotbody && body_plot!(sim,dat,dat_plot;CIs)
        hidedecorations && Plots.plot!(showaxis=false,ticks=false,grid=false,colorbar=false,margin=0mm)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
    if isnothing(video)
        gif(anim;fps=framerate)
    elseif endswith(video,".mp4")
        mp4(anim,video;fps=framerate)
    else
        gif(anim,video;fps=framerate)
    end
end


"""
    plot_logger(fname="WaterLily.log")

Plot the residuals and MG iterations from the log file `fname`.
"""
function plot_logger(fname="WaterLily.log")
    predictor = []; corrector = []
    open(ifelse(fname[end-3:end]==".log",fname[1:end-4],fname)*".log","r") do f
        readline(f) # read first line and dump it
        which = "p"
        while ! eof(f)
            s = split(readline(f) , ",")
            which = s[1] != "" ? s[1] : which
            push!(which == "p" ? predictor : corrector, parse.(Float64,s[2:end]))
        end
    end
    predictor = reduce(hcat,predictor)
    corrector = reduce(hcat,corrector)
    # logged rows: nᵖ, r∞=L∞(p), r₁=L₁(p), ω, (nᵇ)  (row 1 is nᵖ, resets to 0 each time step)

    # per-row series across time steps: `initial` at each step start, `final` at each step end
    steps(M)     = findall(==(0.0), @views M[1,:])
    initial(M,r) = (idx=steps(M); M[r,idx])
    final(M,r)   = (idx=steps(M); vcat(M[r,idx[2:end].-1], M[r,end]))
    iters(M,r)   = clamp.(final(M,r), √1/2, 32) # clamp iteration counts onto the log2 axis
    np   = length(steps(predictor))
    opts = (size=(800,400), dpi=600, alpha=0.8, xlabel="Time step", xlims=(0,np))
    series = ((predictor,:1,"predictor"), (corrector,:2,"corrector"))
    iter_ticks!(pl) = yticks!(pl, [√1/2,1,2,4,8,16,32], ["0","1","2","4","8","16","32"])

    # residual panels: dashed = initial residual, solid = final residual
    function residual_panel(r, ylabel, title)
        pl = plot(; yaxis=:log, ylims=(1e-8,1e0), ylabel, title, opts...)
        for (M,c,name) in series
            plot!(pl, initial(M,r); color=c, ls=:dash, alpha=0.8, label="$name initial")
            plot!(pl, final(M,r);   color=c, lw=2,     alpha=0.8, label=name)
        end; pl
    end
    p1 = residual_panel(2, "L∞-norm", "L∞-norm of residuals")
    p2 = residual_panel(3, "L₁-norm", "L₁-norm of residuals")

    # MG iterations per time step
    p3 = plot(; yaxis=:log2, ylims=(√1/2,32), ylabel="Iterations", title="MG Iterations", opts...)
    for (M,c,name) in series
        plot!(p3, iters(M,1); color=c, lw=2, alpha=0.8, label=name)
    end
    iter_ticks!(p3)

    # fourth panel: relaxation factor ω, or (with BiotSavart) the coupling iterations
    if size(predictor,1) == 4
        p4 = plot(; ylims=(0,1.1), ylabel="ω", title="Relaxation factor ω", opts...)
        for (M,c,name) in series
            plot!(p4, final(M,4); color=c, lw=2, alpha=0.8, label=name)
        end
    else
        p4 = plot(; yaxis=:log2, ylims=(√1/2,32), ylabel="Iterations", title="Biot-Savart", opts...)
        for (M,c,name) in series
            plot!(p4, iters(M,5); color=c, lw=2, alpha=0.8, label=name)
        end
        iter_ticks!(p4)
    end

    plot(p1,p2,p3,p4,layout=@layout [a b c d])
end

end # module
