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
    - `clims::Tuple`: `(min,max)` values to clamp `f` to before plotting. Defaults to `f`'s own
        range (i.e. no clamping).
    - `levels::Int`: Number of contour levels.
    - `xlim::Tuple`, `ylim::Tuple`: Axis limits, in the same shifted coordinates as `shift`. Default
        to the full extent of `f`.
    - `kv...`: Additional keyword arguments passed to `Plots.contourf`.
"""
function flood(f::AbstractArray;shift=(-.5,-.5),cfill=:RdBu_11,clims=(),levels=10, xlim=(0,size(f)[1]), ylim=(0,size(f)[2]),kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).-0.5.+shift[1],axes(f,2).-0.5.+shift[2],f'|>Array,
                   linewidth=0, levels=levels, color=cfill, clims = clims,
                   aspect_ratio=:equal, xlim=xlim, ylim=ylim; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim,dat,dat_plot;levels=[0],lines=:black,CIs=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    copyto!(dat,sim.flow.σ)
    restrict_plot!(dat_plot,dat,CIs)
    contour!(axes(dat_plot,1).-0.5,axes(dat_plot,2).-0.5,dat_plot';levels,lines)
end

restrict_plot!(dat_plot,dat,CIs) = (dat_plot .= value.(@view dat[CIs]); dat_plot)

function vorticity!(dat, sim)
    a = sim.flow.σ
    @WaterLily.inside a[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    copyto!(dat, a)
end

"""
    sim_gif!(sim;duration=1,step=0.1,verbose=true,CIs=inside(sim.flow.p),
                    remeasure=false,plotbody=false,f=vorticity!,fname=nothing,framerate=20,
                    udf=nothing,udf_kwargs=nothing,hidedecorations=false,kv...)

Make a gif of the simulation `sim`, stepping the flow forward and plotting `f(sim)` with `flood` at each frame.

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
    - `fname::String`: Path to save the gif. Defaults to a temporary file.
    - `framerate::Int`: Gif framerate.
    - `udf::Function`: User-defined function passed into `sim_step!`.
    - `udf_kwargs::Dict{Symbol}`: User-defined function keyword arguments passed into `sim_step!`. Needs to be a `Dict{Symbol}` or any
        `Pair{Symbol,Any}` iterator.
    - `hidedecorations::Bool`: Hide the axis ticks, labels, grid and colorbar, and shrink the plot margins to zero.
    - `kv...`: Additional keyword arguments passed to `flood`.
"""
function sim_gif!(sim;duration=1,step=0.1,verbose=true,CIs=inside(sim.flow.p),
                    remeasure=false,plotbody=false,f=vorticity!,fname=nothing,framerate=20,
                    udf=nothing,udf_kwargs=nothing,hidedecorations=false,kv...)
    !isnothing(udf) && !isnothing(udf_kwargs) && (@assert all(isa(kw, Pair{Symbol}) for kw in udf_kwargs) "udf_kwargs needs to contain Pair{Symbol,Any} elements, eg. Dict{Symbol,Any}.")
    isnothing(udf) && (udf_kwargs=[])
    dat = Array(sim.flow.σ)
    dat_plot = value.(dat[CIs])
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
    isnothing(fname) ? gif(anim;fps=framerate) : gif(anim,fname;fps=framerate)
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
    # @log p/c, nᵖ, r∞, r₁, (nᵇ) this is what is logged
    # get index of all time steps
    idx = findall(==(0.0),@views(predictor[1,:]))
    # plot inital L∞ and L₁ norms of residuals for the predictor step
    p1=plot(1:length(idx),predictor[2,idx],color=:1,ls=:dash,alpha=0.8,label="predictor initial r∞",yaxis=:log,size=(800,400),dpi=600,
            xlabel="Time step",ylabel="L∞-norm",title="Residuals",ylims=(1e-8,1e0),xlims=(0,length(idx)))
    p2=plot(1:length(idx),predictor[3,idx],color=:1,ls=:dash,alpha=0.8,label="predictor initial r₁",yaxis=:log,size=(800,400),dpi=600,
            xlabel="Time step",ylabel="L₁-norm",title="Residuals",ylims=(1e-8,1e0),xlims=(0,length(idx)))
    # plot final L∞ and L₁ norms of residuals for the predictor
    plot!(p1,1:length(idx),vcat(predictor[2,idx[2:end].-1],predictor[2,end]),color=:1,lw=2,alpha=0.8,label="predictor r∞")
    plot!(p2,1:length(idx),vcat(predictor[3,idx[2:end].-1],predictor[3,end]),color=:1,lw=2,alpha=0.8,label="predictor r₁")
    # plot the MG iterations for the predictor
    p3=plot(1:length(idx),clamp.(vcat(predictor[1,idx[2:end].-1],predictor[1,end]),√1/2,32),lw=2,alpha=0.8,label="predictor",size=(800,400),dpi=600,
            xlabel="Time step",ylabel="Iterations",title="MG Iterations",ylims=(√1/2,32),xlims=(0,length(idx)),yaxis=:log2)
    yticks!([√1/2,1,2,4,8,16,32],["0","1","2","4","8","16","32"])
    # get index of all time steps
    idx = findall(==(0.0),@views(corrector[1,:]))
    # plot inital L∞ and L₁ norms of residuals for the corrector step
    plot!(p1,1:length(idx),corrector[2,idx],color=:2,ls=:dash,alpha=0.8,label="corrector initial r∞",yaxis=:log)
    plot!(p2,1:length(idx),corrector[3,idx],color=:2,ls=:dash,alpha=0.8,label="corrector initial r₁",yaxis=:log)
    # plot final L∞ and L₁ norms of residuals for the corrector step
    plot!(p1,1:length(idx),vcat(corrector[2,idx[2:end].-1],corrector[2,end]),color=:2,lw=2,alpha=0.8,label="corrector r∞")
    plot!(p2,1:length(idx),vcat(corrector[3,idx[2:end].-1],corrector[3,end]),color=:2,lw=2,alpha=0.8,label="corrector r₁")
    # plot MG iterations of the corrector
    plot!(p3,1:length(idx),clamp.(vcat(corrector[1,idx[2:end].-1],corrector[1,end]),√1/2,32),lw=2,alpha=0.8,label="corrector")
    if size(predictor,1) == 3
        # plot all together
        plot(p1,p2,p3,layout=@layout [a b c])
    else # if the BiotSavart is used, we can plot the coupling iterations
        p4 = plot(1:length(idx),clamp.(vcat(predictor[4,idx[2:end].-1],predictor[4,end]),√1/2,32),lw=2,alpha=0.8,label="predictor",size=(800,400),dpi=600,
                  xlabel="Time step",ylabel="Iterations",title="Biot-Savart",ylims=(√1/2,32),xlims=(0,length(idx)),yaxis=:log2)
        yticks!([√1/2,1,2,4,8,16,32],["0","1","2","4","8","16","32"])
        plot!(p4,1:length(idx),clamp.(vcat(corrector[4,idx[2:end].-1],corrector[4,end]),√1/2,32),lw=2,alpha=0.8,label="corrector")
        # plot all together
        plot(p1,p2,p3,p4,layout=@layout [a b c d])
    end
end

end # module