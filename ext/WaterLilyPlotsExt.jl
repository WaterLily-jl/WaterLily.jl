module WaterLilyPlotsExt

using Plots, WaterLily
import WaterLily: flood,addbody,body_plot!,sim_gif!,plot_logger
gr()

"""
    flood(f)

Plot a filled contour plot of the 2D array `f`. The keyword arguments are passed to `Plots.contourf`.
"""
function flood(f::AbstractArray;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f'|>Array,
                   linewidth=0, levels=levels, color=cfill, clims = clims,
                   aspect_ratio=:equal; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(sim.flow.σ[R]'|>Array;levels,lines)
end

"""
    sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)

Make a gif of the simulation `sim` for `duration` seconds with `step` time steps. The keyword arguments are passed to `flood` and `body_plot!`.
"""
function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure,kv...)
        @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R]; kv...)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
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
    # logged rows: nᵖ, L∞(p), r₂, ω, (nᵇ)  (row 1 is nᵖ, resets to 0 each time step)

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
        end
        pl
    end
    p1 = residual_panel(2, "L∞-norm", "L∞-norm of residuals")
    p2 = residual_panel(3, "L₂-norm", "L₂-norm of residuals")

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