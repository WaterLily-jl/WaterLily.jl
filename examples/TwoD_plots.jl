using Plots
Plots.GRBackend()

function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contour(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0,linecolor=:black,
        fill = (true,palette(cfill)),
        clims = clims, aspect_ratio=:equal; kv...)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)

function sim_gif!(sim;duration=1,step=0.1,verbose=true)
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    gr(show=false)
    @time @gif for tᵢ in t
        sim_step!(sim,tᵢ;verbose)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ,shift=(-0.5,-0.5),clims=(-5,5))
    end
    return
end

function vectors(f::Array)
    x = axes(f,1)' .* ones(size(f,2))
    y = ones(size(f,2))' .* axes(f,2)
    u,v = f[:,:,1]',f[:,:,2]'
    Plots.quiver(x[:],y[:],quiver=(u[:],v[:]),
                aspect_ratio=:equal)
end
