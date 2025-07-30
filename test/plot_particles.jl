using ImageMorphology
import GLMakie
# push!(LOAD_PATH, "../Pathlines.jl/src/" ) # For now import local version of Pathlines
using Pathlines


function sim_gif_particles!(
    sim::Simulation;
    t_i=0.01, duration=2., Δt=0.05, N_particles=2^14, life_particles=100,
    scale=1.0, minsize=0.1, width=1,
    plotbody=true, 
    verbose=true,
    save_path="picture_sim_app/output/particleplot.gif",
    mem=Array,
)

    N=Int(N_particles)
    life=UInt(life_particles)
    t_f = duration

    p = Particles(N,sim.flow.σ;mem,life)
    v = ParticleViz(p, Δt, ; scale=scale, minsize=minsize, width=width)

    fig = GLMakie.Figure(; backgroundcolor = :gray30)
    ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
    GLMakie.scatter!(ax, v.opos; color = :white, markersize = v.omag, rotation = v.odir, marker = GLMakie.Circle)
    GLMakie.hidedecorations!(ax)

    if plotbody
        # Plot the body region - solid color overlay on top of particles
        μ₀ = Array(sim.body.μ₀)
        body_mask = μ₀ .< 0.5  # Solid pixels (true for solid)
        
        # Create a solid color overlay - only show body pixels, everything else transparent
        GLMakie.image!(ax, body_mask; colormap=[:transparent, "#990000"])
    end

    # display(fig)


    # Record video
    @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
        while sim_time(sim)<t
            WaterLily.mom_step!(sim.flow,sim.pois)
            update!(p,sim)
        end
        notify!(v,p,sim.flow.Δt[end-1])
        verbose && println("tU/L=", round(t, digits=4),
                ", Δt=", round(sim.flow.Δt[end], digits=3))
    end
end