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


function create_particle_gif_from_data!(
    sim_data;
    scale=1.0, minsize=0.1, width=1,
    plotbody=true,
    save_path="picture_sim_app/output/particleplot.gif",
    verbose=true
)
    n_frames = length(sim_data.positions)
    verbose && println("Creating particle GIF from $n_frames frames of data")
    
    # Setup figure
    fig = GLMakie.Figure(; backgroundcolor = :gray30)
    ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
    GLMakie.hidedecorations!(ax)
    
    # Create ParticleViz object to maintain proper smoothing and state
    # We'll use the average delta time for initialization
    avg_dt = sum(sim_data.delta_times) / length(sim_data.delta_times)
    
    # Create a dummy Particles object for ParticleViz initialization
    # We need to provide a proper flow field, so we'll create a minimal one
    N = length(sim_data.positions[1])
    
    # Create a minimal flow field structure for Particles initialization
    # Since we're only using this for visualization, we can use a simple approach
    domain_size = size(sim_data.body_mask)
    dummy_σ = zeros(domain_size..., 2)  # Simple 2D velocity field
    
    dummy_p = Particles(N, dummy_σ; mem=Array, life=UInt(100))
    v = ParticleViz(dummy_p, avg_dt; scale=scale, minsize=minsize, width=width)
    
    # Add scatter plot using ParticleViz observables
    GLMakie.scatter!(ax, v.opos; color = :white, markersize = v.omag, rotation = v.odir, marker = GLMakie.Circle)
    
    if plotbody
        body_mask = sim_data.body_mask .< 0.5
        GLMakie.image!(ax, body_mask; colormap=[:transparent, "#990000"])
    end
    
    # Record GIF
    @time GLMakie.record(fig, save_path, sim_data.time_points) do t
        # Find frame index for this time point
        frame_idx = findfirst(x -> x == t, sim_data.time_points)
        
        # Update the dummy particles object with current frame data
        # Ensure positions are 3D vectors (add z=0 if needed)
        current_pos = sim_data.positions[frame_idx]
        current_pos_prev = sim_data.positions_prev[frame_idx]
        
        # Convert 2D positions to 3D by adding z=0 if necessary
        if length(current_pos[1]) == 2
            pos_3d = [SVector(p[1], p[2], 0.0) for p in current_pos]
            pos_prev_3d = [SVector(p[1], p[2], 0.0) for p in current_pos_prev]
        else
            pos_3d = current_pos
            pos_prev_3d = current_pos_prev
        end
        
        dummy_p.position .= pos_3d
        dummy_p.position⁰ .= pos_prev_3d
        
        # Use ParticleViz notify! method for proper smoothing
        notify!(v, dummy_p, sim_data.delta_times[frame_idx])
        
        verbose && println("tU/L=", round(t, digits=4), ", Δt=", round(sim_data.delta_times[frame_idx], digits=3))
    end
end