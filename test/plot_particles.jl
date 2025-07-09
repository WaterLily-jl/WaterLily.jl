using ImageMorphology
import GLMakie
push!(LOAD_PATH, "../Pathlines.jl/src/" ) # For now import local version of Pathlines
using Pathlines


function sim_gif_particles!(
    sim::Simulation;
    t_i=0.01, duration=2., Δt=0.05, N_particles=2^14, life_particles=100,
    scale=1.0, minsize=0.1, width=1,
    plotbody=true, save_path="picture_sim_app/output/particleplot.mp4"
)

    N=Int(N_particles)
    life=UInt(life_particles)
    t_f = duration

    p = Particles(N,sim.flow.σ;mem,life)
    v = ParticleViz(p, Δt, ; scale=scale, minsize=minsize, width=width)


    fig = GLMakie.Figure(; backgroundcolor = :gray30)
    ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
    GLMakie.scatter!(ax, v.opos; color = :white, markersize = v.omag, rotation = v.odir, marker = GLMakie.Circle)

    # Plot the body region filled with black
    μ₀ = airfoil_pixel_body.μ₀
    body_mask = μ₀ .== 0.0  # Original solid pixels

    # For larger holes, use a different approach with label_components
    # First invert the mask so solid is false (0) and fluid is true (1)
    fluid_mask = .!body_mask

    # Label all connected components in the fluid
    labels = ImageMorphology.label_components(fluid_mask)

    # The largest component is the outside fluid region
    # All other components are holes inside the body
    component_sizes = zeros(Int, maximum(labels))
    for i in 1:length(labels)
        if labels[i] > 0
            component_sizes[labels[i]] += 1
        end
    end

    if plotbody
        # Find the label of the largest component (the outside)
        outside_label = argmax(component_sizes)
        # Everything that's not the outside and not already part of the body is a hole
        fill_mask = (labels .!= outside_label) .& fluid_mask
        # Combine the original body and the filled holes
        filled_body = body_mask .| fill_mask
        # Display the filled body
        GLMakie.image!(ax, filled_body; colormap=[:transparent, :black])
        # GLMakie.contour!(ax, μ₀; levels=[0.5], color=:red, linewidth=2)
    end

    # display(fig)


    # Record video
    @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
        while sim_time(sim)<t
            WaterLily.mom_step!(sim.flow,sim.pois)
            update!(p,sim)
        end
        notify!(v,p,sim.flow.Δt[end-1])
    end
end