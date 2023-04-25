using GLMakie
function makie_video!(makie_plot,sim,dat,obs_update!;remeasure=false,name="file.mp4",duration=1,step=0.1,framerate=30,compression=20)
    # Set up viz data and figure
    obs = obs_update!(dat,sim) |> Observable;
    fig, _, _ = makie_plot(obs)
    
    # Run simulation and update figure data
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    record(fig, name, t; framerate, compression) do tᵢ
        sim_step!(sim,tᵢ;remeasure)
        obs[] = obs_update!(dat,sim)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return fig
end
