using GLMakie
function volume_video!(sim,dat,obs_update!;name="file.mp4",duration=1,step=0.1,framerate = 30, compression = 20)
    # Set up viz data and figure
    obs = ω_mag!(dat,sim) |> Observable;
    fig, _, _ = volume(obs, colorrange=(π,4π), algorithm=:absorption)
    
    # Run simulation and update figure data
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    record(fig, name, t; framerate, compression) do tᵢ
        sim_step!(sim,tᵢ)
        obs[] = obs_update!(dat,sim)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return fig
end
