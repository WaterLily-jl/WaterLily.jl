using GLMakie

function volume_video!(sim,data_func;name="file.mp4",duration=1,step=0.1)
    # Set up viz data and figure
    dat = Node(data_func(sim))
    fig = volume(dat,colorrange=(π,4π),algorithm=:absorption,
                    backgroundcolor = RGBf0(0.9, 0.9, 0.9))

    # Run simulation and update figure data
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    record(fig,name,t) do tᵢ
        sim_step!(sim,tᵢ)
        dat[] = data_func(sim)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return sim,fig
end

function contour_video!(sim,data_func,geom_func;name="file.mp4",duration=1,step=0.1)
    # Set up viz data and figure
    fig = contour(geom_func(sim),levels=[0.5])
    dat = Node(data_func(sim))
    contour!(dat,levels=[-7,7],colormap=:balance,alpha=0.2,colorrange=[-7,7])

    # Run simulation and update figure data
    t₀ = round(sim_time(sim))
    t = range(t₀,t₀+duration;step)
    record(fig,name,t) do tᵢ
        sim_step!(sim,tᵢ)
        dat[] = data_func(sim)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return sim,fig
end
