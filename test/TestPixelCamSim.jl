import Pkg
# Activate the project in the parent directory of this script (to use local WaterLily.jl)
 Pkg.activate(joinpath(@__DIR__, ".."))

# Pkg.activate("..")          # Use the project in WaterLily.jl/
# Pkg.activate(".")          # Use the project in WaterLily.jl/
# Pkg.develop(path="..")      # Register WaterLily as a dev package (one time only)

using WaterLily, StaticArrays, Plots, CUDA
CUDA.allowscalar(false)  #Force disable scalar operations for CUDA
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "Pathlines.jl", "src")) # For now import local version 
                                                                        # of Pathlines (Pathlines.jl/src/ needs to be in the 
                                                                        # same dir level as this root dir)
include(joinpath(@__DIR__, "plot_particles.jl"))  # Add module containing particle plotting functions

# set up airfoil image example
function PixelSimAirfoil(image_path; Re=200, ϵ=1, threshold=0.5, diff_threshold=0.2, body_color="gray", max_image_res=800, mem=Array)

    airfoil_pixel_body = WaterLily.PixelBody(
        image_path,
        ϵ=ϵ,
        threshold=threshold,
        diff_threshold=diff_threshold,
        body_color=body_color,
        max_image_res=max_image_res,
        mem=mem,
    ) # setting smooth weighted function

    println("Press Enter to continue...")
    try
        readline()
    catch e
        @warn "No stdin available. Skipping pause." exception=e
    end

    LS, aoa = WaterLily.estimate_characteristic_length(airfoil_pixel_body, method="pca", plot_method=true);

    println("Estimated characteristic length: $(round(LS; digits=2))")
    println("Estimated AoA (deg): $(round(aoa; digits=2))")

    println("Press Enter to continue...")
    try
        readline()
    catch e
        @warn "No stdin available. Skipping pause." exception=e
    end
    
    n, m = size(airfoil_pixel_body.μ₀)

    # make simulation of same size and ϵ
    Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem)
end


function main()
    # Parse arguments passed down from Python script
    args = ARGS
    if length(args) < 2
        println("Usage: julia TestPixelCamSin.jl input.png output.gif")
        return
    end

    input_path = args[1]
    output_path = args[2]
    threshold = parse(Float64, args[3])
    diff_threshold = parse(Float64, args[4])
    body_color = args[5]
    max_image_res = parse(Int64, args[6])
    t_sim = parse(Float64, args[7])
    delta_t = parse(Float64, args[8])
    Re = parse(Float64, args[9])
    ϵ = parse(Float64, args[10])
    verbose = parse(Bool, args[11])
    sim_type =args[12]
    mem_str = args[13]

    if mem_str == "Array"
        mem = Array
    elseif mem_str == "CuArray"
        mem = CuArray
    else
        error("Unsupported mem type: $mem_str. Must be 'Array' or 'CuArray'.")
    end

    # Print settings
    println("Running simulation on: $input_path")
    println("threshold=$threshold")
    println("diff_threshold=$diff_threshold")
    println("body_color=$body_color")
    println("Maximum image resolution=$max_image_res")
    println("Simulation time=$t_sim s (Δt=$delta_t s)")
    println("Re=$Re")
    println("verbose=$verbose,sim_type=$sim_type")
    println("mem=$mem_str")

    # Instantiate the PixelBody simulation
    sim = PixelSimAirfoil(
        input_path,
        threshold=threshold,
        diff_threshold=diff_threshold,
        body_color=body_color,
        max_image_res=max_image_res,
        Re=Re,
        ϵ=ϵ,  # Default value for ϵ, can be adjusted
        mem=mem,
    );


    # Run the simulation
    if sim_type=="particles"

        println("Running particle simulation...")

        sim_gif_particles!(
        sim;
        t_i=0.01, duration=t_sim, Δt=delta_t,
        N_particles=2^14, life_particles=1e3,
        scale=5.0, minsize=0.01, width=0.05,
        plotbody=true, 
        verbose=verbose,
        save_path=output_path,
        mem=mem,
        );

    else
        println("Running WaterLily.sim_gif!...")
        sim_gif!(sim;duration=t_sim,step=delta_t,clims=(-5,5), save_path=output_path, verbose=verbose)

    end


    ## sketchy real-time loop
    # Δt = 0.1
    # while
    #     # async grab image stuff
    #     # when_image_avail && sim.body = grab_new_image(src)
    #     sim_step!(sim,time(sim)+Δt,remeasure=true)
    #     abort_flag && break
    # end

end

main()

# # image_path = "test/resources/airfoil.png"
# # image_path = "test/resources/airfoil_30_deg.png"
# # image_path = "picture_sim_app/input/input.png"
# image_path = "picture_sim_app/input/input_red.png"
# # image_path = "picture_sim_app/input/input_blue.png"
# # image_path = "picture_sim_app/input/input_green.png"

# ϵ=1
# threshold = 0.4
# diff_threshold = 0.2
# max_image_res=800
# Re=200
# body_color="red"
# # mem=Array
# mem=CuArray

# airfoil_pixel_body = WaterLily.PixelBody(
#     image_path,
#     ϵ=ϵ,
#     threshold=threshold,
#     diff_threshold=diff_threshold,
#     body_color="red",
#     max_image_res=max_image_res,
# ) # setting smooth weighted function

# # airfoil_pixel_body = WaterLily.PixelBody(image_path,ϵ=ϵ, threshold=threshold, diff_threshold=diff_threshold, max_image_res=max_image_res, 
# # body_color="red")
# # body_color="blue")
# # body_color="green")
# # body_color="gray")


# LS, aoa = WaterLily.estimate_characteristic_length(airfoil_pixel_body, method="pca", plot_method=true);

# println("Estimated characteristic length: $(round(LS; digits=2))")
# println("Estimated AoA (deg): $(round(aoa; digits=2))")


# n, m = size(airfoil_pixel_body.μ₀)


# # LS = n / 10 # TODO: Arbitrary length scale of 10% of the domain, need to be able to set from image

# # # REMEMBER TO RESTART SIMULATION BEFORE NEW PLOT
# sim = Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem);
 

# verbose=true
# output_path="picture_sim_app/output/output.gif"
# # sim_gif!(sim;duration=20.,step=0.05,clims=(-5,5), save_path=output_path, verbose=verbose);

# sim_gif_particles!(
#  sim;
#  t_i=0.01, duration=20., Δt=0.05,
#  N_particles=2^14, life_particles=1e3,
#  scale=5.0, minsize=0.01, width=0.05,
#  plotbody=true, 
#  save_path="picture_sim_app/output/particleplot.gif",
#  verbose=verbose,
#  mem=mem,
#  );

# # t_i=0.01
# # duration=20.
# # Δt=0.05
# # N_particles=2^14
# # life_particles=1e3

# # scale=5.0
# # minsize=0.01
# # width=0.05
# # plotbody=true
# # save_path="picture_sim_app/output/particleplot.gif"
# # verbose=verbose
# # mem=mem

# # N=Int(N_particles)
# # life=UInt(life_particles)
# # t_f = duration

# # p = Particles(N,sim.flow.σ;mem,life)
# # v = ParticleViz(p, Δt, ; scale=scale, minsize=minsize, width=width)

# # fig = GLMakie.Figure(; backgroundcolor = :gray30)
# # ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
# # GLMakie.scatter!(ax, v.opos; color = :white, markersize = v.omag, rotation = v.odir, marker = GLMakie.Circle)
# # GLMakie.hidedecorations!(ax)

# # # Plot the body region filled with black
# # μ₀ = Array(sim.body.μ₀)
# # body_mask = μ₀ .== 0.0  # Original solid pixels

# # # For larger holes, use a different approach with label_components
# # # First invert the mask so solid is false (0) and fluid is true (1)
# # fluid_mask = .!body_mask

# # # Label all connected components in the fluid
# # labels = ImageMorphology.label_components(fluid_mask)

# # # The largest component is the outside fluid region
# # # All other components are holes inside the body
# # component_sizes = zeros(Int, maximum(labels))
# # for i in 1:length(labels)
# #     if labels[i] > 0
# #         component_sizes[labels[i]] += 1
# #     end
# # end

# # # Find the label of the largest component (the outside)
# # outside_label = argmax(component_sizes)
# # # Everything that's not the outside and not already part of the body is a hole
# # fill_mask = (labels .!= outside_label) .& fluid_mask
# # # Combine the original body and the filled holes
# # filled_body = body_mask .| fill_mask
# # # Display the filled body
# # GLMakie.image!(ax, filled_body; colormap=[:transparent, :black])
# # # GLMakie.contour!(ax, μ₀; levels=[0.5], color=:red, linewidth=2)

# # display(fig)


# # # TODO: TRY TO RUN ONLY SIMULATION WITH NO VISUALIZATION
# # # Record video
# # @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
# #     while sim_time(sim)<t
# #         WaterLily.mom_step!(sim.flow,sim.pois)
# #         update!(p,sim)
# #     end
# #     notify!(v,p,sim.flow.Δt[end-1])
# #     verbose && println("tU/L=", round(t, digits=4),
# #             ", Δt=", round(sim.flow.Δt[end], digits=3))
# # end














# # using ImageMorphology
# # import GLMakie
# # push!(LOAD_PATH, "../Pathlines.jl/src/" ) # For now import local version of Pathlines
# # using Pathlines

# # sim = Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem);

# # N_particles=2^10
# # life_particles=1e3

# # t_i=0.01
# # Δt=0.2
# # duration=10.

# # scale=1.0
# # minsize=0.1
# # width=1

# # plotbody=true
# # save_path="picture_sim_app/output/pathlines.mp4"


# # N=Int(N_particles)
# # life=UInt(life_particles)
# # t_f = duration

# # p = Particles(N,sim.flow.σ;mem,life)
# # dat = tuple.(p.position⁰,p.position) |> Array   

# # bgcolor = :black
# # fig = GLMakie.Figure(; backgroundcolor = :gray30)
# # ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
# # GLMakie.Box(fig,width=800,height=800;color=bgcolor)
# # GLMakie.hidedecorations!(ax)


# # # Plot the body region filled with black
# # μ₀ = airfoil_pixel_body.μ₀
# # body_mask = μ₀ .== 0.0  # Original solid pixels

# # # For larger holes, use a different approach with label_components
# # # First invert the mask so solid is false (0) and fluid is true (1)
# # fluid_mask = .!body_mask

# # # Label all connected components in the fluid
# # labels = ImageMorphology.label_components(fluid_mask)

# # # The largest component is the outside fluid region
# # # All other components are holes inside the body
# # component_sizes = zeros(Int, maximum(labels))
# # for i in 1:length(labels)
# #     if labels[i] > 0
# #         component_sizes[labels[i]] += 1
# #     end
# # end

# # if plotbody
# #     # Find the label of the largest component (the outside)
# #     outside_label = argmax(component_sizes)
# #     # Everything that's not the outside and not already part of the body is a hole
# #     fill_mask = (labels .!= outside_label) .& fluid_mask
# #     # Combine the original body and the filled holes
# #     filled_body = body_mask .| fill_mask
# #     # Display the filled body
# #     GLMakie.image!(ax, filled_body; colormap=[:transparent, :black])
# #     # GLMakie.contour!(ax, μ₀; levels=[0.5], color=:red, linewidth=2)
# # end

# # display(fig)


# # # # Record video
# # # @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
# # #     while sim_time(sim)<t
# # #         WaterLily.mom_step!(sim.flow,sim.pois)
# # #         update!(p,sim)
# # #     end
# # #     notify!(v,p,sim.flow.Δt[end-1])
# # # end

# # @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
# #     GLMakie.Box(fig,width=1600,height=1600,color=(bgcolor,0.2))
# #     while sim_time(sim)<t
# #         WaterLily.mom_step!(sim.flow,sim.pois)
# #         update!(p,sim)
# #         copyto!(dat,tuple.(p.position⁰,p.position))
# #         GLMakie.linesegments!(ax,dat,linewidth=0.1,color=:white)
# #     end
# # end




