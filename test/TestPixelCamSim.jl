import Pkg
# Activate the project in the parent directory of this script (to use local WaterLily.jl)
 Pkg.activate(joinpath(@__DIR__, ".."))

# Pkg.activate("..")          # Use the project in WaterLily.jl/
# Pkg.activate(".")          # Use the project in WaterLily.jl/
# Pkg.develop(path="..")      # Register WaterLily as a dev package (one time only)

using WaterLily, StaticArrays, Plots, StatsBase
using NPZ  # For reading numpy files
try
    using CUDA
    CUDA.allowscalar(false)
catch e
    @warn "CUDA not available, running on CPU only." exception=e
end
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "Pathlines.jl", "src")) # For now import local version 
                                                                        # of Pathlines (Pathlines.jl/src/ needs to be in the 
                                                                        # same dir level as this root dir)
include(joinpath(@__DIR__, "plot_particles.jl"))  # Add module containing particle plotting functions
include(joinpath(@__DIR__, "sim_data_export.jl"))  # Add module for data export

# set up airfoil simulation from boolean mask
function PixelSimAirfoilFromMask(mask_file; Re=200, ϵ=1, LS=nothing, mem=Array)
    # Load the boolean mask from numpy file
    mask = npzread(mask_file)
    
    # Create PixelBody using the new mask constructor
    airfoil_pixel_body = WaterLily.PixelBody(mask; ϵ=ϵ, mem=mem)
    
    # Use provided characteristic length or estimate it
    if LS === nothing
        LS, _ = WaterLily.estimate_characteristic_length(airfoil_pixel_body, method="pca", plot_method=false)
    end
    
    n, m = size(airfoil_pixel_body.μ₀)
    
    # Create simulation
    Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem)
end


# Wrapper function for PyJulia interface
function run_simulation(mask_file, output_path, LS, Re, ϵ, t_sim, delta_t, verbose, sim_type, mem_str)
    """
    Wrapper function to run simulation from PyJulia using pre-computed mask.
    Returns 0 on success, 1 on failure.
    """
    try
        # Convert memory type
        if mem_str == "Array"
            mem = Array
        elseif mem_str == "CuArray"
            mem = CuArray
        else
            error("Unsupported mem type: $mem_str. Must be 'Array' or 'CuArray'.")
        end

        # Print settings
        println("===Running PyJulia Simulation===")
        println("Mask file: $mask_file")
        println("Output: $output_path")
        println("LS: $LS, Re: $Re, ϵ: $ϵ, t_sim: $t_sim, Δt: $delta_t")
        println("Simulation type: $sim_type, Memory: $mem_str")

        # Instantiate the PixelBody simulation from mask
        sim = PixelSimAirfoilFromMask(
            mask_file,
            Re=Re,
            ϵ=ϵ,
            LS=LS,
            mem=mem,
        );

        # Run the simulation
        if sim_type == "particles"
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
        elseif sim_type == "sim_only"
            println("Running simulation for data export...")
            run_simulation_export_data!(
                sim;
                t_i=0.01, duration=t_sim, Δt=delta_t,
                N_particles=2^14, life_particles=1e3,
                save_path=output_path,
                verbose=verbose,
                mem=mem,
            );
        else
            println("Running WaterLily.sim_gif!...")
            sim_gif!(sim; duration=t_sim, step=delta_t, clims=(-5,5), save_path=output_path, verbose=verbose)
        end

        println("✓ Simulation completed successfully")
        return 0
        
    catch e
        println("Simulation failed: $e")
        return 1
    end
end

function main()
    # Parse arguments passed down from Python script
    args = ARGS
    if length(args) < 2
        println("Usage: julia TestPixelCamSim.jl mask_file.npy output.gif")
        return
    end

    mask_file = args[1]
    output_path = args[2]
    LS = parse(Float64, args[3])
    Re = parse(Float64, args[4])
    ϵ = parse(Float64, args[5])
    t_sim = parse(Float64, args[6])
    delta_t = parse(Float64, args[7])
    verbose = parse(Bool, args[8])
    sim_type = args[9]
    mem_str = args[10]

    run_simulation(mask_file, output_path, LS, Re, ϵ, t_sim, delta_t, verbose, sim_type, mem_str)
end


# Only run main if this script is executed directly (not loaded via PyJulia)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# # image_path = "test/resources/airfoil.png"
# # image_path = "test/resources/airfoil_30_deg.png"
# image_path = "picture_sim_app/input/input.png"
# image_path = "picture_sim_app/input/input_red.png"
# # image_path = "picture_sim_app/input/input_blue.png"
# # image_path = "picture_sim_app/input/input_green.png"
# output_path = "picture_sim_app/output/particles.gif"

# threshold = 0.4
# diff_threshold = 0.2
# manual_mode = false
# force_invert_mask = false
# max_image_res=800
# t_sim=20.0
# delta_t=0.05
# Re=200
# ϵ=1
# verbose=true
# body_color="red"
# sim_type="particles"  # "particles" or "gif"
# mem=Array
# # mem=CuArray

# airfoil_pixel_body = WaterLily.PixelBody(
#     image_path,
#     ϵ=ϵ,
#     threshold=threshold,
#     diff_threshold=diff_threshold,
#     body_color="red",
#     max_image_res=max_image_res,
#     manual_mode=manual_mode,
#     force_invert_mask=force_invert_mask,
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
#  t_i=0.01, duration=t_sim, Δt=0.05,
#  N_particles=2^14, life_particles=1e3,
#  scale=5.0, minsize=0.01, width=0.05,
#  plotbody=true, 
#  save_path="picture_sim_app/output/particleplot.gif",
#  verbose=verbose,
#  mem=mem,
#  );

# t_i=0.01
# duration=20.
# Δt=0.05
# N_particles=2^14
# life_particles=1e3

# scale=5.0
# minsize=0.01
# width=0.05
# plotbody=true
# save_path="picture_sim_app/output/particleplot.gif"
# verbose=verbose
# mem=mem

# N=Int(N_particles)
# life=UInt(life_particles)
# t_f = duration

# p = Particles(N,sim.flow.σ;mem,life)
# v = ParticleViz(p, Δt, ; scale=scale, minsize=minsize, width=width)

# fig = GLMakie.Figure(; backgroundcolor = :gray30)
# ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
# GLMakie.scatter!(ax, v.opos; color = :white, markersize = v.omag, rotation = v.odir, marker = GLMakie.Circle)
# GLMakie.hidedecorations!(ax)

# # Plot the body region filled with black
# μ₀ = Array(sim.body.μ₀)
# body_mask = μ₀ .== 0.0  # Original solid pixels

# # For larger holes, use a different approach with label_components
# # First invert the mask so solid is false (0) and fluid is true (1)
# fluid_mask = .!body_mask

# # Label all connected components in the fluid
# labels = ImageMorphology.label_components(fluid_mask)

# # The largest component is the outside fluid region
# # All other components are holes inside the body
# component_sizes = zeros(Int, maximum(labels))
# for i in 1:length(labels)
#     if labels[i] > 0
#         component_sizes[labels[i]] += 1
#     end
# end

# # Find the label of the largest component (the outside)
# outside_label = argmax(component_sizes)
# # Everything that's not the outside and not already part of the body is a hole
# fill_mask = (labels .!= outside_label) .& fluid_mask
# # Combine the original body and the filled holes
# filled_body = body_mask .| fill_mask
# # Display the filled body
# GLMakie.image!(ax, filled_body; colormap=[:transparent, :black])
# # GLMakie.contour!(ax, μ₀; levels=[0.5], color=:red, linewidth=2)

# display(fig)


# # TODO: TRY TO RUN ONLY SIMULATION WITH NO VISUALIZATION
# # Record video
# @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
#     while sim_time(sim)<t
#         WaterLily.mom_step!(sim.flow,sim.pois)
#         update!(p,sim)
#     end
#     notify!(v,p,sim.flow.Δt[end-1])
#     verbose && println("tU/L=", round(t, digits=4),
#             ", Δt=", round(sim.flow.Δt[end], digits=3))
# end














# using ImageMorphology
# import GLMakie
# push!(LOAD_PATH, "../Pathlines.jl/src/" ) # For now import local version of Pathlines
# using Pathlines

# sim = Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem);

# N_particles=2^10
# life_particles=1e3

# t_i=0.01
# Δt=0.2
# duration=10.

# scale=1.0
# minsize=0.1
# width=1

# plotbody=true
# save_path="picture_sim_app/output/pathlines.mp4"


# N=Int(N_particles)
# life=UInt(life_particles)
# t_f = duration

# p = Particles(N,sim.flow.σ;mem,life)
# dat = tuple.(p.position⁰,p.position) |> Array   

# bgcolor = :black
# fig = GLMakie.Figure(; backgroundcolor = :gray30)
# ax = GLMakie.Axis(fig[1, 1]; autolimitaspect=1, backgroundcolor = :gray30)
# GLMakie.Box(fig,width=800,height=800;color=bgcolor)
# GLMakie.hidedecorations!(ax)


# # Plot the body region filled with black
# μ₀ = airfoil_pixel_body.μ₀
# body_mask = μ₀ .== 0.0  # Original solid pixels

# # For larger holes, use a different approach with label_components
# # First invert the mask so solid is false (0) and fluid is true (1)
# fluid_mask = .!body_mask

# # Label all connected components in the fluid
# labels = ImageMorphology.label_components(fluid_mask)

# # The largest component is the outside fluid region
# # All other components are holes inside the body
# component_sizes = zeros(Int, maximum(labels))
# for i in 1:length(labels)
#     if labels[i] > 0
#         component_sizes[labels[i]] += 1
#     end
# end

# if plotbody
#     # Find the label of the largest component (the outside)
#     outside_label = argmax(component_sizes)
#     # Everything that's not the outside and not already part of the body is a hole
#     fill_mask = (labels .!= outside_label) .& fluid_mask
#     # Combine the original body and the filled holes
#     filled_body = body_mask .| fill_mask
#     # Display the filled body
#     GLMakie.image!(ax, filled_body; colormap=[:transparent, :black])
#     # GLMakie.contour!(ax, μ₀; levels=[0.5], color=:red, linewidth=2)
# end

# display(fig)


# # # Record video
# # @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
# #     while sim_time(sim)<t
# #         WaterLily.mom_step!(sim.flow,sim.pois)
# #         update!(p,sim)
# #     end
# #     notify!(v,p,sim.flow.Δt[end-1])
# # end

# @time GLMakie.record(fig,save_path,t_i:Δt:t_f) do t
#     GLMakie.Box(fig,width=1600,height=1600,color=(bgcolor,0.2))
#     while sim_time(sim)<t
#         WaterLily.mom_step!(sim.flow,sim.pois)
#         update!(p,sim)
#         copyto!(dat,tuple.(p.position⁰,p.position))
#         GLMakie.linesegments!(ax,dat,linewidth=0.1,color=:white)
#     end
# end


# # image_path = "picture_sim_app/input/input.png"
# image_path = "picture_sim_app/input/input_red.png"

# # Load Image
# img = WaterLily.load(image_path);
# @show size(img);
# # display(heatmap(Array(img), color=:coolwarm, title="Raw image", aspect_ratio=:equal))

# # Downsize image if max_image_res is provided
# if !isnothing(max_image_res)
#     img = WaterLily.limit_resolution(img, max_image_res)
#     println("Image resized to $(size(img))")
# end

# # Validate the body_color parameter
# valid_colors = ["gray", "red", "green", "blue"];
# if body_color ∉ valid_colors
#     throw(ArgumentError("Unsupported solid color: $body_color. Supported colors are: $(join(valid_colors, ", "))."))
# end;


# # Ensure image is in RBG format
# img_rgb = RGB.(img);
# # Extract channels
# R = WaterLily.channelview(img_rgb)[1, :, :];
# G = WaterLily.channelview(img_rgb)[2, :, :];
# B = WaterLily.channelview(img_rgb)[3, :, :];

# # NOTE: Different cameras and lighting conditions cause the boolean logic when trying to distinguish solid from fluid to 
# # be inverted, depending on the final color hiearchy of the image. Since the padding function pad_to_pow2_with_ghost_cells
# # assume 1 is solid and 0 is fluid, the mask passed down to the padding function also needs to ahdere to this logic. This
# # means that depending on the color hierarchy, the mask logic changes based on the selected threshold values, and
# # the mask logic needs to be inverted. A smart body detection implemenation is used in the following lines to attempt
# # said automatic reversal.

# # Smart body detection uses channel hierarchy to determine masking logic between solid and fluid
# R_mean = WaterLily.mean(R);
# G_mean = WaterLily.mean(G);
# B_mean = WaterLily.mean(B);

# println("Channel means: R=$(round(R_mean, digits=3)), G=$(round(G_mean, digits=3)), B=$(round(B_mean, digits=3))");

# # Analyze channel hierarchy and relative differences between color channels
# R_vs_G_diff = R_mean - G_mean;
# R_vs_B_diff = R_mean - B_mean;
# total_color_range = maximum([maximum(R), maximum(G), maximum(B)]) - 
#                     minimum([minimum(R), minimum(G), minimum(B)]);

# # Logic to determine if mask inversion is nededed based on color hierarchy
# needs_inversion = false;
# if R_mean > G_mean && R_mean > B_mean
#     # Red is dominant channel
#     println("RED DOMINANT camera detected");
    
#     # Scale thresholds based on how much red dominates
#     red_dominance = min(R_vs_G_diff, R_vs_B_diff) / total_color_range;
#     println("Red dominance factor: $(round(red_dominance, digits=3))");
    
#     # Selects threshold and diff_threshold based on red dominance 
#     if red_dominance > 0.05  # Significant red dominance
#         threshold = 0.5 + red_dominance;  # Higher threshold for red-heavy cameras
#         diff_threshold = 0.05 + red_dominance * 0.5;  # Lower diff since red is already elevated
#     else
#         threshold = 0.45; # TODO: Still need to calibrate these two values
#         diff_threshold = 0.15;
#     end
#     needs_inversion = true;  # RED DOMINANT requires inversion (e.g. macbook camera)
    
# elseif G_mean > R_mean && B_mean > R_mean
#     # Red is lowest - likely Logitech-style with suppressed red values
#     println("RED SUPPRESSED camera detected");
    
#     # Scale thresholds based on how much red is suppressed
#     red_suppression = max(G_mean - R_mean, B_mean - R_mean) / total_color_range;
#     println("Red suppression factor: $(round(red_suppression, digits=3))");
    
#     threshold = 0.35 + red_suppression * 0.2  # Lower threshold for red-suppressed cameras
#     diff_threshold = 0.15 + red_suppression * 0.3  # Higher diff needed to detect red
#     needs_inversion = false;  # Logitech-style doesn't need inversion
    
# else
#     # Balanced channels - use adaptive thresholds based on overall range
#     println("BALANCED channels detected")
    
#     # Use the total dynamic range to scale thresholds
#     if total_color_range > 0.5
#         threshold = 0.4
#         diff_threshold = 0.2
#     else
#         # Lower dynamic range needs more sensitive detection
#         threshold = 0.3
#         diff_threshold = 0.1
#     end
#     needs_inversion = false  # Default behavior
# end;

# # Ensure thresholds are within reasonable bounds
# threshold = clamp(threshold, 0.2, 0.7);
# diff_threshold = clamp(diff_threshold, 0.05, 0.4);

# println("FINAL ADAPTIVE THRESHOLDS:")
# println("   threshold = $(round(threshold, digits=3))")
# println("   diff_threshold = $(round(diff_threshold, digits=3))")
# println("   needs_inversion = $needs_inversion")
# println("="^50)


# # TODO: The above logic was developed around the red color. However, other colors can be selected for the solid. For 
# # now, the automated mask inversion logic is only used for the color 'red', but might be useful later.
# if body_color == "red"
#     # Detect red pixels (flip logic for different hierarchies of color channels)
#     red_detected = (R .> threshold) .& ((R .- G) .> diff_threshold) .& ((R .- B) .> diff_threshold)
    
#     if needs_inversion
#         # RED DOMINANT: red_detected=true means solid, so mask should be true for solid
#         mask = red_detected  # 1 for solid (red), 0 for fluid
#         println("Applied mask = red_detected (no inversion)")
#     else
#         # RED SUPPRESSED: red_detected=true means solid, but we need 0 for solid to match padding
#         mask = .!red_detected  # 0 for solid (red), 1 for fluid
#         println("Applied mask = .!red_detected (inverted)")
#     end

# elseif body_color == "green"
#     green_detected = (G .> threshold) .& ((G .- R) .> diff_threshold) .& ((G .- B) .> diff_threshold)
#     # For now, use standard logic for green (can be enhanced later)
#     mask = .!green_detected
    
# elseif body_color == "blue"
#     blue_detected = (B .> threshold) .& ((B .- G) .> diff_threshold) .& ((B .- R) .> diff_threshold)
#     # For now, use standard logic for blue (can be enhanced later)
#     mask = .!blue_detected
# end


# mask = reverse(mask, dims=1)' # Transpose to align matrix indices with physical x-y


# # Pad with zeros (False/fluid) at the border
# mask_padded = WaterLily.pad_to_pow2_with_ghost_cells(mask);
# @show size(mask_padded);
# display(heatmap(Array(mask)', color=:coolwarm, title="Threshold mask", aspect_ratio=:equal))


# # Compute signed distance field
# sdf = Float32.(distance_transform(feature_transform(mask_padded)) .- distance_transform(feature_transform(.!mask_padded)));

# @show size(sdf)
# # Smooth volume fraction field
# μ₀_array = mem(Float32.(WaterLily.μ₀.(sdf, Float32(ϵ))));
# @show size(μ₀_array);

# # TODO: TEMP images for debugging
# display(heatmap(Array(mask_padded)', color=:coolwarm, title="Threshold mask (padded)", aspect_ratio=:equal))
# display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf)", aspect_ratio=:equal))
# display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf between ϵ=-1 and ϵ=1)", aspect_ratio=:equal, clims=(-ϵ, ϵ)))
