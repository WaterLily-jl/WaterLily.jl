import Pkg
# Activate the project in the parent directory of this script (to use local WaterLily.jl)
 Pkg.activate(joinpath(@__DIR__, ".."))

# Pkg.activate("..")          # Use the project in WaterLily.jl/
# Pkg.activate(".")          # Use the project in WaterLily.jl/
# Pkg.develop(path="..")      # Register WaterLily as a dev package (one time only)

using WaterLily, StaticArrays, Plots, StatsBase
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

# image_path = "test/resources/airfoil.png"
# image_path = "test/resources/airfoil_30_deg.png"
image_path = "picture_sim_app/input/input.png"
image_path = "picture_sim_app/input/input_red.png"
# image_path = "picture_sim_app/input/input_blue.png"
# image_path = "picture_sim_app/input/input_green.png"
output_path = "picture_sim_app/output/particles.gif"

threshold = 0.4
diff_threshold = 0.2
max_image_res=800
t_sim=20.0
delta_t=0.05
Re=200
ϵ=1
verbose=true
body_color="red"
sim_type="particles"  # "particles" or "gif"
mem=Array
# mem=CuArray

airfoil_pixel_body = WaterLily.PixelBody(
    image_path,
    ϵ=ϵ,
    threshold=threshold,
    diff_threshold=diff_threshold,
    body_color="red",
    max_image_res=max_image_res,
) # setting smooth weighted function

# airfoil_pixel_body = WaterLily.PixelBody(image_path,ϵ=ϵ, threshold=threshold, diff_threshold=diff_threshold, max_image_res=max_image_res, 
# body_color="red")
# body_color="blue")
# body_color="green")
# body_color="gray")


LS, aoa = WaterLily.estimate_characteristic_length(airfoil_pixel_body, method="pca", plot_method=true);

println("Estimated characteristic length: $(round(LS; digits=2))")
println("Estimated AoA (deg): $(round(aoa; digits=2))")


n, m = size(airfoil_pixel_body.μ₀)


# LS = n / 10 # TODO: Arbitrary length scale of 10% of the domain, need to be able to set from image

# # REMEMBER TO RESTART SIMULATION BEFORE NEW PLOT
sim = Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem);
 

verbose=true
output_path="picture_sim_app/output/output.gif"
# sim_gif!(sim;duration=20.,step=0.05,clims=(-5,5), save_path=output_path, verbose=verbose);

sim_gif_particles!(
 sim;
 t_i=0.01, duration=t_sim, Δt=0.05,
 N_particles=2^14, life_particles=1e3,
 scale=5.0, minsize=0.01, width=0.05,
 plotbody=true, 
 save_path="picture_sim_app/output/particleplot.gif",
 verbose=verbose,
 mem=mem,
 );

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


# image_path = "picture_sim_app/input/input.png"
image_path = "picture_sim_app/input/input_red.png"

# Load Image
img = WaterLily.load(image_path);
@show size(img);
# display(heatmap(Array(img), color=:coolwarm, title="Raw image", aspect_ratio=:equal))

# Downsize image if max_image_res is provided
if !isnothing(max_image_res)
    img = WaterLily.limit_resolution(img, max_image_res)
    println("Image resized to $(size(img))")
end

# Validate the body_color parameter
valid_colors = ["gray", "red", "green", "blue"];
if body_color ∉ valid_colors
    throw(ArgumentError("Unsupported solid color: $body_color. Supported colors are: $(join(valid_colors, ", "))."))
end;


# Ensure image is in RBG format
img_rgb = RGB.(img)
# Extract channels
R = WaterLily.channelview(img_rgb)[1, :, :];
G = WaterLily.channelview(img_rgb)[2, :, :]  ;
B = WaterLily.channelview(img_rgb)[3, :, :];

# Use satistics to determine color domonance hierarchy fo automatic threshold selection (makes detection more robust for varying li)
R_mean = WaterLily.mean(R);
G_mean = WaterLily.mean(G);
B_mean = WaterLily.mean(B);

println("Channel means: R=$(round(R_mean, digits=3)), G=$(round(G_mean, digits=3)), B=$(round(B_mean, digits=3))")

# Analyze channel hierarchy and relative differences
R_vs_G_diff = R_mean - G_mean;
R_vs_B_diff = R_mean - B_mean;
total_color_range = WaterLily.maximum([WaterLily.maximum(R), WaterLily.maximum(G), WaterLily.maximum(B)]) - 
                   WaterLily.minimum([WaterLily.minimum(R), WaterLily.minimum(G), WaterLily.minimum(B)]);

println("R vs G difference: $(round(R_vs_G_diff, digits=3))")
println("R vs B difference: $(round(R_vs_B_diff, digits=3))")
println("Total color range: $(round(total_color_range, digits=3))")

# Decision logic based on relative channel relationships
if R_mean > G_mean && R_mean > B_mean
    # Red is dominant - likely MacBook-style with elevated red values
    println("RED DOMINANT camera detected")
    
    # Scale thresholds based on how much red dominates
    red_dominance = min(R_vs_G_diff, R_vs_B_diff) / total_color_range
    println("Red dominance factor: $(round(red_dominance, digits=3))")
    
    if red_dominance > 0.05  # Significant red dominance
        threshold = 0.5 + red_dominance  # Higher threshold for red-heavy cameras
        diff_threshold = 0.05 + red_dominance * 0.5  # Lower diff since red is already elevated
    else
        threshold = 0.45
        diff_threshold = 0.15
    end
    
elseif G_mean > R_mean && B_mean > R_mean
    # Red is lowest channel
    println("RED SUPPRESSED camera detected")
    
    # Scale thresholds based on how much red is suppressed
    red_suppression = max(G_mean - R_mean, B_mean - R_mean) / total_color_range
    println("Red suppression factor: $(round(red_suppression, digits=3))")
    
    threshold = 0.35 + red_suppression * 0.2  # Lower threshold for red-suppressed cameras
    diff_threshold = 0.15 + red_suppression * 0.3  # Higher diff needed to detect red
    
else
    # Balanced channels - use adaptive thresholds based on overall range
    println("⚖️  BALANCED channels detected - using adaptive thresholds")
    
    # Use the total dynamic range to scale thresholds
    if total_color_range > 0.5
        threshold = 0.4
        diff_threshold = 0.2
    else
        # Lower dynamic range needs more sensitive detection
        threshold = 0.3
        diff_threshold = 0.1
    end
end;

# Ensure thresholds are within reasonable bounds
threshold = clamp(threshold, 0.2, 0.7);
diff_threshold = clamp(diff_threshold, 0.05, 0.4);

println("Using the following threshold values:")
println("   threshold = $(round(threshold, digits=3))")
println("   diff_threshold = $(round(diff_threshold, digits=3))")
println("="^50)



# Step 2: Check individual conditions
condition1 = R .> threshold;
condition2 = (R .- G) .> diff_threshold;
condition3 = (R .- B) .> diff_threshold;
combined_red_detection = condition1 .& condition2 .& condition3;

# Step 3: Check some specific pixel values where red should be detected
red_pixels = findall(combined_red_detection);
if length(red_pixels) > 0
    sample_idx = red_pixels[1:min(5, length(red_pixels))]
    println("\n--- Sample RED pixels (first 5) ---")
    for idx in sample_idx
        println("Pixel $idx: R=$(R[idx]), G=$(G[idx]), B=$(B[idx]), R-G=$(R[idx]-G[idx]), R-B=$(R[idx]-B[idx])")
    end
end;

# Step 4: Final mask (note the .! negation)
mask = .!combined_red_detection;
println("\n--- Final Mask ---")
println("Mask: $(sum(mask)) pixels are TRUE (should be red areas, aka the solid)")
println("Mask: $(sum(.!mask)) pixels are FALSE (should be non-red area, aka the fluid)")

# Display intermediate results
display(heatmap(Array(condition1)', color=:coolwarm, title="Condition 1: R > $threshold", aspect_ratio=:equal))
display(heatmap(Array(condition2)', color=:coolwarm, title="Condition 2: R-G > $diff_threshold", aspect_ratio=:equal))
display(heatmap(Array(condition3)', color=:coolwarm, title="Condition 3: R-B > $diff_threshold", aspect_ratio=:equal))
display(heatmap(Array(combined_red_detection)', color=:coolwarm, title="Combined RED detection", aspect_ratio=:equal))
display(heatmap(Array(mask)', color=:coolwarm, title="Final mask (.! of red detection)", aspect_ratio=:equal))




mask = reverse(mask, dims=1)' # Transpose to align matrix indices with physical x-y
mask_padded = WaterLily.pad_to_pow2_with_ghost_cells(mask);
@show size(mask_padded);
display(heatmap(Array(mask)', color=:coolwarm, title="Threshold mask", aspect_ratio=:equal))


# Compute signed distance field
sdf = Float32.(distance_transform(feature_transform(mask_padded)) .- distance_transform(feature_transform(.!mask_padded)));

@show size(sdf)
# Smooth volume fraction field
μ₀_array = mem(Float32.(WaterLily.μ₀.(sdf, Float32(ϵ))));
@show size(μ₀_array);

# TODO: TEMP images for debugging
display(heatmap(Array(mask_padded)', color=:coolwarm, title="Threshold mask (padded)", aspect_ratio=:equal))
display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf)", aspect_ratio=:equal))
display(heatmap(Array(sdf)', color=:coolwarm, title="Signed Distance Field (sdf between ϵ=-1 and ϵ=1)", aspect_ratio=:equal, clims=(-ϵ, ϵ)))
