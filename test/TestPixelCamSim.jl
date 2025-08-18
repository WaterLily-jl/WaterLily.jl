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
include(joinpath(@__DIR__, "run_sim.jl"))  # Simple simulation runner
include(joinpath(@__DIR__, "plot_heatmaps.jl"))  # Heatmap plotting functions

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
function run_simulation(
    mask_file,
    output_path,
    LS,
    Re,
    ϵ,
    t_sim,
    delta_t,
    verbose,
    mem_str,
    particle_plot_path=nothing,
    pressure_heatmap_path=nothing,
    vorticity_heatmap_path=nothing,

)
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
        println("particle_plot_path: $particle_plot_path")
        println("pressure_heatmap_path: $pressure_heatmap_path")
        println("vorticity_heatmap_path: $vorticity_heatmap_path")
        println("LS: $LS, Re: $Re, ϵ: $ϵ, t_sim: $t_sim, Δt: $delta_t")
        println("Memory: $mem_str")

        # Instantiate the PixelBody simulation from mask
        sim = PixelSimAirfoilFromMask(
            mask_file,
            Re=Re,
            ϵ=ϵ,
            LS=LS,
            mem=mem,
        );

        # Run the simulation. 
        println("Running PixelBody simulation...")
        sim_data = run_simulation_collect_data(
            sim;
            t_i=0.01, duration=t_sim, Δt=delta_t,
            N_particles=2^14, life_particles=1e3,
            # scale=5.0, minsize=0.01, width=0.05,
            verbose=verbose,
            mem=mem,
        )

        println("Simulation data collected: $(length(sim_data.positions)) frames")
        println("Body mask size: $(size(sim_data.body_mask))")
        println("Data ready for post-processing visualization")
    
        
        # Postprocessing (add optional plots). Options are:
            # "particles" (uses PathLines library)
            # "pressure_heatmap"
            # "vorticity_heatmap"

        # Create particle GIF
        if particle_plot_path !== nothing
            println("Creating particle GIF from simulation data...")
            create_particle_gif_from_data!(
                sim_data;
                scale=5.0, minsize=0.01, width=0.05,
                plotbody=true,
                save_path=particle_plot_path,
                verbose=verbose
            )
        end

        # Create pressure heatmap GIF
        if pressure_heatmap_path !== nothing
            println("Creating pressure heatmap GIF from simulation data...")
            create_heatmap_gif_from_data!(
                sim_data.pressure_field;
                save_path=pressure_heatmap_path,
                time_points=sim_data.time_points,
                verbose=verbose,
                plotbody=true,
                auto_clims=false,
                clims=(-1,1),
                invert_colors=true,
            )
        end


        # Create vorticity heatmap GIF
        if vorticity_heatmap_path !== nothing
            println("Creating vorticity heatmap GIF from simulation data...")
            create_heatmap_gif_from_data!(
                sim_data.vorticity_field;
                save_path=vorticity_heatmap_path,
                time_points=sim_data.time_points,
                verbose=verbose,
                clims=(-5,5),
            )
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
    # Parse optional plot paths (will set as nothing if length is too small)
    particle_plot_path = length(args) >= 3 ? args[3] : nothing
    pressure_heatmap_path = length(args) >= 3 ? args[4] : nothing
    vorticity_heatmap_path= length(args) >= 3 ? args[5] : nothing
    LS = parse(Float64, args[6])
    Re = parse(Float64, args[7])
    ϵ = parse(Float64, args[8])
    t_sim = parse(Float64, args[9])
    delta_t = parse(Float64, args[10])
    verbose = parse(Bool, args[11])
    mem_str = args[12]

    run_simulation(mask_file, output_path, LS, Re, ϵ, t_sim, delta_t, verbose, mem_str,
    particle_plot_path, pressure_heatmap_path, vorticity_heatmap_path)
end


# Only run main if this script is executed directly (not loaded via PyJulia)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
