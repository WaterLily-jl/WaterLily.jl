import Pkg
Pkg.activate("..")          # Use the project in WaterLily.jl/
# Pkg.develop(path="..")      # Register WaterLily as a dev package (one time only)

using WaterLily,StaticArrays, Plots

# set up airfoil image example
function PixelSimAirfoil(image_path; Re=200, ϵ=1, threshold=0.5, mem=Array)
    # image_path = "test/resources/airfoil.png"
    # image_path = "test/resources/airfoil_30_deg.png"
    airfoil_pixel_body = WaterLily.PixelBody(image_path,ϵ=ϵ, threshold=threshold, max_image_res=800) # setting smooth weighted function
    
    println("Press Enter to continue...")
    try
        readline()
    catch e
        @warn "No stdin available. Skipping pause." exception=e
    end
    
    n, m = airfoil_pixel_body.μ₀.size
    LS = n / 10 # TODO: Arbitrary length scale of 10% of the domain, need to be able to set from image
    # make simulation of same size and ϵ
    Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=mem)
end


function main()
    args = ARGS
    if length(args) < 2
        println("Usage: julia TestPixelCamSin.jl input.png output.gif")
        return
    end

    input_path = args[1]
    output_path = args[2]
    graycsale_threshold = parse(Float64, args[3])

    println("Running simulation on: $input_path")
    println("Grayscale threshold=$graycsale_threshold")

    # sim = PixelSimAirfoil("test/resources/airfoil_30_deg.png");
    sim = PixelSimAirfoil(input_path, threshold=graycsale_threshold);
    # Initialize the simulation with GPU Array
    # using CUDA
    # sim = PixelSimAirfoil(3*2^6,2^7; mem=CuArray);
    sim_gif!(sim;duration=20.,step=0.1,clims=(-5,5), save_path=output_path)

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