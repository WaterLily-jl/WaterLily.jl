# new file stuff.
using WaterLily,StaticArrays

# # set up airfoil image example
function PixelSimAirfoil(Re=200, ϵ=1, mem=Array)
    # image_path = "test/resources/airfoil.png"
    image_path = "test/resources/airfoil_30_deg.png"
    airfoil_pixel_body = WaterLily.PixelBody(image_path,ϵ=ϵ) # setting smooth weighted function
    n, m = airfoil_pixel_body.μ₀.size
    LS = n / 10 # TODO: Arbitrary length scale of 10% of the domain, need to be able to set from image
    # make simulation of same size and ϵ
    Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ=ϵ, mem=Array)
end


# run example
using Plots
sim = PixelSimAirfoil();
# Initialize the simulation with GPU Array
# using CUDA
# sim = PixelSimAirfoil(3*2^6,2^7; mem=CuArray);
sim_gif!(sim;duration=20.,step=0.1,clims=(-5,5))



## sketchy real-time loop
# Δt = 0.1
# while
#     # async grab image stuff
#     # when_image_avail && sim.body = grab_new_image(src)
#     sim_step!(sim,time(sim)+Δt,remeasure=true)
#     abort_flag && break
# end