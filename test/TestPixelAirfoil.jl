# new file stuff.
using WaterLily,StaticArrays

# TODO: Move to PixelBody in src
function WaterLily.measure!(a::Flow{2,T},body::PixelBody;t=zero(T),ϵ=1) where {T}
    a.V .= zero(T); a.μ₀ .= one(T); a.μ₁ .= zero(T)
    @assert size(a.p)==size(body.μ₀) # move to the constructor?
    WaterLily.apply!((i,x)->WaterLily.interp(x,body.μ₀),a.μ₀)
    BC!(a.μ₀,zeros(SVector{2,T}),false,a.perdir) # BC on μ₀, don't fill normal component yet
end

WaterLily.measure_sdf!(a::AbstractArray,body::PixelBody,t=0;kwargs...) = @warn "Can't do this yet"

# # set up airfoil image example
function PixelSimAirfoil(Re=200, ϵ=1)
    # image_path = "test/resources/airfoil.png"
    image_path = "test/resources/airfoil_30_deg.png"
    airfoil_pixel_body = WaterLily.PixelBody(image_path,ϵ=ϵ) # setting smooth weighted function
    n, m = airfoil_pixel_body.μ₀.size
    LS = n / 10 # TODO: Arbitrary length scale of 10% of the domain, need to be able to set from image
    # make simulation of same size and ϵ
    Simulation((n-2,m-2), (1,0), LS; body=airfoil_pixel_body, ν=LS/Re, ϵ)
end


# run example
using Plots
sim = PixelSimAirfoil();
sim_gif!(sim;duration=20.,step=0.1,clims=(-5,5))


## sketchy real-time loop
# Δt = 0.1
# while
#     # async grab image stuff
#     # when_image_avail && sim.body = grab_new_image(src)
#     sim_step!(sim,time(sim)+Δt,remeasure=true)
#     abort_flag && break
# end