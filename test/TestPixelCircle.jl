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

# set up circle example
circle(radius,center) = AutoBody((x,t) -> √sum(abs2, x .- center) - radius)
function PixelSimCircle(n=3*2^5,m=2^6;R=m/8,Re=200,ϵ=1)
    # replace with image stuff within the constructor
    circ_AB = circle(R,m/2-1)
    d = zeros(n+2,m+2) # this will be the simulation scalar array size
    WaterLily.measure_sdf!(d,circ_AB)
    circ_Pixel = PixelBody(WaterLily.μ₀.(d,ϵ)) # setting smooth weighted function

    # make simulation of same size and ϵ
    Simulation((n,m),(1,0),2R;body=circ_Pixel,ν=2R/Re,ϵ)
end

# run example
using Plots
sim = PixelSimCircle();
sim_gif!(sim;duration=20.,step=0.1,clims=(-5,5))


## sketchy real-time loop
# Δt = 0.1
# while
#     # async grab image stuff
#     # when_image_avail && sim.body = grab_new_image(src)
#     sim_step!(sim,time(sim)+Δt,remeasure=true)
#     abort_flag && break
# end