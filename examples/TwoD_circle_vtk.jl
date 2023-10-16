using WaterLily

# parameters
L=2^4
Re=250
U =1

# make a body
radius, center = L/2, 3L
Body = AutoBody((x,t)->√sum(abs2, [x[1],x[2],0.0] .- [center,center,0.0]) - radius)
# 2D
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)
# 3D
# sim = Simulation((8L,6L,16),(U,0,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u
pressure(a::Simulation) = a.flow.p
body(a::Simulation) = a.flow.μ₀
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            a.flow.σ)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => body,
    "Vorticity_Z" => vorticity,
)# this maps what to write to the name in the file

# make the writer
wr = vtkWriter("TwoD_circle"; attrib=custom_attrib)

# intialize
t₀ = sim_time(sim)
duration = 10
tstep = 0.1

# step and write
@time for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # write data
    write!(wr, sim)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)
