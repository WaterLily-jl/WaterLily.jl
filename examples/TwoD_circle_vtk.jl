using WaterLily
using WriteVTK
using StaticArrays
function circle(p=4;Re=250,mem=Array,U=1)
    # Define simulation size, geometry dimensions, viscosity
    L=2^p
    center,r = SA[3L,3L,0], L/2
    ν = U*L/Re

    # make a body
    norm2(x) = √sum(abs2,x)
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        norm2(SA[x,y,0])-r
    end

    Simulation((8L,6L,16),(U,0,0),L;ν,body,mem)
end

# import CUDA
# @assert CUDA.functional()
# sim = circle(mem=CUDA.CuArray);
sim = circle();

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u |> Array;
pressure(a::Simulation) = a.flow.p |> Array;
body(a::Simulation) = (measure_sdf!(a.flow.σ, a.body); 
                       a.flow.σ |> Array;)
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            a.flow.σ |> Array;)
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
