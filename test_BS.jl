using WaterLily
using LinearAlgebra
include("examples/TwoD_plots.jl")
include("BiotSavart.jl")

"""
RankineVortex(i,xy,center,R,Γ)
"""
function RankineVortex(i, xy, center, R=4, Γ=1)
    xy = (xy .- 1.5 .- center)
    x,y = xy
    θ = atan(y,x)
    r = norm(xy)
    vθ =Γ/2π*(r<=R ? r/R^2 : 1/r)
    v = [-vθ*sin(θ),vθ*cos(θ)]
    return v[i]
end

# some definitons
U = 1
Re = 250
m, n = 2^6, 2^6
println("Field size: $m x $n")

# make a simulation
sim = Simulation((n,m), (U,0), m; ν=U*m/Re, T=Float64)
u = copy(sim.flow.u); u .= 0.0

# make a Rankine vortex
f(i,x) = RankineVortex(i,x,(m/2,m/2),10, 1)

# apply it to the flow
apply!(f, sim.flow.u)
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)

# full kernel integral
println("Computing velocity field standard")
@time BiotSavart!(u,sim.flow.σ)

# error
BC!(sim.flow.u,zeros(2))
BC!(u,zeros(2))
println("L₂-norm error u-velocity ", WaterLily.L₂(u[:,:,1].-sim.flow.u[:,:,1]))
println("L₂-norm error v-velocity ", WaterLily.L₂(u[:,:,2].-sim.flow.u[:,:,2]))


# test Fast Multipole Method for Biot-Savart
u .= 0.0
ml_BS = MultiLevelPoisson(sim.flow.p,sim.flow.μ₀,sim.flow.σ)
WaterLily.ml_ω!(ml_BS, sim.flow)
# compute the whole velocity field
N,n = WaterLily.size_u(u)
println("Computing velocity field FMM")
@time for i ∈ 1:n, Is ∈ WaterLily.inside_u(N,i)
    u[Is,i] = WaterLily.u_ω(i,loc(i,Is),ml_BS)
end

# error
BC!(sim.flow.u,zeros(2))
BC!(u,zeros(2))
println("L₂-norm error u-velocity ", WaterLily.L₂(u[:,:,1].-sim.flow.u[:,:,1]))
println("L₂-norm error v-velocity ", WaterLily.L₂(u[:,:,2].-sim.flow.u[:,:,2]))
