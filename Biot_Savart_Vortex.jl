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

# make a simulation
sim = Simulation((n,m), (U,0), m; ν=U*m/Re, T=Float64)
u = copy(sim.flow.u)

# make a Rankine vortex
f(i,x) = RankineVortex(i,x,(m/2,m/2),10, 1)

# apply it to the flow
apply!(f, sim.flow.u)

flood(sim.flow.u[:,:,1]; shift=(-0.5,-0.5))
flood(sim.flow.u[:,:,2]; shift=(-0.5,-0.5))

# compute vorticity
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
flood(sim.flow.σ; shift=(-0.5,-0.5))

# compute velocity, one cell at a time
N,n = WaterLily.size_u(u)
# for i ∈ 1:n, Is ∈ WaterLily.inside_u(N,i)
    # BiotSavart!(i,Is,u,sim.flow.σ)
# end
BiotSavart!(u,sim.flow.σ)

κ = copy(sim.flow.σ); κ .= 0.0
# compute the full kernel and then dot it
# for i ∈ 1:n, Is ∈ WaterLily.inside_u(N,i)
    # BiotSavartKernel!(i,Is,κ)
    # u[Is,i] = dot(κ,sim.flow.σ)
# end

# pos = zeros(size(u)...)
# for Is ∈ WaterLily.inside(κ)
#     pos[Is,:] = loc(0,Is)
# end
# # method where we pass it a empty kernel array a a position array
# for i ∈ 1:n, Is ∈ WaterLily.inside_u(N,i)
#     BiotSavart!(i,loc(i,Is),κ,pos,WaterLily.inside_u(N,i))
#     u[Is,i] = dot(κ,sim.flow.σ)
# end


# error
BC!(sim.flow.u,zeros(2))
BC!(u,zeros(2))
println("L₂-norm error u-velocity ", WaterLily.L₂(u[:,:,1].-sim.flow.u[:,:,1]))
println("L₂-norm error v-velocity ", WaterLily.L₂(u[:,:,2].-sim.flow.u[:,:,2]))

# check
flood(u[:,:,1].-sim.flow.u[:,:,1]; shift=(-0.5,-0.5))
flood(u[:,:,2].-sim.flow.u[:,:,2]; shift=(-0.5,-0.5))


# # show Kernel
# xs = collect(1:0.01:100)
# kernel = zeros(length(xs))
# ϵ = 1e-6
# for i in range(1,length(xs))
#     r = [xs[i],0.0]
#     rⁿ = norm(r)^2
#     kernel[i] =  r[1]/(2π*rⁿ+ϵ^2)
# end
# plot(xs, abs.(kernel), scale=:log10)