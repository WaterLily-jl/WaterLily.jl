using WaterLily
using Plots; gr()
include("BiotSavart.jl")

# some definitons
U = 1
Re = 250
m, n = 2^8, 3*2^7
radius, center = 32, 128+1.5
Px, Py = 64, 28

# make a circle body
body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=33)]

# make a simulation
sim = Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T=Float64)

# duration of the simulation
duration = 10
step = 0.1
t₀ = 0.0

# N,_ = WaterLily.size_u(sim.flow.u)
u = copy(sim.flow.u)
N,_ = WaterLily.size_u(sim.flow.u)

# init plot
l = @layout [ a ; b c ]
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
BiotSavart!(WaterLily.slice(N,Px,1),u,sim.flow.σ)
p1 = contourf(axes(sim.flow.σ,1),axes(sim.flow.σ,2),clamp.(sim.flow.σ*sim.L/sim.U,-5,5)',
              linewidth=0,color=:RdBu_11,clims=(-5,5),levels=10, 
              aspect_ratio=:equal)
p2 = plot(sim.flow.u[Px,2:end-1,1],1:m,color=:blue,label="u",xlims=(-1,2))
p3 = plot(1:n,sim.flow.u[2:end-1,Py,1],color=:blue,label="u b",ylims=(-1,2))
p = plot(p1,p2,p3,layout = l)

plot!(p[1],[Px,Px],[1,256],color=:black,linestyle=:dash,legend=:none)
plot!(p[1],[1,n],[Py,Py],color=:black,linestyle=:dash,legend=:none)
# plot!(p[1],[1,n],[228,228],color=:black,linestyle=:dash,legend=:none)

plot!(p[2],sim.flow.u[Px,2:end-1,2],1:m,color=:red,label="v")
plot!(p[2],1.0.+u[Px,2:end-1,1],1:m,color=:blue,linestyle=:dashdot,label="u BS")
plot!(p[2],     u[Px,2:end-1,2],1:m,linestyle=:dashdot,color=:red,label="v BS")

BiotSavart!(WaterLily.slice(N,Py,2),u,sim.flow.σ)
BiotSavart!(WaterLily.slice(N,m-Py,2),u,sim.flow.σ)
plot!(p[3],1:n,sim.flow.u[2:end-1,Py,2],color=:red,label="v b")
plot!(p[3],1:n,1.0.+u[2:end-1,Py,1],color=:blue,linestyle=:dashdot,label="u b BS")
plot!(p[3],1:n,     u[2:end-1,Py,2],color=:red,linestyle=:dashdot,label="v b BS")
plot!(p[3],1:n,     u[2:end-1,m-Py,1],color=:blue,linestyle=:dot,label="u t BS")
plot!(p[3],1:n,     u[2:end-1,m-Py,2],color=:red,linestyle=:dot,label="v t BS")

@time anim = @animate for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # update flow
        mom_step!(sim.flow,sim.pois)
                
        # compute motion and acceleration 1DOF
        Δt = sim.flow.Δt[end]
        t += Δt
    end

    # plot vorticity
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
    p[1][1][:z] = clamp.(sim.flow.σ*sim.L/sim.U,-5,5)

    BiotSavart!(WaterLily.slice(N,Px,1),u,sim.flow.σ)
    p[2][1][:x] = sim.flow.u[Px,2:end-1,1]
    p[2][2][:x] = sim.flow.u[Px,2:end-1,2]
    p[2][3][:x] = 1.0.+u[Px,2:end-1,1]
    p[2][4][:x] =      u[Px,2:end-1,2]
    # mass imbalance
    ∮u = ∮(u,N,Px,1)

    BiotSavart!(WaterLily.slice(N,Py,2),u,sim.flow.σ)
    BiotSavart!(WaterLily.slice(N,m-Py,2),u,sim.flow.σ)
    p[3][1][:y] = sim.flow.u[2:end-1,Py,1]
    p[3][2][:y] = sim.flow.u[2:end-1,Py,2]
    p[3][3][:y] = 1.0.+u[2:end-1,Py,1]
    p[3][4][:y] =      u[2:end-1,Py,2]
    p[3][5][:y] = 1.0.+u[2:end-1,m-Py,1]
    p[3][6][:y] =      u[2:end-1,m-Py,2]
    # mass imbalance
    ∮v = ∮(u,N,Py,2)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    println("mass imbalance ",∮u,", ",∮v)
endo
gif(anim,"Biot_Savart_Profile.gif")
