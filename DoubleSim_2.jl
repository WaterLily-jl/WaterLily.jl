using WaterLily
include("examples/TwoD_plots.jl")
addbody(x,y;c=:grey) = Plots.plot!(Shape(x,y), c=c, legend=false)

# uses the b-field to impose all BC on the a field
function BC_new!(a,b)
    ϵ = 0.0
    N1,n1 = WaterLily.size_u(a)
    N2,n2 = WaterLily.size_u(b)
    ∂I = CartesianIndex(ntuple(i->Int64((N2[i]-N1[i])/2), n1))
    for j ∈ 1:n1, i ∈ 1:n1
        if i==j # Normal direction
            @WaterLily.loop a[I,i] = b[I+∂I,i]+ϵ over I ∈ WaterLily.slice(N1,1,j)
            @WaterLily.loop a[I,i] = b[I+∂I,i]+ϵ over I ∈ WaterLily.slice(N1,2,j)
            @WaterLily.loop a[I,i] = b[I+∂I,i]+ϵ over I ∈ WaterLily.slice(N1,N1[j],j)
        else # Tangential directions, Neumann
            # @WaterLily.loop a[I,i] = a[I+δ(j,I),i] over I ∈ WaterLily.slice(N1,1,j)
            # @WaterLily.loop a[I,i] = a[I-δ(j,I),i] over I ∈ WaterLily.slice(N1,N1[j],j)
            @WaterLily.loop a[I,i] = b[I+∂I,i]+ϵ over I ∈ WaterLily.slice(N1,1,j)
            @WaterLily.loop a[I,i] = b[I+∂I,i]+ϵ over I ∈ WaterLily.slice(N1,N1[j],j)
        end
    end
end


@fastmath function mom_step_inner!(a::Flow,b::AbstractPoisson,c)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); BC_new!(a.u,c)
    WaterLily.project!(a,b); BC_new!(a.u,c)
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); a.u ./= 2; BC_new!(a.u,c)
    b.x ./=2 #scaled pressure guess, no necessary
    WaterLily.project!(a,b); BC_new!(a.u,c)
    push!(a.Δt,WaterLily.CFL(a))
end


# some definitons
U = 1
Re = 250
m, n = 2^7, 3*2^6
n = n
m = m
println("$m x $n: ", prod((m,n)))
radius = m/4

# make a circle body
body1 = AutoBody((x,t)->√sum(abs2, x .- [n/4,m/2]) - radius)
body2 = AutoBody((x,t)->√sum(abs2, x .- [3n/4,m]) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=33)]

# make a simulation
sim_small = Simulation((n,m), (U,0), radius; ν=U*radius/Re, body=body1, T=Float32)
sim_large = Simulation((2n,2m), (U,0), radius; ν=U*radius/Re, body=body2, T=Float32)

# duration of the simulation
duration = 40
step = 0.1
t₀ = 0.0

@time @gif for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background
    t = sum(sim_large.flow.Δt[1:end-1])
    while t < tᵢ*sim_large.L/sim_large.U

        # update flow
        mom_step!(sim_large.flow,sim_large.pois)
        mom_step_inner!(sim_small.flow,sim_small.pois,sim_large.flow.u)
                
        # compute motion and acceleration 1DOF
        Δt = sim_large.flow.Δt[end]
        sim_small.flow.Δt[end] = Δt # synch time steps
        t += Δt
    end

    # plot vorticity
    # @inside sim_large.flow.σ[I] = WaterLily.curl(3,I,sim_large.flow.u)*sim_large.L/sim_large.U
    # flood(sim_large.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
    # addbody(real(z.+3n/4),imag(z.+im*m))
    # Plots.plot!([n/2,n/2,3n/2,3n/2,n/2],[m/2,3m/2,3m/2,m/2,m/2],color=:black,legend=:none,linestyle=:dash)

    @inside sim_small.flow.σ[I] = WaterLily.curl(3,I,sim_small.flow.u)*sim_small.L/sim_small.U
    flood(sim_small.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
    addbody(real(z.+n/4),imag(z.+im*m/2))
    println(sim_small.flow.u[10,10,1])

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim_large.flow.Δt[end],digits=3))
    println("          Δt=",round(sim_small.flow.Δt[end],digits=3))
end

# plot vorticity
# @inside sim_large.flow.σ[I] = WaterLily.curl(3,I,sim_large.flow.u)*sim_large.L/sim_large.U
# flood(sim_large.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
# body_plot!(sim_large)
# Plots.plot!([n/2,n/2,3n/2,3n/2,n/2],[m/2,3m/2,3m/2,m/2,m/2],color=:black,legend=:none,linestyle=:dash)

# # plot vorticity
# @inside sim_small.flow.σ[I] = WaterLily.curl(3,I,sim_small.flow.u)*sim_small.L/sim_small.U
# flood(sim_small.flow.σ; shift=(-0.5,-0.5), clims=(-5,5))
# body_plot!(sim_small)