using WaterLily
using LinearAlgebra
using Plots; gr()
include("examples/TwoD_plots.jl")
include("BiotSavart.jl")

# function ∮(a::AbstractArray{T},j) where T
#     N,n = WaterLily.size_u(a)
#     return ∮(a,N,N[j],j)
# end

# function ∮(a,N,s,j) 
#     sm = 0.0
#     for I ∈ WaterLily.slice(N.-1,s,j,2) # remove ghosts
#         sm += a[I,j] 
#     end
#     return sm
# end

function BC_BiotSavart!(a,A,ω)
    N,n = WaterLily.size_u(a)
    @inside ω[I] = WaterLily.curl(3,I,a)
    area = ntuple(i->prod(N.-2)/(N[i]-2),n) # are of domain (without ghosts)
    for j ∈ 1:n, i ∈ 1:n
        k = i%2+1 # the only component not zero in the vorticity
        if i==j # Normal direction, Dirichlet
            if i==1
                for s ∈ (1,2,N[j])
                    for I ∈ WaterLily.slice(N,s,j)
                        a[I,i] = A[i]
                        @WaterLily.loop a[I,i] += K(i,I,k,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
                    end
                end
                # @WaterLily.loop a[I,i] = A[i] over I ∈ WaterLily.slice(N,N[j],j)
            else
                for s ∈ (1,2,N[j])
                    for I ∈ WaterLily.slice(N,s,j)
                        a[I,i] = A[i]
                        @WaterLily.loop a[I,i] += K(i,I,k,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
                    end
                end
            end
            # make global face flux zero
            # ∮u = (∮(a,N,N[j],j)-∮(a,N,2,j))/∮(ones((N...,j)),j)# mass flux imbalance in domain
            # @WaterLily.loop a[I,i] += ∮u/2 over I ∈ WaterLily.slice(N,2,j)
            # @WaterLily.loop a[I,i] -= ∮u/2 over I ∈ WaterLily.slice(N,N[j],j)
        else    # Tangential directions, Neumanns
            # @WaterLily.loop a[I,i] = a[I+δ(j,I),i] over I ∈ WaterLily.slice(N,1,j)
            # @WaterLily.loop a[I,i] = a[I-δ(j,I),i] over I ∈ WaterLily.slice(N,N[j],j)
            for s ∈ (1,N[j])
                for I ∈ WaterLily.slice(N,s,j)
                    a[I,i] = A[i]
                    @WaterLily.loop a[I,i] += K(i,I,k,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
                end
            end
        end
    end
end

# function BC_BiotSavart!(a,A,ω)
#     N,n = WaterLily.size_u(a)
#     @inside ω[I] = WaterLily.curl(3,I,a)
#     area = ntuple(i->prod(N.-2)/(N[i]-2),n) # are of domain (without ghosts)
#     for j ∈ 1:n, i ∈ 1:n
#         k = i%2+1 # the only component not zero in the vorticity
#         for s ∈ (1,2,N[j]-1,N[j])
#             for I ∈ WaterLily.slice(N,s,j)
#                 a[I,i] = A[i]
#                 @WaterLily.loop a[I,i] += K(i,I,k,J)*ω[J] over J ∈ WaterLily.inside_u(N,i)
#             end
#         end
#     end
# end

BC!(a,A,σ) = BC_BiotSavart!(a,A,σ)
# BC!(a,A,σ) = WaterLily.BC!(a,A)

@fastmath function mom_step_inner!(a::Flow,b::AbstractPoisson)
    a.u⁰ .= a.u; a.u .= 0
    # predictor u → u'
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); 
    BC!(a.u,a.U,a.σ)
    WaterLily.project!(a,b)
    BC!(a.u,a.U,a.σ)
    
    # corrector u → u¹
    WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
    WaterLily.BDIM!(a); a.u ./= 2; 
    BC!(a.u,a.U,a.σ)
    WaterLily.project!(a,b);
    BC!(a.u,a.U,a.σ)
    push!(a.Δt,WaterLily.CFL(a))
end


# some definitons
U = 1
Re = 250
m, n = 2^6, 2^7
println("$m x $n: ", prod((m,n)))
radius = 16
center = m/2 + 1.5

# make a circle body
body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
z = [radius*exp(im*θ) for θ ∈ range(0,2π,length=33)]

# make a simulation
sim = Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T=Float64)

# duration of the simulation
duration = 0.0
step = 0.1
t₀ = 0.0

# init plot
N,dim = WaterLily.size_u(sim.flow.u)
Px = 1; Py1 = 2; Py2=N[2]
l = @layout [ a ; b c ]
@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
p1 = contourf(axes(sim.flow.σ,1),axes(sim.flow.σ,2),clamp.(sim.flow.σ'*sim.L/sim.U,-5,5),
              linewidth=0,color=:RdBu_11,clims=(-5,5),levels=10, 
              aspect_ratio=:equal)
p2 = plot(sim.flow.u[Px,3:end-1,1],1:m-1,color=:blue,label="u",xlims=(-1,2))
p3 = plot(1:n-1,sim.flow.u[3:end-1,Py1,1],color=:blue,label="u b",ylims=(-1,2))
p = plot(p1,p2,p3,layout = l)
plot!(p[2],sim.flow.u[Px,3:end-1,2],1:m-1,color=:red,label="v")
plot!(p[3],1:n-1,sim.flow.u[3:end-1,Py1,2],color=:red,label="v b")
plot!(p[3],1:n-1,sim.flow.u[3:end-1,Py2,1],color=:blue,linestyle=:dot,label="u t")
plot!(p[3],1:n-1,-sim.flow.u[3:end-1,Py2,2],color=:red,linestyle=:dot,label="v t")

# @time anim = @animate for tᵢ in range(t₀,t₀+duration;step)

    # # update until time tᵢ in the background
    # t = sum(sim.flow.Δt[1:end-1])
    # while t < tᵢ*sim.L/sim.U

        # update flow
        # mom_step_inner!(sim.flow,sim.pois)
                
    #     # compute motion and acceleration 1DOF
    #     Δt = sim.flow.Δt[end]
    #     t += Δt
    # end

sim.flow.u⁰ .= sim.flow.u; sim.flow.u .= 0
# predictor u → u'
WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,ν=sim.flow.ν)
WaterLily.BDIM!(sim.flow); 
BC!(sim.flow.u,sim.flow.U,sim.flow.σ)
@inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
flood(sim.flow.σ)
for i in 1:1
    WaterLily.project!(sim.flow,sim.pois)
    BC!(sim.flow.u,sim.flow.U,sim.flow.σ)
end
@inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
flood(sim.flow.σ)


# WaterLily.project!(sim.flow,sim.pois)
# @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
# flood(sim.flow.σ)

# BC!(sim.flow.u,sim.flow.U,sim.flow.σ)
# @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
# flood(sim.flow.σ)


# # corrector u → u¹
# WaterLily.conv_diff!(sim.flow.f,sim.flow.u,sim.flow.σ,ν=sim.flow.ν)
# WaterLily.BDIM!(sim.flow); sim.flow.u ./= 2; 
# BC!(sim.flow.u,sim.flow.U,sim.flow.σ)
# WaterLily.project!(sim.flow,sim.pois);
# BC!(sim.flow.u,sim.flow.U,sim.flow.σ)
# push!(sim.flow.Δt,WaterLily.CFL(sim.flow))

# # plot vorticity
# @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)
# p[1][1][:z] = clamp.(sim.flow.σ*sim.L/sim.U,-5,5)

# p[2][1][:x] = sim.flow.u[Px,3:end-1,1]
# p[2][2][:x] = sim.flow.u[Px,3:end-1,2]

# p[3][1][:y] = sim.flow.u[3:end-1,Py1,1]
# p[3][2][:y] = sim.flow.u[3:end-1,Py1,2]
# p[3][3][:y] = sim.flow.u[3:end-1,Py2,1]
# p[3][4][:y] = -sim.flow.u[3:end-1,Py2,2]
# p
    # print time step
#     println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
# end
# gif(anim, "Biot_Savart_Circle.gif")


# l = @layout [ a b c ]
# p1 = contourf(axes(sim.flow.p,2),axes(sim.flow.p,1),sim.flow.p,aspect_ratio=:equal,legend=:none)
# p2 = contourf(axes(sim.flow.p,2),axes(sim.flow.p,1),sim.flow.u[:,:,1],aspect_ratio=:equal,legend=:none)
# p3 = contourf(axes(sim.flow.p,2),axes(sim.flow.p,1),sim.flow.u[:,:,2],aspect_ratio=:equal,legend=:none)
# p = plot(p1,p2,p3,layout = l)
# N, = WaterLily.size_u(sim.flow.u)
# println("Inflow  :", WaterLily.∮(sim.flow.u,N,2,2))
# println("Outflow :", WaterLily.∮(sim.flow.u,N,N[2],2))
# plot(sim.flow.u[:,2,2],label="v botton")
# plot!(-sim.flow.u[:,end,2],label="v top")
# plot!(1.0.-sim.flow.u[:,2,1],label="u bottom")
# plot!(1.0.-sim.flow.u[:,end,1],label="u top")

# plot(sim.flow.u[1,:,1],label="u inlet")
# plot!(sim.flow.u[2,:,1],label="u inlet")

# flood(sim.flow.u[:,:,1])
