using WaterLily,StaticArrays

import WaterLily: divisible,restrict!
function MLArray(x)
    N = size(x)
    levels = [N]
    while all(N .|> divisible) && prod(N .-2) > 1000
        N = @. 1+N÷2
        push!(levels,N)
    end
    zeros_like_x(N) = (y = similar(x,N); fill!(y,0); y)
    return Tuple(zeros_like_x(N) for N in levels)
end
ml_restrict!(ml) = for l ∈ 2:lastindex(ml)
    restrict!(ml[l],ml[l-1])
end

import WaterLily: up,@loop,permute
function _u_ω(x,dis,l,R,biotsavart,s=0f0)
    while l>1
        # find Region close to x
        dx = 2f0^(l-1)
        Rclose = inR(x/dx .- dis,R):inR(x/dx .+ dis,R)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (s += biotsavart(r(x,I,dx),I,l))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(up(first(Rclose))):last(up(last(Rclose)))
    end

    # top level contribution
    for I ∈ R
        s += biotsavart(r(x,I),I,l)
    end
    return s
end
# u_ω(i,x::SVector{2},ω) = (l=lastindex(ω); _u_ω(x,7,l,inside(ω[l]),(r,I,l=1)->@fastmath @inbounds((2i-3)*ω[l][I]*r[i%2+1])/(2π*r'*r)))
u_ω(i,I::CartesianIndex{3},ω) = _u_ω(loc(i,I,Float32),1,lastindex(ω[1]),inside(ω[1][end]),
    @inline (r,I,l) -> permute((j,k)->@inbounds(ω[k][l][I]*r[j]),i)/√sum(abs2,r)^3)/Float32(4π)
r(x,I::CartesianIndex,dx=1) = x-dx*(SA_F32[I.I...] .- 1.5f0) # faster than loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

# Fill ghosts assuming potential flow outside the domain
# function biotBC!(u,U,ω)
#     fill_ω!(ω,u,Val(n)) # set-up ω
#     N,n = size_u(u)
#     for i ∈ 1:n
#         for s ∈ (2,N[i]) # Domain faces, biotsavart+background
#             @loop u[I,i] = u_ω(i,I,ω)+U[i] over I ∈ slice(N,s,i)
#         end
#         for j ∈ 1:n
#             j==i && continue
#             @loop u[I,j] = u[I+δ(i,I),j]-WaterLily.∂(j,CartesianIndex(I+δ(i,I),i),u) over I ∈ slice(N.-1,1,i,3)
#             @loop u[I,j] = u[I-δ(i,I),j]+WaterLily.∂(j,CartesianIndex(I,i),u) over I ∈ slice(N.-1,N[i],i,3)
#         end
#         # final Normal direction, incompresibility
#         @loop u[I,i] += div(I,u) over I ∈ slice(N.-1,1,i,3)
#     end
# end
# compute cell-centered ωᵢ and restrict down to lower levels
function _fill_ω!(ω,i,u)
    top = ω[1]
    @loop top[I] = centered_ω(i,I,u) over I ∈ inside(top,buff=2)
    ml_restrict!(ω)
end
fill_ω!(ω,u) = foreach(i->_fill_ω!(ω[i],i,u),1:3)
# fill_ω!(ω,u,::Val{2}) = _fill_ω!(ω,3,u)
centered_ω(i,I,u) = permute((j,k)->WaterLily.∂(k,j,I,u),i)

N=128
a = Flow((N,N,N),(0,0,1),f=Array,T=Float32); u=a.u;
ω = ntuple(i->MLArray(a.σ),3);
using CUDA
a_cu = Flow((N,N,N),(0,0,1),f=CuArray,T=Float32); u_cu = a_cu.u;
ω_cu = ntuple(i->MLArray(a_cu.σ),3);

using BenchmarkTools
@btime fill_ω!($ω,$u);
@btime CUDA.@sync fill_ω!($ω_cu,$u_cu);

biotBC!(u,ω) = ((N,n)=WaterLily.size_u(u); @loop u[I,n]=u_ω(n,I,ω) over I ∈ WaterLily.slice(N,2,n))
@btime biotBC!($u,$ω)
@btime CUDA.@sync biotBC!($u_cu,$ω_cu)
@show

# # Check reconstruction on lamb dipole
# using CUDA,SpecialFunctions,ForwardDiff
# CUDA.allowscalar(false)
# function lamb_dipole(N;D=3N/4,U=1,mem=Array)
#     β = 2.4394π/D
#     C = -2U/(β*besselj0(β*D/2))
#     function ψ(x,y)
#         r = √(x^2+y^2)
#         ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
#     end
#     center = SA[N/2,N/2]
#     function uλ(i,xy)
#         x,y = xy-center
#         ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
#     end
#     Simulation((N, N), (1,0), D; uλ, mem) # Don't overwrite ghosts with BCs
# end

# begin
#     sim = lamb_dipole(3*512,mem=Array); σ = sim.flow.σ; u = sim.flow.u;
#     ml = MLArray(σ);
#     @time CUDA.@sync biotBC!(u,sim.flow.U,ml); #10x slower on GPU 🤢
#     @assert sum(abs2,u-sim.flow.u⁰)/sim.L<2e-4
#     @time CUDA.@sync BC!(u,sim.flow.U);
# end

# function hill_vortex(N;D=3N/4,U=1,mem=Array)
#     function uλ(i,xyz)
#         q = xyz .- N/2; x,y,z = q; r = √(q'*q); θ = acos(z/r); ϕ = atan(y,x)
#         v_r = ifelse(2r<D,-1.5*(1-(2r/D)^2),1-(D/2r)^3)*U*cos(θ)
#         v_θ = ifelse(2r<D,1.5-3(2r/D)^2,-1-0.5*(D/2r)^3)*U*sin(θ)
#         i==1 && return sin(θ)*cos(ϕ)*v_r+cos(θ)*cos(ϕ)*v_θ
#         i==2 && return sin(θ)*sin(ϕ)*v_r+cos(θ)*sin(ϕ)*v_θ
#         cos(θ)*v_r-sin(θ)*v_θ
#     end
#     Simulation((N, N, N), (0,0,U), D; uλ, mem) # Don't overwrite ghosts with BCs
# end

# begin
#     sim = hill_vortex(128,mem=Array); σ = sim.flow.σ; u = sim.flow.u;
#     ω = ntuple(i->MLArray(σ),3);
#     @time CUDA.@sync biotBC!(u,sim.flow.U,ω);  #70x slower on GPU 🤢
#     @assert sum(abs2,u-sim.flow.u⁰)/sim.L^2<1e-4
#     @time CUDA.@sync BC!(u,sim.flow.U); 
# end

# # biotsavart momentum step
# function biot_mom_step!(a,b,ml;use_biotsavart=true)
#     a.u⁰ .= a.u; WaterLily.scale_u!(a,0)
#     # predictor u → u'
#     WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
#     WaterLily.BDIM!(a);
#     biot_project!(a,b,ml;use_biotsavart)
#     # corrector u → u¹
#     WaterLily.conv_diff!(a.f,a.u,a.σ,ν=a.ν)
#     WaterLily.BDIM!(a); WaterLily.scale_u!(a,0.5)
#     biot_project!(a,b,ml;use_biotsavart,w=0.5)
#     push!(a.Δt,WaterLily.CFL(a))
# end
# function biot_project!(a::Flow{n},ml_b::MultiLevelPoisson,ml_ω;w=1,use_biotsavart=true,log=false,tol=1e-3,itmx=32) where n
#     use_biotsavart ? biotBC!(a.u,a.U,ml_ω) : BC!(a.u,a.U) 
#     b = ml_b.levels[1]; @inside b.z[I] = WaterLily.div(I,a.u)
#     dt = w*a.Δt[end]; b.x .*= dt
#     WaterLily.residual!(b); r₂ = L₂(b); nᵖ = 0;
#     use_biotsavart && (update_resid!(b.r,b.L,b.x,ml_ω))
#     while r₂>tol && nᵖ<itmx
#         ml_ω[1] .= b.x
#         WaterLily.Vcycle!(ml_b)
#         WaterLily.smooth!(b)
#         b.ϵ .= b.x .- ml_ω[1]; ml_ω[1] .= 0
#         use_biotsavart && update_resid!(b.r,b.L,b.ϵ,ml_ω)
#         r₂ = L₂(b); nᵖ+=1
#         log && @show nᵖ,r₂
#     end
#     push!(ml_b.n,nᵖ)
#     for i ∈ 1:n
#         @loop a.u[I,i] -= b.L[I,i]*WaterLily.∂(i,I,b.x) over I ∈ inside(b.x)
#     end
#     use_biotsavart ? biotBC!(a.u,a.U,ml_ω) : BC!(a.u,a.U) 
#     b.x ./= dt
# end
# function update_resid!(r,L,ϵ,ml_ω)
#     # get pressure-induced vorticity
#     top = ml_ω[1]
#     @loop top[I] = ω_from_p(I,L,ϵ) over I ∈ inside(top,buff=2)
#     ml_restrict!(ml_ω)

#     # update residual on boundaries
#     N,n = size_u(L);
#     for i ∈ 1:n
#         @loop r[I] -= u_ω(i,loc(i,I,Float32),ml_ω) over I ∈ slice(N.-1,2,i,2)
#         @loop r[I] += u_ω(i,loc(i,I+δ(i,I),Float32),ml_ω) over I ∈ slice(N.-1,N[i]-1,i,2)
#     end

#     # correct global resid
#     res = sum(r)/sum(2 .* (N .- 2))
#     for i ∈ 1:n
#         @loop r[I] -= res over I ∈ slice(N.-1,2,i,2)
#         @loop r[I] -= res over I ∈ slice(N.-1,N[i]-1,i,2)
#     end
# end 
# @fastmath function ω_from_p(I::CartesianIndex,L,ϵ)
#     @inline u(I,i) = @inbounds(-L[I,i]*WaterLily.∂(i,I,ϵ))
#     @inline ∂(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
#                  -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
#     return permute((j,k)->∂(k,j,I,u),3)
# end

# # Check pressure solver convergence on circle
# include("examples/TwoD_plots.jl")
# circ(D,U=1,m=11D÷8;mem=Array) = Simulation((2D,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4,mem)
# sim = circ(256); ml = MLArray(sim.flow.σ);
# while sim_time(sim)<1.2
#     biot_mom_step!(sim.flow,sim.pois,ml)
#     sim_time(sim)%0.1<sim.flow.Δt[end]/sim.L && @show sim_time(sim),sim.flow.Δt[end],sim.pois.n[end]
# end
# flood(sim.flow.p|>Array,border=:none)
# @inside sim.flow.σ[I] = centered_ω₃(I,sim.flow.u)*sim.L/sim.U
# flood(sim.flow.σ|>Array,border=:none,legend=false,clims=(-25,25))

# using BenchmarkTools
# using CUDA
# circ(D,U=1,m=2D;mem=Array) = Simulation((2D,m), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- m/2)-D/2),ν=U*D/1e4,mem)
# sim = circ(2048;mem=CUDA.CuArray); a = sim.flow; b = sim.pois.levels[1]; ml = MLArray(a.σ);
# @btime CUDA.@sync WaterLily.conv_diff!($a.f,$a.u⁰,$a.σ,ν=$a.ν); # 80ms, 25ms
# @btime CUDA.@sync WaterLily.BDIM!($a); # 25ms, 14ms
# @btime CUDA.@sync biotBC!($a.u,$a.U,$ml) # 60ms, 530ms
# @btime CUDA.@sync update_resid!($b.r,$b.L,$b.x,$ml) # 80ms, 530ms
# @btime CUDA.@sync sum($b.r) # 2.6ms, 1ms
# @btime CUDA.@sync WaterLily.Vcycle!($sim.pois) # 140ms. 34ms
# @btime CUDA.@sync WaterLily.smooth!($b) # 230ms, 54ms