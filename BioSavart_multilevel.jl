using WaterLily,StaticArrays
import WaterLily: @loop,divisible,restrict!,permute,up,inside,slice,size_u

# 2D Multi-level Biot-Savart functions 
function MLArray(x)
    levels = [copy(x)]
    N = size(x)
    while all(N .|> divisible)
        N = @. 1+N÷2
        y = similar(x,N); fill!(y,0)
        push!(levels,y)
    end
    return levels
end
ml_restrict!(ml) = for l ∈ 2:lastindex(ml)
    restrict!(ml[l],ml[l-1])
end

function u_ω(i,x,ml_ω,dis=2.5f0)
    # initialize at bottom level
    ui = zero(eltype(x)); j = i%2+1
    l = lastindex(ml_ω)
    ω = ml_ω[l]
    R = inside(ω)
    dx = 2^(l-1)

    # loop levels
    while l>1
        # find Region close to x
        Rclose = inR(x/dx .-dis,R):inR(x/dx .+dis,R)

        # get contributions outside Rclose
        for I ∈ R
            !(I ∈ Rclose) && (ui += ω[I]*biotsavart(r(x,I,dx),j))
        end

        # move "up" one level within Rclose
        l -= 1
        R = first(up(first(Rclose))):last(up(last(Rclose)))
        ω = ml_ω[l]
        dx = 2^(l-1)
    end

    # top level contribution
    for I ∈ R
        ui += ω[I]*biotsavart(r(x,I,dx),j)
    end
    return ui
end
biotsavart(r,j) = Float32((3-2j)*r[j]/(2π*r'*r))
r(x,I,dx) = x-dx*loc(0,I,Float32)
inR(x,R) = clamp(CartesianIndex(round.(Int,x .+ 1.5f0)...),R)
Base.clamp(I::CartesianIndex,R::CartesianIndices) = CartesianIndex(clamp.(I.I,first(R).I,last(R).I))

function biotBC!(u,U,ml_ω)
    # fill top level with cell-centered ω₃ and restrict down to lower levels
    @inside ml_ω[1][I] = centered_ω₃(I,u)
    ml_restrict!(ml_ω)

    # fill BCs using multi-level biotsavart
    N,n = size_u(u)
    for j ∈ 1:n, i ∈ 1:n
        for s ∈ ifelse(i==j,(1,2,N[j]),(1,N[j]))
            @loop u[I,i] = u_ω(i,loc(i,I,Float32),ml_ω)+U[i] over I ∈ slice(N,s,j)
        end
    end
end
centered_ω₃(I,u) = permute((j,k)->WaterLily.∂(k,j,I,u),3)

# Check reconstruction on lamb dipole
using SpecialFunctions,ForwardDiff
function lamb_dipole(N;D=3N/4,U=1)
    β = 2.4394π/D
    C = -2U/(β*besselj0(β*D/2))
    function ψ(x,y)
        r = √(x^2+y^2)
        ifelse(r ≥ D/2, U*((D/2r)^2-1)*y, C*besselj1(β*r)*y/r)
    end
    center = SA[N/2,N/2]
    function uλ(i,xy)
        x,y = xy-center
        ifelse(i==1,ForwardDiff.derivative(y->ψ(x,y),y)+1+U,-ForwardDiff.derivative(x->ψ(x,y),x))
    end
    Simulation((N, N), (1,0), D; uλ) # Don't overwrite ghosts with BCs
end

begin
    sim = lamb_dipole(128); σ = sim.flow.σ; u = sim.flow.u;
    ml = MLArray(σ);
    @time biotBC!(u,sim.flow.U,ml);
    @assert sum(abs2,u-sim.flow.u⁰)/sim.L<2e-4
    @time BC!(u,sim.flow.U); #only x10 faster
end

# residual update from pressure (not matching direct calc yet)
function update_resid!(b,ml_ω)
    @inside ml_ω[1][I] = ω_from_p(I,b)
    ml_restrict!(ml_ω)

    # updat residual on boundaries
    N,n = size_u(u)
    for i ∈ 1:n
        @loop b.r[I] -= u_ω(i,loc(i,I,Float32),ml_ω) over I ∈ slice(N.-1,2,i,2)
        @loop b.r[I] += u_ω(i,loc(i,I+δ(i,I),Float32),ml_ω) over I ∈ slice(N.-1,N[i]-1,i,2)
    end
end 
function ω_from_p(I::CartesianIndex,b::AbstractPoisson)
    u(I,i) = @inbounds(-b.L[I,i]*WaterLily.∂(i,I,b.x))
    ∂(i,j,I,u) = (u(I+δ(j,I),i)+u(I+δ(j,I)+δ(i,I),i)
                 -u(I-δ(j,I),i)-u(I-δ(j,I)+δ(i,I),i))/4
    return permute((j,k)->∂(k,j,I,u),3)
end

# Check pressure solver convergence on circle
include("examples/TwoD_plots.jl")
circ(N;D=3N/4,U=1) = Simulation((N, N), (U,0), D; body=AutoBody((x,t)->√sum(abs2,x .- N/2)-D/2))
begin 
    sim = circ(128); a = sim.flow; b = sim.pois; a.u .= 0; ml = MLArray(a.σ);
    WaterLily.conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν);
    WaterLily.BDIM!(a);
    for i in 0:10
        biotBC!(a.u,a.U,ml);
        @inside a.σ[I] = WaterLily.div(I,a.u)
        @show L₂(a.σ),maximum(a.u)
        L₂(a.σ)<1e-3 && (@show i; break)
        sim.pois.x .= 0
        WaterLily.project!(a,b;itmx=1)
        update_resid!(b.levels[1],ml)
        @show L₂(b.levels[1].r) # not matching yet!
    end
    @assert abs(maximum(a.u)/2-1)<1e-2
end